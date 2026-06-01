"""
steps/s8_eruption_prediction/eruption_prediction.py
-----------------------------------------------------
Step 8 — Eruption prediction.

Python equivalent of evalTeeth_v2_toothage_determ_rel2.m (eruption models).

Methodological note: the reference MATLAB analysis used fitlme
(linear mixed models) for both variables 009 and 010.

Models:
  Parsimonious (MATLAB lines 23-24):
    009 ~ enet_corrected_gap + (1|site)
    010 ~ enet_corrected_gap + adjusted_age + (1|site)

  Full covariate model (MATLAB lines 26-27):
    009 ~ GA + BMI + race + sex + QC_PCA_1-5 + smoking_binary + (1|site)
    010 ~ GA + BMI + race + sex + QC_PCA_1-5 + smoking_binary + (1|site)

Exclusion: remove rows where var_009 or var_010 ∈ {777, 999}.

Expected fixed-effect estimates from the reference analysis:
  eruption_009  β ≈ −0.29  t ≈ −2.86
  teeth_010     β ≈  0.90  t ≈  4.09

Usage:
    python -m steps.s8_eruption_prediction.eruption_prediction --config config.yaml
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils.io import load_config, get_config_parser


def _extract_lme_params(res) -> dict:
    out = {}
    for param in res.fe_params.index:
        out[param] = {
            "beta": float(res.fe_params[param]),
            "se":   float(res.bse_fe[param]),
            "t":    float(res.tvalues[param]),
            "p":    float(res.pvalues[param]),
        }
    return out


def run_eruption_predictions(csv_path: str, cfg: dict, output_path: str) -> dict:
    """Fit eruption LME models and save results.

    MATLAB reference: evalTeeth_v2_toothage_determ_rel2.m lines 23-27

    Parameters
    ----------
    csv_path : str
        Path to tooth_with_gaps.csv (output of step 6).
    cfg : dict
        Loaded config.yaml dict.
    output_path : str
        Where to write eruption_association_stats.json.
    """
    er_cfg       = cfg["eruption"]
    site_var     = er_cfg["site_var"]
    adj_age_var  = er_cfg["adjusted_age_var"]
    var_009      = er_cfg["var_009"]
    var_010      = er_cfg["var_010"]
    exclude_vals = er_cfg["exclude_codes"]
    gap_var      = "enet_corrected_gap"

    smoke_bin = cfg["exposure"]["smoking_bin_var"]
    ga_var    = "sed_basic_demographics_gestational_age_delivery"
    bmi_var   = "pex_bm_health_preg__healthhx_011"
    qc_vars   = [f"t2_qc_pca_{i}" for i in range(1, 6)]
    smoke_raw = cfg["exposure"]["smoking_raw_var"]

    print(f"Loading {csv_path}...")
    tooth = pd.read_csv(csv_path)

    # Binarise smoking if not already present
    if smoke_bin not in tooth.columns and smoke_raw in tooth.columns:
        tooth[smoke_bin] = (tooth[smoke_raw] > 0).astype(float)

    all_results = {}

    # -----------------------------------------------------------------------
    # Helper: filter and run LME
    # -----------------------------------------------------------------------
    def _run_lme(outcome_var, formula, label, tooth_df):
        if outcome_var not in tooth_df.columns:
            print(f"  {label}: '{outcome_var}' not found. Skipping.")
            return None
        df = tooth_df.copy()
        # MATLAB: exclude 777, 999
        df = df[~df[outcome_var].isin(exclude_vals)].dropna(
            subset=[outcome_var, site_var, gap_var]
        )
        print(f"\n  {label} — n={len(df)}")
        if len(df) < 20:
            print(f"  Too few cases ({len(df)}). Skipping.")
            return None
        try:
            res = smf.mixedlm(formula, data=df, groups=df[site_var]).fit(reml=True)
            print(res.summary())
            params = _extract_lme_params(res)
            params["n"] = len(df)
            return params
        except Exception as e:
            print(f"  LME error: {e}")
            return None

    # -----------------------------------------------------------------------
    # Parsimonious models
    # MATLAB lines 23-24:
    #   fitlme(tooth, 'ph_cg_ecls__medhist_009 ~ enet_corrected_gap + (1|site)')
    #   fitlme(tooth, 'ph_cg_ecls__medhist_010 ~ enet_corrected_gap + adjusted_age + (1|site)')
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARSIMONIOUS MODELS")
    print("=" * 60)

    f009_pars = f"{var_009} ~ {gap_var}"
    r = _run_lme(var_009, f009_pars, "009 parsimonious", tooth)
    if r:
        all_results["009_parsimonious"] = r

    f010_pars = f"{var_010} ~ {gap_var} + {adj_age_var}"
    r = _run_lme(var_010, f010_pars, "010 parsimonious", tooth)
    if r:
        all_results["010_parsimonious"] = r

    # -----------------------------------------------------------------------
    # Full covariate models
    # MATLAB lines 26-27 (similar formula to exposure models)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FULL COVARIATE MODELS")
    print("=" * 60)

    full_preds = (
        [ga_var, bmi_var, "C(child_race_cat)", "C(child_sex_cat)"]
        + qc_vars
        + [smoke_bin]
    )
    full_pred_str = " + ".join(full_preds)

    f009_full = f"{var_009} ~ {full_pred_str}"
    r = _run_lme(var_009, f009_full, "009 full", tooth)
    if r:
        all_results["009_full"] = r

    f010_full = f"{var_010} ~ {full_pred_str}"
    r = _run_lme(var_010, f010_full, "010 full", tooth)
    if r:
        all_results["010_full"] = r

    # -----------------------------------------------------------------------
    # Validation vs. MATLAB expected betas
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION vs. REFERENCE ANALYSIS")
    print("=" * 60)
    expected = cfg["eruption"]["expected_betas"]
    print(f"{'Model':<20} {'β':>8} {'t':>8}  {'MATLAB β':>10} {'MATLAB t':>10}")
    print("-" * 65)

    for result_key, exp_key, label in [
        ("009_parsimonious", "eruption_009",   "009 gap β"),
        ("010_parsimonious", "teeth_count_010", "010 gap β"),
    ]:
        if result_key in all_results:
            params = all_results[result_key]
            val    = params.get(gap_var) or params.get(adj_age_var)
            # get gap_var specifically
            val = params.get(gap_var)
            if val:
                exp_b = expected.get(exp_key, {}).get("beta", "?")
                exp_t = expected.get(exp_key, {}).get("t",    "?")
                print(
                    f"{label:<20} {val['beta']:>8.3f} {val['t']:>8.2f}"
                    f"  {exp_b:>10}  {exp_t:>10}"
                )

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {output_path}")
    return all_results


def main():
    parser = get_config_parser("Step 8: Eruption prediction")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    root = Path(args.config).resolve().parent

    in_path  = root / cfg["data"]["outputs"]["tooth_with_gaps"]
    out_path = root / cfg["data"]["outputs"]["eruption_stats"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_eruption_predictions(str(in_path), cfg, str(out_path))


if __name__ == "__main__":
    main()
