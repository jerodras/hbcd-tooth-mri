"""
steps/s7_exposure_associations/exposure_associations.py
---------------------------------------------------------
Step 7 — Exposure associations with tooth age gap.

Python equivalent of evalTeeth_v2_toothage_determ_rel2.m (exposure models).

Models fitted (all with site random intercept):
  1. No BMI: gap ~ GA + race + sex + QC_PCA_1-5 + smoking_binary + (1|site)
  2. With BMI: gap ~ GA + BMI + race + sex + QC_PCA_1-5 + smoking_binary + (1|site)
  3. With BMI + HC: same + head_circumference + (1|site)

Note: smoking is binarised before modeling (pex_bm_assistv2_post__use_001 > 0).

Expected fixed-effect estimates from the reference analysis:
  sex (Male)   β ≈ 0.29  t ≈ 9.0
  race (Black) β ≈ 0.16  t ≈ 4.0
  GA           β ≈ −0.025 t ≈ −2.7
  smoking      β ≈ −0.12  t ≈ −2.4

Usage:
    python -m steps.s7_exposure_associations.exposure_associations --config config.yaml
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils.io import load_config, get_config_parser


def _extract_lme_params(res) -> dict:
    """Pull fixed-effects beta/SE/t/p from a statsmodels MixedLM result."""
    out = {}
    for param in res.fe_params.index:
        out[param] = {
            "beta": float(res.fe_params[param]),
            "se":   float(res.bse_fe[param]),
            "t":    float(res.tvalues[param]),
            "p":    float(res.pvalues[param]),
        }
    return out


def run_exposure_associations(csv_path: str, cfg: dict, output_path: str) -> dict:
    """Fit all exposure LME models and save results to JSON.

    MATLAB reference: evalTeeth_v2_toothage_determ_rel2.m lines 9-21

    Parameters
    ----------
    csv_path : str
        Path to tooth_with_gaps.csv (output of step 6).
    cfg : dict
        Loaded config.yaml dict.
    output_path : str
        Where to write exposure_association_stats.json.
    """
    exp_cfg   = cfg["exposure"]
    site_var  = exp_cfg["site_var"]
    gap_var   = exp_cfg["gap_var"]
    smoke_raw = exp_cfg["smoking_raw_var"]
    smoke_bin = exp_cfg["smoking_bin_var"]

    print(f"Loading {csv_path}...")
    tooth = pd.read_csv(csv_path)

    # Binarise smoking: MATLAB line 9 — tooth.pex_bm_assistv2_post__use_001_bin = col > 0
    if smoke_raw in tooth.columns:
        tooth[smoke_bin] = (tooth[smoke_raw] > 0).astype(float)
    else:
        print(f"Warning: {smoke_raw} not found. smoking_binary will be all NaN.")
        tooth[smoke_bin] = np.nan

    # Confirm site variable exists
    if site_var not in tooth.columns:
        print(f"Warning: site variable '{site_var}' not found. LME random effects unavailable.")

    all_results = {}

    # -----------------------------------------------------------------------
    # Model definitions (MATLAB: evalTeeth_v2_toothage_determ_rel2.m lines 12-21)
    # Each formula is the fixed-effects part; site random intercept is added below.
    # -----------------------------------------------------------------------
    ga_var  = "sed_basic_demographics_gestational_age_delivery"
    bmi_var = "pex_bm_health_preg__healthhx_011"
    hc_var  = "ph_ch_anthro_head_001__03"
    qc_vars = [f"t2_qc_pca_{i}" for i in range(1, 6)]

    def _build_formula(gap, preds):
        return f"{gap} ~ " + " + ".join(preds)

    common_preds = (
        [f"C(child_race_cat)", "C(child_sex_cat)"]
        + qc_vars
        + [smoke_bin]
    )

    models = {
        # MATLAB line 12 — no BMI, no HC
        "no_bmi": _build_formula(gap_var, [ga_var] + common_preds),
        # MATLAB line 16 — with BMI, no HC
        "with_bmi": _build_formula(gap_var, [ga_var, bmi_var] + common_preds),
        # MATLAB line 20 — with BMI + HC
        "with_bmi_hc": _build_formula(gap_var, [ga_var, bmi_var] + common_preds + [hc_var]),
    }

    for model_name, formula in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Formula: {formula}")
        # Collect required variables from formula (rough parse)
        req_vars = [gap_var, site_var] + [
            c for c in tooth.columns
            if c in formula and c != gap_var
        ]
        req_vars = list(dict.fromkeys(req_vars))  # deduplicate, preserve order
        df_lme   = tooth[req_vars].dropna()
        print(f"Complete cases: {len(df_lme)}")

        if len(df_lme) < 50:
            print("Skipping (too few complete cases).")
            continue
        if site_var not in df_lme.columns:
            print("Skipping (no site variable).")
            continue

        try:
            res = smf.mixedlm(formula, data=df_lme, groups=df_lme[site_var]).fit(reml=True)
            print(res.summary())
            all_results[model_name] = _extract_lme_params(res)
            all_results[model_name]["n"] = len(df_lme)
        except Exception as e:
            print(f"LME error ({model_name}): {e}")

    # -----------------------------------------------------------------------
    # Also run enet_corrected_gap → stepwise model for completeness
    # -----------------------------------------------------------------------
    for gap in ["stepwise_corrected_gap"]:
        if gap not in tooth.columns:
            continue
        for variant, formula_base in [
            ("no_bmi",      _build_formula(gap, [ga_var] + common_preds)),
            ("with_bmi",    _build_formula(gap, [ga_var, bmi_var] + common_preds)),
        ]:
            model_name = f"stepwise_{variant}"
            req_vars = [gap, site_var] + [
                c for c in tooth.columns
                if c in formula_base and c != gap
            ]
            req_vars = list(dict.fromkeys(req_vars))
            df_lme   = tooth[req_vars].dropna()
            if len(df_lme) < 50 or site_var not in df_lme.columns:
                continue
            try:
                res = smf.mixedlm(formula_base, data=df_lme, groups=df_lme[site_var]).fit(reml=True)
                all_results[model_name] = _extract_lme_params(res)
                all_results[model_name]["n"] = len(df_lme)
            except Exception as e:
                print(f"LME error ({model_name}): {e}")

    # -----------------------------------------------------------------------
    # Print compact validation table vs. MATLAB expected betas
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("VALIDATION vs. REFERENCE ANALYSIS")
    print(f"{'='*60}")
    expected = cfg["exposure"]["expected_betas"]
    if "with_bmi" in all_results:
        params = all_results["with_bmi"]
        print(f"{'Parameter':<40} {'β':>8} {'t':>8}  {'MATLAB β':>10} {'MATLAB t':>10}")
        print("-" * 80)
        matlab_map = {
            "sex_male":   ("C(child_sex_cat)[T.Male]",    expected.get("sex_male",   {})),
            "race_black": ("C(child_race_cat)[T.Black]",  expected.get("race_black", {})),
            "ga":         (ga_var,                         expected.get("ga",         {})),
            "smoking":    (smoke_bin,                      expected.get("smoking",    {})),
        }
        for label, (param_key, exp_vals) in matlab_map.items():
            # Try exact match, then partial
            val = params.get(param_key, None)
            if val is None:
                for k in params:
                    if param_key.lower() in k.lower():
                        val = params[k]
                        break
            if val:
                print(
                    f"{param_key:<40} {val['beta']:>8.3f} {val['t']:>8.2f}"
                    f"  {exp_vals.get('beta', '?'):>10}  {exp_vals.get('t', '?'):>10}"
                )

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {output_path}")
    return all_results


def main():
    parser = get_config_parser("Step 7: Exposure associations")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    root = Path(args.config).resolve().parent

    in_path  = root / cfg["data"]["outputs"]["tooth_with_gaps"]
    out_path = root / cfg["data"]["outputs"]["exposure_stats"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_exposure_associations(str(in_path), cfg, str(out_path))


if __name__ == "__main__":
    main()
