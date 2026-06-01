"""
steps/s5_merge_and_qc/merge_data.py
-------------------------------------
Step 5b — Merge wide morphology table with HBCD phenotypic data.

Python equivalent of mergeTables_rel2.m.

Merge order (matches MATLAB script exactly):
  1. Demographics (sed_basic_demographics) + MRI QC (img_mriqc_T2w)
     - PCA on QC numeric columns → t2_qc_pca_1..5
     - Filter to V02 sessions
  2. Anthro (ph_ch_anthro) — V02
  3. EPDS (pex_bm_epds) — V01
  4. BMI (pex_bm_health_preg__healthhx) — V01
  5. Smoking (pex_bm_assistv2) — V02
  6. Chronic conditions / diabetes (pex_bm_health_preg__chroncond) — V01
  7. Gestational diabetes (pex_bm_healthv2_preg) — V02
  8. PTSD (pex_bm_str__ptsd) — V02
  9. Eruption / medical history (ph_cg_ecls__medhist)

Then:
  - Compute pma_wks = candidate_age_years * 52 + GA_delivery; filter pma_wks < 48
  - Create categorical columns: child_race_cat, child_ethnicity_cat, child_sex_cat
  - Winsorize weight and head circumference at ±3 SD
  - Merge final_wide_table (wide morphology stats) last

Usage:
    python -m steps.s5_merge_and_qc.merge_data --config config.yaml
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.io import load_config, get_config_parser


def _read_tsv(path: str, id_cols: list | None = None) -> pd.DataFrame:
    """Read a TSV with participant_id and session_id forced to str."""
    dtype = {c: str for c in (id_cols or ["participant_id", "session_id"])}
    return pd.read_csv(path, sep="\t", dtype=dtype)


def merge_demographics(wide_morph_path: str, cfg: dict, root: Path, output_path: str) -> pd.DataFrame:
    """Merge wide morphology table with all HBCD phenotypic tables.

    MATLAB reference: mergeTables_rel2.m
    """
    merge_cfg  = cfg["merge"]
    pheno_cfg  = cfg["data"]["phenotype"]
    V01        = merge_cfg["session_v01"]   # 'V01'
    V02        = merge_cfg["session_v02"]   # 'V02'

    print(f"Loading morphology data from {wide_morph_path}...")
    tooth = pd.read_csv(wide_morph_path)

    # -----------------------------------------------------------------------
    # 1. Demographics + MRI QC → PCA → outer join → filter V02
    # MATLAB: pca(tmpqc_num) → score(:,1:5); outerjoin(demos,qc); filter V02
    # -----------------------------------------------------------------------
    demos = _read_tsv(str(root / pheno_cfg["demographics"]))
    qc    = _read_tsv(str(root / pheno_cfg["mriqc"]))

    qc_v02 = qc[qc["session_id"].str.contains(V02, na=False)].copy()
    # MATLAB uses columns 4:end for PCA (drop participant_id, session_id, first numeric col)
    qc_numeric = qc_v02.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    qc_num_no_na = qc_numeric.dropna()
    pca = PCA(n_components=5)
    pca_scores = pca.fit_transform(qc_num_no_na)
    pca_df = pd.DataFrame(
        pca_scores,
        columns=[f"t2_qc_pca_{i}" for i in range(1, 6)],
        index=qc_num_no_na.index,
    )
    qc_v02 = qc_v02.join(pca_df)

    # MATLAB: outerjoin(demos_only, qc_only, Keys={participant_id, session_id})
    t2_demos = pd.merge(demos, qc_v02, on=["participant_id", "session_id"], how="outer")
    t2_demos = t2_demos[t2_demos["session_id"].str.contains(V02, na=False)]

    tooth = pd.merge(tooth, t2_demos.drop_duplicates(subset=["participant_id"]),
                     on="participant_id", how="inner")

    # -----------------------------------------------------------------------
    # 2. Anthropometry (V02)
    # MATLAB: anthro(ses-V02, [participant_id, len, head, wei])
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["anthro"]
    if p.exists():
        anthro = _read_tsv(str(p))
        anthro = anthro[anthro["session_id"].str.contains(V02, na=False)]
        cols = ["participant_id", "ph_ch_anthro_len_001__03",
                "ph_ch_anthro_head_001__03", "ph_ch_anthro_wei_001__03"]
        cols = [c for c in cols if c in anthro.columns]
        tooth = pd.merge(tooth, anthro[cols].drop_duplicates(subset="participant_id"),
                         on="participant_id", how="left")

    # -----------------------------------------------------------------------
    # 3. EPDS (V01)
    # MATLAB: epds(ses-V01, [participant_id, pex_bm_epds_total_score])
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["epds"]
    if p.exists():
        epds = _read_tsv(str(p))
        epds = epds[epds["session_id"].str.contains(V01, na=False)]
        cols = ["participant_id", "pex_bm_epds_total_score"]
        cols = [c for c in cols if c in epds.columns]
        tooth = pd.merge(tooth, epds[cols].drop_duplicates(subset="participant_id"),
                         on="participant_id", how="left")

    # After EPDS join, filter to V02 sessions only (matches MATLAB line 51)
    tooth = tooth[tooth["session_id"].str.contains(V02, na=False)]

    # -----------------------------------------------------------------------
    # 4. BMI (V01) — exclude notvars
    # MATLAB: bmi(ses-V01, bmi_vars) minus bmi_notvars
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["bmi"]
    if p.exists():
        bmi      = _read_tsv(str(p))
        bmi      = bmi[bmi["session_id"].str.contains(V01, na=False)]
        bmi_cols = ["participant_id",
                    "pex_bm_health_preg__healthhx_011",
                    "pex_bm_health_preg__healthhx__preghx_001"]
        notvars  = merge_cfg["bmi_notvars"]
        bmi_cols = [c for c in bmi_cols if c in bmi.columns and c not in notvars]
        tooth = pd.merge(tooth, bmi[bmi_cols].drop_duplicates(subset="participant_id"),
                         on="participant_id", how="left")

    tooth = tooth[tooth["session_id"].str.contains(V02, na=False)]

    # -----------------------------------------------------------------------
    # 5. Smoking (V02)
    # MATLAB: smoking(ses-V02, [participant_id, pex_bm_assistv2_post__use_001])
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["smoking"]
    if p.exists():
        smoking = _read_tsv(str(p))
        smoking = smoking[smoking["session_id"].str.contains(V02, na=False)]
        cols    = ["participant_id", "pex_bm_assistv2_post__use_001"]
        cols    = [c for c in cols if c in smoking.columns]
        tooth   = pd.merge(tooth, smoking[cols].drop_duplicates(subset="participant_id"),
                           on="participant_id", how="left")

    tooth = tooth[tooth["session_id"].str.contains(V02, na=False)]

    # -----------------------------------------------------------------------
    # 6. Chronic conditions / diabetes (V01)
    # MATLAB: diab(ses-V01, [participant_id, chroncond_001___3, chroncond_001___4])
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["chroncond"]
    if p.exists():
        diab = _read_tsv(str(p))
        diab = diab[diab["session_id"].str.contains(V01, na=False)]
        cols = ["participant_id",
                "pex_bm_health_preg__chroncond_001___3",
                "pex_bm_health_preg__chroncond_001___4"]
        cols = [c for c in cols if c in diab.columns]
        tooth = pd.merge(tooth, diab[cols].drop_duplicates(subset="participant_id"),
                         on="participant_id", how="left")

    tooth = tooth[tooth["session_id"].str.contains(V02, na=False)]

    # -----------------------------------------------------------------------
    # 7. Gestational diabetes (V02) — exclude notvars
    # MATLAB: diab(ses-V02, diab_vars) minus gest_diab_notvars
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["healthv2"]
    if p.exists():
        diab2    = _read_tsv(str(p))
        diab2    = diab2[diab2["session_id"].str.contains(V02, na=False)]
        notvars2 = merge_cfg["gest_diab_notvars"]
        cols     = ["participant_id", "pex_bm_healthv2_preg__compl_001___1"]
        cols     = [c for c in cols if c in diab2.columns and c not in notvars2]
        tooth    = pd.merge(tooth, diab2[cols].drop_duplicates(subset="participant_id"),
                            on="participant_id", how="left")

    tooth = tooth[tooth["session_id"].str.contains(V02, na=False)]

    # -----------------------------------------------------------------------
    # 8. PTSD (V02)
    # MATLAB: ptsd(ses-V02, contains(varnames, 'total'))
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["ptsd"]
    if p.exists():
        ptsd = _read_tsv(str(p))
        ptsd = ptsd[ptsd["session_id"].str.contains(V02, na=False)]
        total_cols = [c for c in ptsd.columns if "total" in c]
        if total_cols:
            cols = ["participant_id", total_cols[0]]
            cols = [c for c in cols if c in ptsd.columns]
            tooth = pd.merge(tooth, ptsd[cols].drop_duplicates(subset="participant_id"),
                             on="participant_id", how="left")

    tooth = tooth[tooth["session_id"].str.contains(V02, na=False)]

    # -----------------------------------------------------------------------
    # 9. Eruption / medical history (ph_cg_ecls__medhist)
    # MATLAB: loaded separately in evalTeeth_v2_toothage_determ_rel2.m, merged on participant_id
    # -----------------------------------------------------------------------
    p = root / pheno_cfg["ecls_medhist"]
    if p.exists():
        ecls = _read_tsv(str(p))
        ecls_cols = [c for c in ecls.columns
                     if "009" in c or "010" in c or "adjusted_age" in c]
        ecls_sub = ecls[["participant_id"] + ecls_cols].copy()
        # Keep the row with non-missing 009 when duplicates exist
        ecls_sub["_missing_009"] = ecls_sub[
            [c for c in ecls_sub.columns if "009" in c][0]
        ].isna() if any("009" in c for c in ecls_sub.columns) else False
        ecls_sub = (ecls_sub.sort_values("_missing_009")
                             .drop_duplicates(subset="participant_id")
                             .drop(columns=["_missing_009"]))
        tooth = pd.merge(tooth, ecls_sub, on="participant_id", how="left")

    # -----------------------------------------------------------------------
    # Compute PMA and filter
    # MATLAB: pma_wks = candidate_age*52 + GA_delivery; filter pma_wks < 48
    # -----------------------------------------------------------------------
    age_col = "img_mriqc_T2w_candidate_age"
    ga_col  = "sed_basic_demographics_gestational_age_delivery"
    if age_col in tooth.columns and ga_col in tooth.columns:
        tooth["pma_wks"] = tooth[age_col] * 52 + tooth[ga_col]
        pma_max = cfg["merge"]["pma_max_weeks"]
        tooth   = tooth[tooth["pma_wks"] < pma_max]

    # -----------------------------------------------------------------------
    # Categorical mappings
    # MATLAB: categorical(sed_basic_demographics_child_race, 0:6, race_labels)
    # -----------------------------------------------------------------------
    race_map = {int(k): v for k, v in merge_cfg["race_map"].items()}
    if "sed_basic_demographics_child_race" in tooth.columns:
        tooth["child_race_cat"] = (
            tooth["sed_basic_demographics_child_race"].map(race_map).fillna("Other")
        )
    else:
        tooth["child_race_cat"] = "Other"

    eth_map = {int(k): v for k, v in merge_cfg["ethnicity_map"].items()}
    if "sed_basic_demographics_child_ethnicity" in tooth.columns:
        tooth["child_ethnicity_cat"] = (
            tooth["sed_basic_demographics_child_ethnicity"].map(eth_map).fillna("NonHisp")
        )

    sex_map = {int(k): v for k, v in merge_cfg["sex_map"].items()}
    if "sed_basic_demographics_sex" in tooth.columns:
        tooth["child_sex_cat"] = tooth["sed_basic_demographics_sex"].map(sex_map)

    # -----------------------------------------------------------------------
    # Winsorize weight and head circumference at ±3 SD
    # MATLAB: w_p3sd = nanmean + 3*nanstd; clip
    # -----------------------------------------------------------------------
    n_sd = merge_cfg["winsorize_sd"]
    for col in merge_cfg["winsorize_cols"]:
        if col in tooth.columns:
            mn  = tooth[col].mean()
            std = tooth[col].std()
            tooth[col] = tooth[col].clip(lower=mn - n_sd * std, upper=mn + n_sd * std)

    print(f"Final merged dataframe: {len(tooth)} rows.")
    tooth.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")
    return tooth


def main():
    parser = get_config_parser("Step 5b: Merge morphology with demographics")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    root = Path(args.config).resolve().parent

    in_path  = root / cfg["data"]["outputs"]["wide_stats"]
    out_path = root / cfg["data"]["outputs"]["merged"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merge_demographics(str(in_path), cfg, root, str(out_path))


if __name__ == "__main__":
    main()
