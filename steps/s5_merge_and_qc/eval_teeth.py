"""
steps/s5_merge_and_qc/eval_teeth.py
-------------------------------------
Step 5c — QC, outlier detection, and derived morphology features.

Python equivalent of evalTeeth_v2_rel2.m (QC section) and the outlier
filter logic in evalTeeth_v2_pma_pred_rel2.m.

Outputs:
  - evaluated_tooth_clean.csv  — QC-passed rows for modeling (steps 6-8)
  - evaluated_tooth_full.csv   — all rows with outlier flags appended

Usage:
    python -m steps.s5_merge_and_qc.eval_teeth --config config.yaml
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils.io import load_config, get_config_parser


def rigid_procrustes(mean_shape: np.ndarray, ind_shape: np.ndarray) -> float:
    """Procrustes distance without scaling or reflection.

    Equivalent to MATLAB PROCRUSTES(..., 'scaling', false, 'reflection', false).
    Returns the normalised squared-error distance d.
    """
    c_m = np.mean(mean_shape, axis=0)
    c_i = np.mean(ind_shape,  axis=0)
    X   = mean_shape - c_m
    Y   = ind_shape  - c_i
    norm_X_sq = np.sum(X ** 2)
    if norm_X_sq == 0:
        return np.nan
    H = Y.T @ X
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt
    Z = Y @ R
    return float(np.sum((X - Z) ** 2) / norm_X_sq)


def compute_distances_and_outliers(tooth: pd.DataFrame, jaw: str = "upper") -> pd.DataFrame:
    """Compute positional and shape outlier flags for one jaw.

    MATLAB equivalent: evalTeeth_v2_rel2.m upper/lower shape/pos outlier computation.
    """
    cols_x = sorted([c for c in tooth.columns if f"center_coord_x_{jaw}_seg" in c],
                    key=lambda c: int(c.split("_seg")[-1]))
    cols_y = sorted([c for c in tooth.columns if f"center_coord_y_{jaw}_seg" in c],
                    key=lambda c: int(c.split("_seg")[-1]))
    cols_z = sorted([c for c in tooth.columns if f"center_coord_z_{jaw}_seg" in c],
                    key=lambda c: int(c.split("_seg")[-1]))

    X = tooth[cols_x].values
    Y = tooth[cols_y].values
    Z = tooth[cols_z].values
    N, n_pts = X.shape

    mean_x = np.nanmean(X, axis=0)
    mean_y = np.nanmean(Y, axis=0)
    mean_z = np.nanmean(Z, axis=0)
    mean_shape_full = np.column_stack((mean_x, mean_y, mean_z))

    pos_distances     = np.zeros(N)
    shape_disparities = np.zeros(N)

    for i in range(N):
        ix, iy, iz = X[i], Y[i], Z[i]
        if np.any(np.isnan(ix)) or np.any(np.isnan(iy)) or np.any(np.isnan(iz)):
            pos_distances[i]     = np.nan
            shape_disparities[i] = np.nan
            continue
        dx = ix - mean_x
        dy = iy - mean_y
        dz = iz - mean_z
        pos_distances[i]     = np.nanmean(dx ** 2 + dy ** 2 + dz ** 2)
        ind_shape = np.column_stack((ix, iy, iz))
        shape_disparities[i] = rigid_procrustes(mean_shape_full, ind_shape)

    pos_thresh   = np.nanmean(pos_distances)   + 2.0 * np.nanstd(pos_distances,   ddof=1)
    shape_thresh = np.nanmean(shape_disparities) + 2.0 * np.nanstd(shape_disparities, ddof=1)

    is_pos_outlier   = (pos_distances   > pos_thresh)   | np.isnan(pos_distances)
    is_shape_outlier = (shape_disparities > shape_thresh) | np.isnan(shape_disparities)
    is_any_outlier   = is_pos_outlier | is_shape_outlier

    res = pd.DataFrame({
        f"{jaw}_is_pos_outlier":   is_pos_outlier.astype(int),
        f"{jaw}_is_shape_outlier": is_shape_outlier.astype(int),
        f"{jaw}_is_any_outlier":   is_any_outlier.astype(int),
        f"{jaw}_shape_disparity":  shape_disparities,
    }, index=tooth.index)

    # Arch geometry: width (seg1→seg10) and arc length (sum of sequential distances)
    if len(cols_x) >= 10:
        w_dx = X[:, 9] - X[:, 0]
        w_dy = Y[:, 9] - Y[:, 0]
        w_dz = Z[:, 9] - Z[:, 0]
        res[f"{jaw}_arch_width"] = np.sqrt(w_dx ** 2 + w_dy ** 2 + w_dz ** 2)

        dx_seq = np.diff(X, axis=1)
        dy_seq = np.diff(Y, axis=1)
        dz_seq = np.diff(Z, axis=1)
        res[f"{jaw}_arc_length"] = np.nansum(
            np.sqrt(dx_seq ** 2 + dy_seq ** 2 + dz_seq ** 2), axis=1
        )
    else:
        res[f"{jaw}_arch_width"] = np.nan
        res[f"{jaw}_arc_length"] = np.nan

    return res


def eval_teeth(merged_path: str, cfg: dict, clean_path: str, full_path: str) -> pd.DataFrame:
    """Add derived features, outlier flags, and filter to clean dataset.

    MATLAB reference: evalTeeth_v2_rel2.m (derived vars + outliers)
                      evalTeeth_v2_pma_pred_rel2.m (filter logic, lines 8-26)
    """
    sds = cfg["qc"]["outlier_sd"]

    print(f"Loading merged data from {merged_path}...")
    tooth = pd.read_csv(merged_path)

    # --- Derived volume features ---
    tooth["tooth_volume"] = (
        tooth["whole_hyperintense_volume_mm3"]
        + tooth["whole_hypointense_volume_mm3"]
        + tooth["whole_edge_volume_mm3"]
    )
    tooth["mineral_ratio"]  = tooth["whole_hypointense_volume_mm3"] / tooth["whole_hyperintense_volume_mm3"]
    tooth["mineral_ratio2"] = tooth["whole_hypointense_volume_mm3"] / tooth["tooth_volume"]

    tooth["lower_vol"] = (
        tooth["hyperintense_volume_mm3_lower"]
        + tooth["hypointense_volume_mm3_lower"]
        + tooth["edge_volume_mm3_lower"]
    )
    tooth["upper_vol"] = (
        tooth["hyperintense_volume_mm3_upper"]
        + tooth["hypointense_volume_mm3_upper"]
        + tooth["edge_volume_mm3_upper"]
    )
    tooth["mineral_ratio2_lower"] = tooth["hypointense_volume_mm3_lower"] / tooth["lower_vol"]
    tooth["mineral_ratio2_upper"] = tooth["hypointense_volume_mm3_upper"] / tooth["upper_vol"]

    # --- Shape and position outliers ---
    up_res = compute_distances_and_outliers(tooth, "upper")
    lo_res = compute_distances_and_outliers(tooth, "lower")
    tooth  = pd.concat([tooth, up_res, lo_res], axis=1)

    tooth["arch_width_ratio"]  = tooth["upper_arch_width"]  / tooth["lower_arch_width"]
    tooth["arch_length_ratio"] = tooth["upper_arc_length"]  / tooth["lower_arc_length"]

    # --- Volume outliers (2 SD) ---
    for jaw, vol_col in [("upper", "upper_vol"), ("lower", "lower_vol")]:
        mn = np.nanmean(tooth[vol_col])
        sd = np.nanstd(tooth[vol_col], ddof=1)
        vol_outlier = (tooth[vol_col] < (mn - sds * sd)) | (tooth[vol_col] > (mn + sds * sd))
        tooth[f"{jaw}_is_any_outlier"] = tooth[f"{jaw}_is_any_outlier"] | vol_outlier

    # --- Shape disparity outliers (used in filter) ---
    # MATLAB: evalTeeth_v2_pma_pred_rel2.m lines 8-19
    for jaw in ["lower", "upper"]:
        disp_col   = f"{jaw}_shape_disparity"
        mn         = np.nanmean(tooth[disp_col])
        sd         = np.nanstd(tooth[disp_col], ddof=1)
        tooth[f"{jaw}_shape_outlier"] = (
            (tooth[disp_col] < mn - sds * sd)
            | (tooth[disp_col] > mn + sds * sd)
            | np.isnan(tooth[disp_col])
        ).astype(int)

    # --- Filter mask ---
    # MATLAB: valid_rows = ~upper_is_any_outlier & ~lower_is_any_outlier
    #                      & ~upper_shape_outlier & ~lower_shape_outlier
    valid_rows = (
        (~tooth["upper_is_any_outlier"].astype(bool))
        & (~tooth["lower_is_any_outlier"].astype(bool))
        & (~tooth["upper_shape_outlier"].astype(bool))
        & (~tooth["lower_shape_outlier"].astype(bool))
        & (~tooth["pma_wks"].isna())
    )
    tooth_clean = tooth[valid_rows].copy()
    print(f"Total rows: {len(tooth)} | Clean (valid for modeling): {len(tooth_clean)}")

    # --- Example LME (mirrors MATLAB evalTeeth_v2_rel2.m first model) ---
    # tooth.upper_vol ~ child_sex_cat + pma_wks + (1|recruitment_site)
    site_col = "sed_basic_demographics_recruitment_site"
    if site_col in tooth_clean.columns:
        try:
            mdl = smf.mixedlm(
                "upper_vol ~ C(child_sex_cat) + pma_wks",
                tooth_clean,
                groups=tooth_clean[site_col],
            ).fit(reml=True)
            print("\n--- Example LME: upper_vol ~ sex + pma_wks + (1|site) ---")
            print(mdl.summary())
        except Exception as e:
            print(f"Could not fit example LME: {e}")

    tooth_clean.to_csv(clean_path, index=False)
    tooth.to_csv(full_path, index=False)
    print(f"Clean dataset saved to {clean_path}")
    print(f"Full (unfiltered) dataset saved to {full_path}")
    return tooth_clean


def main():
    parser = get_config_parser("Step 5c: QC and outlier detection")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    root = Path(args.config).resolve().parent

    merged_path = root / cfg["data"]["outputs"]["merged"]
    clean_path  = root / cfg["data"]["outputs"]["evaluated_clean"]
    full_path   = root / cfg["data"]["outputs"]["evaluated_full"]
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    eval_teeth(str(merged_path), cfg, str(clean_path), str(full_path))


if __name__ == "__main__":
    main()
