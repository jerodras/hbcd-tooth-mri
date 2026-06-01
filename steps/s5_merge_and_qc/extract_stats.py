"""
steps/s5_merge_and_qc/extract_stats.py
---------------------------------------
Step 5a — Pivot long-format feature table to wide format.

Python equivalent of extractArcStats.m.
Reads the frozen long-format CSV from step 4 and pivots it to a wide table
with one row per subject, suitable for merging with demographics.

Usage:
    python -m steps.s5_merge_and_qc.extract_stats --config config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from utils.io import load_config, get_config_parser


def extract_and_pivot_stats(csv_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Load long-format tooth statistics and pivot to wide format.

    Produces three levels of features and merges them by participant_id:
      - whole_dentition: one row per subject
      - per_arch: one row per (subject, arch) → pivoted to columns
      - per_segment: one row per (subject, arch, segment) → pivoted to columns

    Parameters
    ----------
    csv_path : str
        Path to teeth_master_statistics_hbcd_rel2.csv (or ROCH equivalent).
    output_path : str, optional
        If provided, save the wide table here.

    Returns
    -------
    pd.DataFrame
        Wide-format morphology table.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, na_values=["NA", "NaN"])
    print(f"Loaded {len(df)} rows.")

    data_vars = [
        "hyperintense_volume_mm3",
        "hyperintense_mean_intensity",
        "hypointense_volume_mm3",
        "hypointense_mean_intensity",
        "edge_volume_mm3",
        "edge_mean_intensity",
    ]

    # --- 1. Whole Dentition Level ---
    whole = df[df["level"] == "whole_dentition"].copy()
    whole_wide = whole[["participant_id"] + data_vars].copy()
    whole_wide = whole_wide.rename(columns={c: f"whole_{c}" for c in data_vars})

    # --- 2. Per-Arch Level ---
    per_arch = df[df["level"] == "per_arch"].copy()
    per_arch_wide = per_arch.pivot(
        index="participant_id", columns="arch", values=data_vars
    )
    per_arch_wide.columns = [f"{col}_{arch}" for col, arch in per_arch_wide.columns]
    per_arch_wide = per_arch_wide.reset_index()

    # --- 3. Per-Segment Level ---
    per_segment = df[df["level"] == "per_segment"].copy()
    per_segment["segment_id"] = per_segment["segment_id"].apply(
        lambda x: str(int(x)) if pd.notnull(x) else str(x)
    )
    per_segment["segment_identifier"] = (
        per_segment["arch"].astype(str) + "_seg" + per_segment["segment_id"]
    )
    data_vars_seg = data_vars + ["center_coord_z", "center_coord_y", "center_coord_x"]
    per_seg_wide = per_segment.pivot(
        index="participant_id", columns="segment_identifier", values=data_vars_seg
    )
    per_seg_wide.columns = [f"{col}_{seg}" for col, seg in per_seg_wide.columns]
    per_seg_wide = per_seg_wide.reset_index()

    # --- Merge all three levels ---
    print("Joining all data levels...")
    final_wide = whole_wide.merge(per_arch_wide, on="participant_id", how="outer")
    final_wide = final_wide.merge(per_seg_wide,  on="participant_id", how="outer")
    print(f"Final wide table size: {final_wide.shape}")

    if output_path:
        final_wide.to_csv(output_path, index=False)
        print(f"Saved wide table to {output_path}")

    return final_wide


def main():
    parser = get_config_parser("Step 5a: Pivot long-format feature table to wide format")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    root = Path(args.config).resolve().parent

    in_path  = root / cfg["data"]["frozen"]["hbcd"]
    out_path = root / cfg["data"]["outputs"]["wide_stats"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extract_and_pivot_stats(str(in_path), str(out_path))


if __name__ == "__main__":
    main()
