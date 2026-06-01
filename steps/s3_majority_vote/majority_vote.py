"""
steps/s3_majority_vote/majority_vote.py
----------------------------------------
DOCUMENTATION ONLY — not re-executed as part of the active pipeline.

Aggregates per-fold NIfTI predictions via 4-of-5 supermajority vote into a
single binary segmentation mask per subject.

Usage (for future re-runs only):
    python -m steps.s3_majority_vote.majority_vote \\
        --input_dir  /path/to/inference_output \\
        --output_dir /path/to/simplified_labels \\
        --threshold  4 \\
        --n_folds    5
"""

import argparse
import os
import glob
from collections import defaultdict

import nibabel as nib
import numpy as np


def create_majority_vote_labels(
    input_dir: str,
    output_dir: str,
    vote_threshold: int = 4,
    expected_folds: int = 5,
) -> None:
    """Majority-vote fold predictions into final binary masks.

    Parameters
    ----------
    input_dir : str
        Directory containing per-fold prediction NIfTIs matching
        ``*_pred_fold*_epoch100_fullres.nii.gz``.
    output_dir : str
        Where to write the final ``*_label.nii.gz`` files.
    vote_threshold : int
        Minimum positive votes (out of expected_folds) for a voxel to be 1.
    expected_folds : int
        Number of fold files expected per subject. Subjects with fewer folds
        are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)

    pattern   = os.path.join(input_dir, "*_pred_fold*_epoch100_fullres.nii.gz")
    all_files = glob.glob(pattern)
    if not all_files:
        print(f"Error: No files found matching {pattern}")
        return

    grouped = defaultdict(list)
    for f in all_files:
        identifier = os.path.basename(f).split("_pred_fold")[0]
        grouped[identifier].append(f)

    print(f"Found {len(all_files)} files across {len(grouped)} subjects.")

    for identifier, fold_files in grouped.items():
        if len(fold_files) != expected_folds:
            print(
                f"Warning: {identifier} has {len(fold_files)} folds "
                f"(expected {expected_folds}). Skipping."
            )
            continue

        print(f"Processing: {identifier}")
        try:
            first_nii = nib.load(fold_files[0])
            sum_mask  = np.zeros(first_nii.get_fdata().shape, dtype=np.uint8)

            for f in fold_files:
                sum_mask += (nib.load(f).get_fdata() > 0).astype(np.uint8)

            majority_mask = (sum_mask >= vote_threshold).astype(np.uint8)
            out_basename  = f"{identifier.replace('_T2w', '')}_label.nii.gz"
            nib.save(
                nib.Nifti1Image(majority_mask, first_nii.affine, first_nii.header),
                os.path.join(output_dir, out_basename),
            )
            print(f"  Saved: {out_basename}")
        except Exception as e:
            print(f"  Error processing {identifier}: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Majority-vote fold predictions (documentation only)")
    parser.add_argument("--input_dir",  default="/path/to/inference_output_v2_release2")
    parser.add_argument("--output_dir", default="/path/to/simplified_labels_release2")
    parser.add_argument("--threshold",  type=int, default=4)
    parser.add_argument("--n_folds",    type=int, default=5)
    args = parser.parse_args()

    create_majority_vote_labels(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vote_threshold=args.threshold,
        expected_folds=args.n_folds,
    )
