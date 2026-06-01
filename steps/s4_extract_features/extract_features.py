"""
steps/s4_extract_features/extract_features.py
----------------------------------------------
DOCUMENTATION ONLY — not re-executed as part of the active pipeline.

Extracts 201 image-derived features (IDFs) per subject from T2w images and
majority-vote segmentations. Output is the long-format master statistics CSV
consumed by step 5 (s5_merge_and_qc).

Key algorithm details:
  - Upper/lower parcellation: two largest connected components, split by axis 2
  - Skeleton regularization: Gaussian smoothing σ=1.5, hole-filling, threshold 0.5
  - Tissue classification: 2-component GMM on T2 intensity within mask,
    combined with distance transform edge probability (sigmoid, threshold=2 voxels)
  - 3 tissue classes: hyperintense (1), hypointense/mineralization (2), edge (3)
  - 10 segments per arch skeleton, nearest-neighbor ordering from min-X endpoint
  - Stats per level: volume_mm3 and mean_intensity for each tissue class,
    plus center coordinates at segment level

Paths are read from config.yaml (passed via --config).

Usage (for future re-runs only):
    python -m steps.s4_extract_features.extract_features --config config.yaml
"""

import argparse
import csv
import os
import glob

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import ListedColormap
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from sklearn.mixture import GaussianMixture

from utils.io import load_config, get_config_parser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
parser = get_config_parser("Feature extraction (documentation only — not re-run)")
args = parser.parse_args()
cfg = load_config(args.config)
fe  = cfg["feature_extraction"]

UPPER_LOWER_SPLIT_AXIS   = 2
APPLY_SKELETON_REGULARIZATION = True
GAUSSIAN_SIGMA           = fe["skeleton_sigma"]
EDGE_DISTANCE_THRESHOLD  = fe["edge_sigmoid_threshold"]
NUM_SKELETON_SEGMENTS    = fe["n_arch_segments"]

T2_IMAGE_DIR      = "./data/t2_images_REV"
SEGMENTATION_DIR  = "./data/simplified_labels_release2"
MASTER_OUTPUT_DIR = "./data/output_release2"


# ---------------------------------------------------------------------------
# Core processing functions (logic unchanged from process_teeth_v9.py)
# ---------------------------------------------------------------------------

def parcellate_upper_lower(seg_data, axis=2):
    """Split a binary segmentation into two largest connected components."""
    labels, num_features = ndi.label(seg_data)
    if num_features < 2:
        print("Warning: Found fewer than 2 components. Cannot split.")
        return seg_data, np.zeros_like(seg_data)
    component_sizes = ndi.sum(seg_data, labels, index=np.arange(1, num_features + 1))
    largest_indices = np.argsort(component_sizes)[-2:] + 1
    centers = ndi.center_of_mass(seg_data, labels, largest_indices)
    if centers[0][axis] > centers[1][axis]:
        upper_label, lower_label = largest_indices[0], largest_indices[1]
    else:
        upper_label, lower_label = largest_indices[1], largest_indices[0]
    upper_mask = (labels == upper_label).astype(np.uint8)
    lower_mask = (labels == lower_label).astype(np.uint8)
    return upper_mask, lower_mask


def get_skeleton(mask):
    """Compute 3D skeleton of a binary mask."""
    return skeletonize(mask).astype(np.uint8)


def classify_tissues(seg_mask, t2_image):
    """Classify tissues using GMM intensity + distance-transform edge probability."""
    classification_map    = np.zeros_like(seg_mask, dtype=np.uint8)
    tooth_voxels_indices  = np.where(seg_mask > 0)
    p_hypo  = np.zeros_like(t2_image, dtype=float)
    p_hyper = np.zeros_like(t2_image, dtype=float)
    tooth_intensities = t2_image[tooth_voxels_indices].reshape(-1, 1)
    if len(tooth_intensities) > 2:
        gmm = GaussianMixture(n_components=2, random_state=0).fit(tooth_intensities)
        hypo_idx  = np.argmin(gmm.means_)
        hyper_idx = np.argmax(gmm.means_)
        probs     = gmm.predict_proba(t2_image.ravel().reshape(-1, 1))
        p_hypo    = probs[:, hypo_idx].reshape(t2_image.shape)
        p_hyper   = probs[:, hyper_idx].reshape(t2_image.shape)
    distance_from_edge = ndi.distance_transform_edt(seg_mask)
    p_edge     = 1.0 - sigmoid((distance_from_edge - EDGE_DISTANCE_THRESHOLD) * 2)
    p_interior = 1.0 - p_edge
    scores = np.stack([
        np.zeros_like(t2_image),
        p_hyper,
        p_hypo * p_interior,
        p_hypo * p_edge,
    ], axis=0)
    raw = np.argmax(scores, axis=0)
    classification_map[tooth_voxels_indices] = raw[tooth_voxels_indices]
    return classification_map


def get_skeleton_points_ordered(skeleton_img):
    """Ordered skeleton points: nearest-neighbor from min-X endpoint."""
    points = np.argwhere(skeleton_img > 0)
    if len(points) == 0:
        return np.array([])
    endpoints = []
    for p in points:
        n_neighbors = np.sum(skeleton_img[p[0]-1:p[0]+2, p[1]-1:p[1]+2, p[2]-1:p[2]+2]) - 1
        if n_neighbors == 1:
            endpoints.append(p)
    if endpoints:
        endpoints   = np.array(endpoints)
        start_point = endpoints[np.argmin(endpoints[:, 2])]
    else:
        print("Warning: Skeleton has no endpoints (loop). Starting at min X coordinate.")
        start_point = points[np.argmin(points[:, 2])]
    start_idx       = np.where((points == start_point).all(axis=1))[0][0]
    ordered_points  = [points[start_idx]]
    remaining       = np.delete(points, start_idx, axis=0)
    while len(remaining) > 0:
        last     = ordered_points[-1]
        dists    = cdist([last], remaining)[0]
        nearest  = np.argmin(dists)
        ordered_points.append(remaining[nearest])
        remaining = np.delete(remaining, nearest, axis=0)
    return np.array(ordered_points)


def analyze_stats_along_skeleton(ordered_points, tissue_map, t2_image, voxel_vol, num_segments):
    """Statistics per tissue class in segments along the skeleton."""
    if len(ordered_points) == 0:
        return []
    distances          = np.sqrt(np.sum(np.diff(ordered_points, axis=0) ** 2, axis=1))
    arc_length         = np.insert(np.cumsum(distances), 0, 0)
    segment_boundaries = np.linspace(0, arc_length[-1], num_segments + 1)
    stats_by_segment   = []
    for i in range(num_segments):
        start_len, end_len = segment_boundaries[i], segment_boundaries[i + 1]
        indices = np.where((arc_length >= start_len) & (arc_length < end_len))[0]
        if len(indices) == 0:
            continue
        center_point  = ordered_points[indices[len(indices) // 2]]
        radius        = 10
        z, y, x = np.ogrid[:tissue_map.shape[0], :tissue_map.shape[1], :tissue_map.shape[2]]
        roi_mask = (x - center_point[2]) ** 2 + (y - center_point[1]) ** 2 + (z - center_point[0]) ** 2 <= radius ** 2
        seg_stats = {"segment_id": i, "center_coord": center_point.tolist()}
        for tissue_id, tissue_name in zip([1, 2, 3], ["Hyperintense", "Hypointense", "Edge"]):
            final_mask = roi_mask & (tissue_map == tissue_id)
            n_voxels   = np.sum(final_mask)
            key        = tissue_name.lower()
            seg_stats[f"{key}_volume_mm3"]      = n_voxels * voxel_vol
            seg_stats[f"{key}_mean_intensity"]  = np.mean(t2_image[final_mask]) if n_voxels > 0 else 0
        stats_by_segment.append(seg_stats)
    return stats_by_segment


def save_nifti(data, original_nii, filename):
    nib.save(nib.Nifti1Image(data.astype(np.uint8), original_nii.affine, original_nii.header), filename)


def flatten_stats_to_rows(stats_dict, subject_id):
    """Convert nested per-subject stats dict into flat CSV rows."""
    rows = []
    base_coord_row = {"center_coord_z": "NA", "center_coord_y": "NA", "center_coord_x": "NA"}
    row = {"subject_id": subject_id, "level": "whole_dentition", "arch": "NA", "segment_id": "NA"}
    row.update(base_coord_row)
    row.update(stats_dict["whole_dentition"])
    rows.append(row)
    for arch_name in ["upper_arch", "lower_arch"]:
        row = {"subject_id": subject_id, "level": "per_arch",
               "arch": arch_name.replace("_arch", ""), "segment_id": "NA"}
        row.update(base_coord_row)
        row.update(stats_dict[arch_name])
        rows.append(row)
    for arch_name in ["upper_arch_segments", "lower_arch_segments"]:
        for segment_data in stats_dict[arch_name]:
            row = {"subject_id": subject_id, "level": "per_segment",
                   "arch": arch_name.replace("_arch_segments", ""),
                   "segment_id": segment_data["segment_id"]}
            for key, value in segment_data.items():
                if "coord" not in key and "segment_id" not in key:
                    row[key] = value
            coords = segment_data.get("center_coord", ["NA", "NA", "NA"])
            row["center_coord_z"] = coords[0]
            row["center_coord_y"] = coords[1]
            row["center_coord_x"] = coords[2]
            rows.append(row)
    return rows


def visualize_results(t2_data, upper_mask, lower_mask,
                      upper_skeleton_points, lower_skeleton_points,
                      tissue_map, output_path):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Tooth Analysis Results", fontsize=20)
    projection_axes = tuple(i for i in range(t2_data.ndim) if i != UPPER_LOWER_SPLIT_AXIS)
    lower_slice_idx = np.argmax(np.sum(lower_mask, axis=projection_axes))
    upper_slice_idx = np.argmax(np.sum(upper_mask, axis=projection_axes))
    center_slice    = np.argmax(np.sum(upper_mask | lower_mask, axis=projection_axes))
    ax1 = fig.add_subplot(2, 3, 1)
    slices = [slice(None)] * 3
    slices[UPPER_LOWER_SPLIT_AXIS] = lower_slice_idx
    ax1.imshow(t2_data[tuple(slices)].T, cmap="gray", origin="lower")
    ax1.imshow(lower_mask[tuple(slices)].T,
               cmap=ListedColormap([(0,0,0,0), (1,0,0,0.5)]), origin="lower")
    ax1.set_title(f"Lower Arch (Slice {lower_slice_idx})")
    ax1.axis("off")
    ax2 = fig.add_subplot(2, 3, 2)
    slices[UPPER_LOWER_SPLIT_AXIS] = upper_slice_idx
    ax2.imshow(t2_data[tuple(slices)].T, cmap="gray", origin="lower")
    ax2.imshow(upper_mask[tuple(slices)].T,
               cmap=ListedColormap([(0,0,0,0), (0,0,1,0.5)]), origin="lower")
    ax2.set_title(f"Upper Arch (Slice {upper_slice_idx})")
    ax2.axis("off")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    if upper_skeleton_points.size > 0:
        ax3.plot(upper_skeleton_points[:, 2], upper_skeleton_points[:, 1],
                 upper_skeleton_points[:, 0], color="blue", linewidth=3, label="Upper")
    if lower_skeleton_points.size > 0:
        ax3.plot(lower_skeleton_points[:, 2], lower_skeleton_points[:, 1],
                 lower_skeleton_points[:, 0], color="red", linewidth=3, label="Lower")
    ax3.set_title("3D Skeletons"); ax3.legend()
    for i, slice_idx in enumerate([center_slice - 1, center_slice, center_slice + 1]):
        ax = fig.add_subplot(2, 3, 4 + i)
        if not (0 <= slice_idx < tissue_map.shape[UPPER_LOWER_SPLIT_AXIS]):
            ax.text(0.5, 0.5, "Slice out of bounds", ha="center", va="center")
            ax.axis("off")
            continue
        slices[UPPER_LOWER_SPLIT_AXIS] = slice_idx
        ax.imshow(tissue_map[tuple(slices)].T, cmap="nipy_spectral",
                  origin="lower", vmin=0, vmax=3)
        ax.set_title(f"Tissue Map (Slice {slice_idx})")
        ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
    t2_files = sorted(glob.glob(os.path.join(T2_IMAGE_DIR, "sub-*_T2w.nii.gz")))
    if not t2_files:
        print(f"Error: No T2w files found in '{T2_IMAGE_DIR}'.")
        return
    print(f"Found {len(t2_files)} subjects.")
    all_rows = []

    for t2_path in t2_files:
        try:
            base_name  = os.path.basename(t2_path).replace("_T2w.nii.gz", "")
            subject_id = base_name.split("_")[0]
            print(f"\n--- Processing: {subject_id} ---")
            label_path        = os.path.join(SEGMENTATION_DIR, base_name + "_label.nii.gz")
            subject_output_dir = os.path.join(MASTER_OUTPUT_DIR, base_name)
            os.makedirs(subject_output_dir, exist_ok=True)
            if not os.path.exists(label_path):
                print(f"Label not found: {label_path}. Skipping.")
                continue
            seg_nii       = nib.load(label_path)
            t2_nii        = nib.load(t2_path)
            seg_data      = seg_nii.get_fdata().astype(np.uint8)
            t2_data       = t2_nii.get_fdata()
            voxel_volume  = np.prod(seg_nii.header.get_zooms())
            upper_mask, lower_mask = parcellate_upper_lower(seg_data, axis=UPPER_LOWER_SPLIT_AXIS)
            save_nifti(upper_mask, seg_nii, os.path.join(subject_output_dir, "debug_upper_mask_raw.nii.gz"))
            save_nifti(lower_mask, seg_nii, os.path.join(subject_output_dir, "debug_lower_mask_raw.nii.gz"))
            if APPLY_SKELETON_REGULARIZATION:
                print(f"  Gaussian regularization σ={GAUSSIAN_SIGMA}")
                filled_upper = ndi.binary_fill_holes(upper_mask)
                smooth_upper = ndi.gaussian_filter(filled_upper.astype(float), sigma=GAUSSIAN_SIGMA)
                regularized_upper = (smooth_upper > 0.5).astype(np.uint8)
                filled_lower = ndi.binary_fill_holes(lower_mask)
                smooth_lower = ndi.gaussian_filter(filled_lower.astype(float), sigma=GAUSSIAN_SIGMA)
                regularized_lower = (smooth_lower > 0.5).astype(np.uint8)
                save_nifti(regularized_upper, seg_nii, os.path.join(subject_output_dir, "debug_upper_mask_regularized.nii.gz"))
                save_nifti(regularized_lower, seg_nii, os.path.join(subject_output_dir, "debug_lower_mask_regularized.nii.gz"))
            else:
                regularized_upper = upper_mask
                regularized_lower = lower_mask
            upper_skel = get_skeleton(regularized_upper)
            lower_skel = get_skeleton(regularized_lower)
            save_nifti(upper_skel, seg_nii, os.path.join(subject_output_dir, "debug_upper_skeleton.nii.gz"))
            save_nifti(lower_skel, seg_nii, os.path.join(subject_output_dir, "debug_lower_skeleton.nii.gz"))
            upper_pts = get_skeleton_points_ordered(upper_skel)
            lower_pts = get_skeleton_points_ordered(lower_skel)
            full_mask         = (upper_mask | lower_mask).astype(np.uint8)
            tissue_map        = classify_tissues(full_mask, t2_data)
            save_nifti(tissue_map, seg_nii, os.path.join(subject_output_dir, "debug_tissue_classification_map.nii.gz"))
            all_stats = {"whole_dentition": {}, "upper_arch": {}, "lower_arch": {}}
            for tissue_id, name in zip([1, 2, 3], ["Hyperintense", "Hypointense", "Edge"]):
                m = tissue_map == tissue_id
                all_stats["whole_dentition"][f"{name.lower()}_volume_mm3"] = np.sum(m) * voxel_volume
                all_stats["whole_dentition"][f"{name.lower()}_mean_intensity"] = np.mean(t2_data[m]) if np.any(m) else 0
            for arch_name, arch_mask in zip(["upper_arch", "lower_arch"], [upper_mask, lower_mask]):
                for tissue_id, name in zip([1, 2, 3], ["Hyperintense", "Hypointense", "Edge"]):
                    m = (tissue_map == tissue_id) & (arch_mask > 0)
                    all_stats[arch_name][f"{name.lower()}_volume_mm3"] = np.sum(m) * voxel_volume
                    all_stats[arch_name][f"{name.lower()}_mean_intensity"] = np.mean(t2_data[m]) if np.any(m) else 0
            all_stats["upper_arch_segments"] = analyze_stats_along_skeleton(upper_pts, tissue_map, t2_data, voxel_volume, NUM_SKELETON_SEGMENTS)
            all_stats["lower_arch_segments"] = analyze_stats_along_skeleton(lower_pts, tissue_map, t2_data, voxel_volume, NUM_SKELETON_SEGMENTS)
            all_rows.extend(flatten_stats_to_rows(all_stats, subject_id))
            visualize_results(
                t2_data=t2_data, upper_mask=upper_mask, lower_mask=lower_mask,
                upper_skeleton_points=upper_pts, lower_skeleton_points=lower_pts,
                tissue_map=tissue_map,
                output_path=os.path.join(subject_output_dir, "summary_visualization.png"),
            )
            print(f"Done: {subject_id}")
        except Exception as e:
            print(f"FAILED: {os.path.basename(t2_path)} — {e}")
            continue

    if all_rows:
        out_csv = os.path.join(MASTER_OUTPUT_DIR, "master_statistics.csv")
        headers = list(all_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Master CSV saved: {out_csv}")
    else:
        print("No subjects processed. No CSV written.")


if __name__ == "__main__":
    main()
