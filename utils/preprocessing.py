"""
utils/preprocessing.py
-----------------------
Shared preprocessing functions used by training (s1) and inference (s2).

Smart crop logic:
- Threshold: 10th percentile of voxels > 0.01
- Binary opening (iterations=2)
- Anchor to superior-most point (bbox_max[0]) on axis 0
- Center on bbox center for axes 1, 2
- Pad with zeros to target shape (160, 224, 256)
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import binary_opening


def smart_crop_and_pad(
    image_vol: np.ndarray,
    label_vol: np.ndarray,
    target_shape: Tuple[int, int, int] = (160, 224, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply smart crop to image and label volumes, returning both at target_shape.

    Parameters
    ----------
    image_vol : np.ndarray
        3D float image array.
    label_vol : np.ndarray
        3D label array with the same spatial dimensions as image_vol.
    target_shape : tuple of int
        Desired output shape (D, H, W). Default: (160, 224, 256).

    Returns
    -------
    image_cropped : np.ndarray  shape == target_shape
    label_cropped : np.ndarray  shape == target_shape
    """
    target = np.array(target_shape)
    orig_shape = np.array(image_vol.shape)

    # --- 1. Threshold & binary opening to find main object ---
    try:
        threshold = np.quantile(image_vol[image_vol > 0.01], 0.10)
    except (IndexError, ValueError):
        threshold = 0.0

    mask = image_vol > threshold
    mask = binary_opening(mask, iterations=2)

    coords = np.argwhere(mask)
    if coords.size == 0:
        bbox_center = orig_shape // 2
        top_of_head_idx = int(bbox_center[0]) + (target_shape[0] // 2)
    else:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0) + 1
        bbox_center = (bbox_min + bbox_max) // 2
        # Anchor to superior-most point on axis 0
        top_of_head_idx = int(bbox_max[0])

    # --- 2. Define crop window ---
    start_0 = top_of_head_idx - target[0]
    end_0   = top_of_head_idx

    start_1 = int(bbox_center[1]) - (target[1] // 2)
    end_1   = start_1 + target[1]

    start_2 = int(bbox_center[2]) - (target[2] // 2)
    end_2   = start_2 + target[2]

    starts = np.array([start_0, start_1, start_2])
    ends   = np.array([end_0,   end_1,   end_2])

    # --- 3. Apply crop + zero-padding to both volumes ---
    outputs = []
    for volume in [image_vol, label_vol]:
        crop_starts = np.maximum(0, starts).astype(int)
        crop_ends   = np.minimum(orig_shape, ends).astype(int)

        slices = tuple(slice(s, e) for s, e in zip(crop_starts, crop_ends))
        cropped = volume[slices]

        pad_needed = target - np.array(cropped.shape)
        pad_before = np.maximum(0, -starts).astype(int)
        pad_after  = np.maximum(0, pad_needed - pad_before).astype(int)
        pad_width  = tuple((b, a) for b, a in zip(pad_before, pad_after))

        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            padded = np.pad(cropped, pad_width, mode="constant", constant_values=0)
        else:
            padded = cropped

        assert padded.shape == tuple(target_shape), (
            f"Shape mismatch: expected {target_shape}, got {padded.shape}"
        )
        outputs.append(padded)

    return outputs[0], outputs[1]


def preprocess_for_inference(
    image_vol: np.ndarray,
    target_shape: Tuple[int, int, int] = (160, 224, 256),
) -> Tuple[np.ndarray, dict]:
    """Smart-crop an image volume for inference and return crop parameters.

    Unlike smart_crop_and_pad (which handles image+label pairs), this version
    operates on a single image and returns the crop parameters needed to
    reconstruct predictions back to native space.

    Returns
    -------
    padded_vol : np.ndarray  shape == target_shape
    crop_params : dict
        Keys: original_shape, crop_starts, crop_ends, pad_before, cropped_vol_shape
    """
    target     = np.array(target_shape)
    orig_shape = np.array(image_vol.shape)

    try:
        threshold = np.quantile(image_vol[image_vol > 0.01], 0.10)
    except (IndexError, ValueError):
        threshold = 0.0

    mask   = image_vol > threshold
    mask   = binary_opening(mask, iterations=2)
    coords = np.argwhere(mask)

    if coords.size == 0:
        bbox_center      = orig_shape // 2
        top_of_head_idx  = int(bbox_center[0]) + (target_shape[0] // 2)
    else:
        bbox_min         = coords.min(axis=0)
        bbox_max         = coords.max(axis=0) + 1
        bbox_center      = (bbox_min + bbox_max) // 2
        top_of_head_idx  = int(bbox_max[0])

    starts = np.array([
        top_of_head_idx - target[0],
        int(bbox_center[1]) - (target[1] // 2),
        int(bbox_center[2]) - (target[2] // 2),
    ])

    crop_starts = np.maximum(0, starts).astype(int)
    crop_ends   = np.minimum(orig_shape, starts + target).astype(int)
    slices      = tuple(slice(s, e) for s, e in zip(crop_starts, crop_ends))
    cropped     = image_vol[slices]

    pad_needed = target - np.array(cropped.shape)
    pad_before = np.maximum(0, -starts).astype(int)
    pad_after  = np.maximum(0, pad_needed - pad_before).astype(int)
    pad_width  = tuple((b, a) for b, a in zip(pad_before, pad_after))
    padded     = np.pad(cropped, pad_width, mode="constant", constant_values=0)

    crop_params = {
        "original_shape":    orig_shape,
        "crop_starts":       crop_starts,
        "crop_ends":         crop_ends,
        "pad_before":        pad_before,
        "cropped_vol_shape": np.array(cropped.shape),
    }
    return padded, crop_params


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Min-max normalize a volume to [0, 1]. Returns zeros if flat."""
    lo, hi = image.min(), image.max()
    if hi > lo:
        return (image - lo) / (hi - lo)
    return image - lo
