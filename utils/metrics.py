"""
utils/metrics.py
----------------
Shared loss and evaluation metrics used by training (s1) and inference (s2).
"""

import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    Parameters
    ----------
    pred   : Tensor shape (B, 1, D, H, W), sigmoid probabilities
    target : Tensor shape (B, 1, D, H, W), binary labels
    smooth : Laplace smoothing constant
    """
    pred   = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + smooth
    )
    return 1.0 - dice.mean()


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Hard Dice score (threshold at 0.5).

    Parameters
    ----------
    pred   : Tensor shape (B, 1, D, H, W), sigmoid probabilities
    target : Tensor shape (B, 1, D, H, W), binary labels
    smooth : Laplace smoothing constant
    """
    pred_bin   = (pred > 0.5).float()
    target_bin = (target > 0.5).float()
    intersection = (pred_bin * target_bin).sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (
        pred_bin.sum(dim=(2, 3, 4)) + target_bin.sum(dim=(2, 3, 4)) + smooth
    )
    return dice.mean()
