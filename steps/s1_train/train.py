"""
steps/s1_train/train.py
-----------------------
DOCUMENTATION ONLY — not re-executed as part of the active pipeline.

Multi-Scale 3D U-Net Segmentation with K-Fold Cross Validation and Augmentation.

This script was used to train the U-Net ensemble that produced the pretrained
.pth weights in pretrained/. The frozen outputs from this step (per-fold
predictions) were consumed by step 3 (majority vote).

Training hyperparameters:
  - Architecture: UNet3D (base_features=16) — see utils/model.py
  - Epochs: 100, Batch size: 3, LR: 1e-4
  - K-folds: 5, Augmentation multiplier: 5
  - Augmentation: RandomAffine(degrees=5, translation=2)
  - Input shape: (1, 1, 160, 224, 256)

Paths are read from config.yaml.
"""

import os
import glob
import random
import csv

import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler, autocast
import torchio as tio
from sklearn.model_selection import KFold

from utils.model import UNet3D
from utils.preprocessing import smart_crop_and_pad, normalize_minmax
from utils.metrics import dice_loss, dice_score
from utils.io import load_config, get_config_parser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
parser = get_config_parser("Train UNet3D (documentation only — not re-run)")
args = parser.parse_args()
cfg = load_config(args.config)

seed = 42
random.seed(seed)
torch.manual_seed(seed)

num_epochs    = 100
batch_size    = 3
learning_rate = 1e-4
k_folds       = 5
aug_multiplier = 5

target_shape = tuple(cfg["unet"]["input_shape"][2:])  # (160, 224, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (edit in config.yaml, not here)
images_dir      = "./data/t2_images_REV"
labels_dir      = "./data/teeth_labels_REV"
predictions_dir = "./data/full_resolution_predictions_REV"
models_dir      = "./pretrained"

os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

csv_filename = (
    f"multi_scale_kfold_segmentation_metrics_REV"
    f"_k{k_folds}_e{num_epochs}_aug{aug_multiplier}_lr{learning_rate:.0e}.csv"
)

affine_transform = tio.RandomAffine(
    degrees=(5, 5, 5),
    translation=(2, 2, 2),
    p=1.0,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FullResolutionPitDataset(Dataset):
    """Loads full-resolution 3D NIfTI images and segmentation labels."""

    def __init__(self, images_dir, labels_dir, target_shape=target_shape, normalize=True):
        self.image_paths  = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
        self.label_paths  = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
        assert len(self.image_paths) == len(self.label_paths), (
            "Mismatch in number of images and labels."
        )
        self.subject_ids  = [os.path.basename(p).split(".")[0] for p in self.image_paths]
        self.target_shape = target_shape
        self.normalize    = normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_nifti = nib.load(self.image_paths[idx])
        label_nifti = nib.load(self.label_paths[idx])
        image = image_nifti.get_fdata().astype(np.float32)
        label = label_nifti.get_fdata().astype(np.float32)
        label = (label > 0.5).astype(np.float32)
        if self.normalize:
            image = normalize_minmax(image)
        image_adj, label_adj = smart_crop_and_pad(image, label, self.target_shape)
        return {
            "image":      torch.from_numpy(image_adj).unsqueeze(0),
            "label":      torch.from_numpy(label_adj).unsqueeze(0),
            "subject_id": self.subject_ids[idx],
            "affine":     image_nifti.affine,
        }


class AugmentedDataset(Dataset):
    """Returns aug_multiplier augmented copies of each sample."""

    def __init__(self, base_dataset, multiplier=5, transform=None):
        self.base_dataset = base_dataset
        self.multiplier   = multiplier
        self.transform    = transform

    def __len__(self):
        return len(self.base_dataset) * self.multiplier

    def __getitem__(self, idx):
        sample   = self.base_dataset[idx // self.multiplier]
        if self.transform is not None:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=sample["image"], affine=np.eye(4)),
                label=tio.LabelMap(tensor=sample["label"],  affine=np.eye(4)),
            )
            transformed = self.transform(subject)
            sample["image"] = transformed["image"].data
            sample["label"] = transformed["label"].data
        return sample


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
full_dataset = FullResolutionPitDataset(images_dir, labels_dir, target_shape=target_shape)
kfold        = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
all_indices  = np.arange(len(full_dataset))
all_metrics  = []

fold_num = 0
for train_idx, test_idx in kfold.split(all_indices):
    fold_num += 1
    print(f"\n========== Fold {fold_num}/{k_folds} ==========")

    train_subset = Subset(full_dataset, train_idx)
    test_subset  = Subset(full_dataset, test_idx)

    if aug_multiplier > 1:
        train_subset = AugmentedDataset(train_subset, multiplier=aug_multiplier, transform=affine_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_subset,  batch_size=1, shuffle=False)

    model     = UNet3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler    = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss    = dice_loss(outputs, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        fold_test_scores = []
        for batch in test_loader:
            with torch.no_grad():
                outputs = model(batch["image"].to(device))
            fold_test_scores.append(dice_score(outputs, batch["label"].to(device)).item())
        test_dice_mean = np.mean(fold_test_scores)
        test_dice_std  = np.std(fold_test_scores)

        all_metrics.append({
            "fold": fold_num, "epoch": epoch + 1,
            "train_loss": avg_loss,
            "test_dice_mean": test_dice_mean,
            "test_dice_std": test_dice_std,
        })
        print(
            f"Fold {fold_num} Epoch {epoch+1}/{num_epochs} — "
            f"Loss: {avg_loss:.4f} | Test Dice: {test_dice_mean:.4f} ± {test_dice_std:.4f}"
        )

        if epoch == num_epochs - 1:
            for batch in test_loader:
                subject_id = batch["subject_id"][0]
                affine     = batch["affine"][0]
                with torch.no_grad():
                    pred = model(batch["image"].to(device))
                pred_bin = (pred > 0.5).float().cpu().numpy()[0, 0]
                pred_filename = os.path.join(
                    predictions_dir,
                    f"{subject_id}_augs{aug_multiplier}_fold{fold_num}_of_{k_folds}_epoch{epoch+1}_REV.nii.gz",
                )
                nib.save(nib.Nifti1Image(pred_bin, affine), pred_filename)

    model_save_path = os.path.join(models_dir, f"unet3d_REV_fold{fold_num}_epoch{num_epochs}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for Fold {fold_num} saved to: {model_save_path}")

# Write metrics CSV
with open(csv_filename, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["fold", "epoch", "train_loss", "test_dice_mean", "test_dice_std"])
    writer.writeheader()
    writer.writerows(all_metrics)
print(f"Metrics saved to {csv_filename}")
