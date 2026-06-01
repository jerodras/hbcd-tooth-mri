"""
steps/s2_inference/inference.py
--------------------------------
DOCUMENTATION ONLY — not re-executed as part of the active pipeline.

Runs trained U-Net ensemble (5 folds) over new T2w NIfTI images and saves
per-fold binary predictions in full native resolution.

Config-driven: pass --cohort hbcd or --cohort roch to select the image
directory and output directory (defined in config.yaml or as CLI args).

Source scripts unified:
  segTeeth_v2_inference_release2.py  (HBCD Release 2)
  segTeeth_v2_inference_ROCH.py      (UPSIDE/ROCH cohort)
"""

import argparse
import os
import glob

import numpy as np
import nibabel as nib
import torch

from utils.model import UNet3D
from utils.preprocessing import preprocess_for_inference, normalize_minmax
from utils.io import load_config


def reconstruct_from_smart_crop(
    prediction_vol: np.ndarray, params: dict
) -> np.ndarray:
    """Place the cropped prediction back into native image space."""
    reconstructed = np.zeros(params["original_shape"], dtype=prediction_vol.dtype)
    pb = params["pad_before"]
    cs = params["cropped_vol_shape"]
    unpadded = prediction_vol[pb[0]:pb[0]+cs[0], pb[1]:pb[1]+cs[1], pb[2]:pb[2]+cs[2]]
    place = tuple(slice(s, e) for s, e in zip(params["crop_starts"], params["crop_ends"]))
    reconstructed[place] = unpadded
    return reconstructed


def run_inference(images_dir: str, output_dir: str, models_dir: str, target_shape=(160, 224, 256)):
    """Apply all fold models to every image in images_dir."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_paths = glob.glob(os.path.join(models_dir, "unet3d_REV*.pth"))
    if not model_paths:
        raise FileNotFoundError(f"No .pth model files found in {models_dir}.")
    print(f"Found {len(model_paths)} models.")

    image_paths = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    if not image_paths:
        raise FileNotFoundError(f"No .nii.gz images found in {images_dir}.")
    print(f"Found {len(image_paths)} images.")

    model = UNet3D(in_channels=1, out_channels=1, features=16).to(device)

    for img_idx, img_filepath in enumerate(image_paths):
        img_filename = os.path.basename(img_filepath)
        print(f"\n--- Image {img_idx+1}/{len(image_paths)}: {img_filename} ---")
        try:
            img_nifti = nib.load(img_filepath)
            image_data = img_nifti.get_fdata().astype(np.float32)
            normalized = normalize_minmax(image_data)
            processed, crop_params = preprocess_for_inference(normalized, target_shape)
            input_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0).to(device)

            for model_idx, model_path in enumerate(model_paths):
                model_name = os.path.basename(model_path)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                prediction = (output[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                reconstructed = reconstruct_from_smart_crop(prediction, crop_params)

                model_info  = model_name.replace("unet3d_", "").replace(".pth", "").replace("REV_", "")
                base_name   = img_filename.replace(".nii.gz", "")
                out_path    = os.path.join(output_dir, f"{base_name}_pred_{model_info}_fullres.nii.gz")
                nib.save(nib.Nifti1Image(reconstructed, img_nifti.affine), out_path)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"  Error processing {img_filename}: {e}. Skipping.")

    print("\nInference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net inference (documentation only)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--cohort", choices=["hbcd", "roch"], default="hbcd",
        help="Which cohort to run inference on."
    )
    parser.add_argument("--images_dir",  default=None, help="Override images directory")
    parser.add_argument("--output_dir",  default=None, help="Override output directory")
    parser.add_argument("--models_dir",  default="pretrained")
    args = parser.parse_args()

    if args.cohort == "hbcd":
        images_dir = args.images_dir or "./data/t2_images_hbcd_release2"
        output_dir = args.output_dir or "./data/inference_output_v2_release2"
    else:
        images_dir = args.images_dir or "./data/t2_images_ROCH"
        output_dir = args.output_dir or "./data/inference_output_v2_ROCH"

    run_inference(images_dir, output_dir, args.models_dir)
