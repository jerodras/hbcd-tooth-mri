# MRI-Based Dental Maturity in Newborns

This repository contains the analysis code used for MRI-based neonatal tooth segmentation, dental feature extraction, tooth age prediction, and downstream association analyses.

The code is organized as a numbered workflow under `steps/`. Each step corresponds to one stage of the analysis pipeline described in the manuscript, "MRI-based dental maturity in newborns reflects prenatal exposures and predicts timing of primary tooth eruption."

## Repository Structure

```text
steps/
  s0_label/                  Manual labeling protocol
  s1_train/                  3D U-Net training script
  s2_inference/              Ensemble inference on T2-weighted MRI
  s3_majority_vote/          Majority-vote segmentation aggregation
  s4_extract_features/       Dental feature extraction from segmentations
  s5_merge_and_qc/           Table pivoting, phenotype merge, and QC
  s6_predict_age/            Tooth age prediction and age-gap estimation
  s7_exposure_associations/  Prenatal exposure association models
  s8_eruption_prediction/    Tooth eruption prediction models
```

## Workflow

1. `s0_label`: Documents the manual and assisted segmentation protocol used to create whole-dentition labels for model training.
2. `s1_train`: Trains a five-fold 3D U-Net ensemble for whole-dentition segmentation.
3. `s2_inference`: Applies trained fold models to neonatal T2-weighted MRI scans.
4. `s3_majority_vote`: Combines fold-wise predictions into one binary segmentation per subject using a four-of-five vote threshold.
5. `s4_extract_features`: Extracts tooth volume, tissue intensity, mineralization-related, and arch geometry features.
6. `s5_merge_and_qc`: Converts long-format image-derived features to wide format, merges phenotype tables, and applies QC/outlier filters.
7. `s6_predict_age`: Predicts postmenstrual age from dental features and derives a bias-corrected tooth age gap.
8. `s7_exposure_associations`: Tests associations between tooth age gap and prenatal/perinatal exposures using mixed-effects models.
9. `s8_eruption_prediction`: Tests whether neonatal tooth age gap predicts later primary tooth eruption outcomes.

## Data and Reproducibility

The repository does not include HBCD imaging data, controlled-access phenotype tables, generated NIfTI files, or model weights. Scripts expect local paths and analysis parameters to be supplied through `config.yaml`.

Python package dependencies are listed in `requirements.txt`.

## Typical Invocation

Each executable step is designed to be run as a module from the repository root:

```bash
python -m steps.s5_merge_and_qc.extract_stats --config config.yaml
python -m steps.s5_merge_and_qc.merge_data --config config.yaml
python -m steps.s5_merge_and_qc.eval_teeth --config config.yaml
python -m steps.s6_predict_age.predict_pma --config config.yaml
python -m steps.s7_exposure_associations.exposure_associations --config config.yaml
python -m steps.s8_eruption_prediction.eruption_prediction --config config.yaml
```

The segmentation training and inference scripts are retained for methodological transparency and future re-runs, but require the corresponding training images, labels, preprocessing utilities, and pretrained weights.

## Notes

