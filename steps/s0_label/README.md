# Step 0: Ground Truth Labeling

**Documentation only — not re-executed.**

## Protocol

Manual segmentation of whole dentition in T2-weighted neonatal MRI.

- Tool: nnInteractive (interactive neural-network-assisted segmentation)
- Cohort: n=100 HBCD subjects (training set for the U-Net)
- Labeling procedure:
  1. Initial automated proposals generated via nnInteractive
  2. Expert review and correction by trained annotator
  3. Independent adjudication for ambiguous cases
- Output: binary NIfTI masks (`*_label.nii.gz`) co-registered to T2w space

## Label Classes

| Value | Structure |
|-------|-----------|
| 0 | Background |
| 1 | Whole dentition (both arches) |

## Quality Control

Labels were reviewed for:
- Complete arch coverage (upper and lower)
- No inclusion of mandible/maxilla bone
- Consistent superior/inferior boundaries across subjects
