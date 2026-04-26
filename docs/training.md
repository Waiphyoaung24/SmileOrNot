# Training the SmileOrNot YOLO26n model

This document narrates the training run on Ultralytics Platform that produced
`weights/best.pt`. Reviewers can't watch you click through the Platform UI, so
this is where the visual evidence lives.

## Dataset

- **Source:** CelebA (~200K celebrity face images, distributed via Hugging Face).
- **Subset:** 10,000 images — 5,000 smiling + 5,000 neutral, balanced.
- **Auto-labeling:** A pretrained `yolo11n-face` detector produced one bounding
  box per image. Class label came from CelebA's `Smiling` attribute (1 → smiling,
  -1 → neutral).
- **Refinement:** Smart Annotation pass on Ultralytics Platform corrected
  misaligned boxes and a small number of wrong-class labels.

![Smart Annotation refinement before/after](img/smart_annotation_before_after.png)
*<small>Before/after: red boxes are auto-labeled; green are post-refinement.</small>*

## Training configuration

- **Base model:** `yolo26n.pt` (released January 2026, NMS-free architecture)
- **Epochs:** 50
- **Image size:** 640
- **Batch size:** auto (Platform-selected)
- **Optimizer:** AdamW (Ultralytics default)
- **Augmentation:** Mosaic + flip (defaults)
- **Hardware:** Single A100 on Ultralytics Platform cloud
- **Wall clock:** ~30 minutes

## Results

- **mAP50:** ~0.94 on the held-out validation split
- **mAP50-95:** ~0.78
- **Final weights:** ~6 MB (`weights/best.pt`)

![Training loss curves](img/loss_curves.png)
![Validation mAP](img/val_metrics.png)

## Inference latency on the deployment target

The model runs on a Hugging Face free CPU Space (2 vCPU, no GPU). Measured
end-to-end latency on a 480×360 JPEG input:

- **First request after cold start:** ~400 ms
- **Steady state:** ~150-220 ms per frame
- **Effective live FPS:** 4-6 (single user, adaptive client loop)

## Notes for reproduction

The exact subset and split are deterministic given `SEED=42` (used by
`scripts/02_autolabel.py` and `scripts/03_split.py`). Results may differ
slightly from the numbers above due to Platform-side randomness in
augmentation seeding.
