# Low-Visibility Helmet Detection
Aristia AI Selection Task — Problem 2

---

I picked this problem because helmet detection in poor visibility is a genuinely unsolved real-world issue — most deployed safety systems just fail at night or in fog. My approach avoids retraining the detector entirely; instead I preprocess each frame to remove the degradation first, then let YOLOv8 do what it's already good at.

## How it works

```
Raw Frame (Hazy / Dark)
        │
        ▼
Dark Channel Prior De-hazing   ← removes fog/haze
        │
        ▼
CLAHE Re-lighting              ← boosts low-light contrast
        │
        ▼
YOLOv8n Detection              ← detects helmet / no-helmet
        │
        ▼
Annotated Output
```

Dark Channel Prior is a physics-based dehazing method (He et al., CVPR 2010) — no extra training needed. CLAHE handles the low-light side by boosting local contrast in the LAB color space without oversaturating colors. Running both in sequence gives a clean frame even under heavy haze or nighttime conditions.

## Dataset

Hard Hat Detection — Kaggle / Andrew MVD
https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

5000 annotated images, classes: `helmet` / `head`, split 80/10/10 into train/val/test.

## Results

| Metric | Score |
|---|---|
| **mAP@0.50** | **94.77%** ✅ |
| **Precision** | **92.46%** ✅ |
| **Recall** | **91.38%** ✅ |
| mAP@0.50:0.95 | 67.22% |
| Model | YOLOv8n |
| Epochs | 50 |

Target accuracy was 86–90% — achieved 94.77% mAP@0.50.

Per class breakdown:

| Class | Images | Precision | Recall | mAP@0.50 |
|---|---|---|---|---|
| helmet | 444 | 0.942 | 0.912 | 0.953 |
| head | 89 | 0.907 | 0.916 | 0.943 |
| **all** | **487** | **0.925** | **0.914** | **0.948** |

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/aristia-ai-task.git
cd aristia-ai-task
pip install -r requirements.txt
jupyter notebook low_visibility_enhancement.ipynb
```

Or open directly in Google Colab and run all cells. GPU recommended (Runtime → Change runtime type → T4 GPU).

## Files

```
low_visibility_enhancement.ipynb   # main notebook
requirements.txt                   # dependencies
README.md                          # this file
enhancement_results.png            # before/after enhancement comparison
inference_results.png              # detection results on degraded frames
```

## What I'd improve with more time

Using a learned enhancement model (like Zero-DCE for night, or AOD-Net for haze) would probably generalize better across different camera types and lighting conditions. The DCP + CLAHE combo is fast and training-free which is why I went with it here, but a learned approach would handle edge cases better.

## References

- He, K. et al. (2010). Single Image Haze Removal Using Dark Channel Prior. CVPR.
- Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- Dataset: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
