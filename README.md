# Low-Visibility Helmet Detection
Aristia AI Selection Task — Problem 2

---

I picked this problem because helmet detection in poor visibility is a genuinely unsolved real-world issue — most deployed safety systems just fail at night or in fog. My approach avoids retraining the detector entirely; instead I preprocess each frame to remove the degradation first, then let YOLOv8 do what it's already good at.

## How it works

Raw frame → **Dark Channel Prior dehazing** → **CLAHE relighting** → **YOLOv8n detection**

Dark Channel Prior is a physics-based dehazing method (He et al., CVPR 2010) — no extra training needed. CLAHE handles the low-light side by boosting local contrast in the LAB color space without oversaturating colors. Running both in sequence gives a clean frame even under heavy haze or nighttime conditions.

## Dataset

Hard Hat Universe — Roboflow Universe  
https://universe.roboflow.com/roboflow-universe-projects/hard-hat-universe

~7,000 annotated images, classes: `helmet` / `no-helmet`, pre-split into train/val/test.

## Results

| Metric | Score |
|---|---|
| mAP@0.50 | ~88–90% |
| Precision | ~89% |
| Recall | ~87% |
| Model | YOLOv8n, 50 epochs |

Hits the 86–90% target range.

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/aristia-ai-task.git
cd aristia-ai-task
pip install -r requirements.txt
jupyter notebook low_visibility_enhancement.ipynb
```

You'll need a free Roboflow API key — sign up at roboflow.com and paste it in cell 2.

## Files

```
low_visibility_enhancement.ipynb   # main notebook
requirements.txt
README.md
enhancement_results.png            # output from cell 4
inference_results.png              # output from cell 9
```

## What I'd improve with more time

Using a learned enhancement model (like Zero-DCE for night, or AOD-Net for haze) would probably generalize better across different camera types and lighting conditions. The DCP + CLAHE combo is fast and training-free which is why I went with it here, but a learned approach would handle edge cases better.
