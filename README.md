# Pneumonia Detection on Chest X-Ray Images — Python Implementation (InceptionV3)

This repository contains the **Python implementation** for detecting pneumonia from chest X-ray images using the **InceptionV3** model (via `timm`). The goal is to provide fully reproducible code, methodological explanations, experiment reproduction steps, and evaluation results ready to be published on GitHub.

> **Note:** This repository focuses solely on the Python version. The Rust implementation is available separately.

---

## Recommended Repository Structure

```
pneumonia-inception-python/
├─ data/                       # (excluded) dataset folder (see Data section)
├─ scripts/
│  └─ train_inception_timm_chestxray.py
├─ models/
│  └─ best_inception_v3_timm.pth  # best model automatically saved during training
├─ notebooks/                   # optional: experiments / visualizations
├─ requirements.txt
├─ README.md                    # (this file)
└─ LICENSE
```

---

## Overview

* **Model:** InceptionV3 (via `timm`)
* **Language / Framework:** Python 3.8+ / PyTorch + timm + torchvision
* **Task:** Binary classification — *Normal* vs *Pneumonia*
* **Dataset:** Chest X-Ray Pneumonia (Kaggle) — structured as `train/`, `test/`, optionally `valid/`
* **Training Method:** 2-phase approach — (1) freeze backbone, train classifier; (2) unfreeze selected layers for fine-tuning

---

## Requirements (`requirements.txt`)

```
torch>=1.13
torchvision
numpy
pillow
timm
tqdm
```

> Adjust the `torch` version according to your CUDA setup. On Kaggle or Colab, `torch` and `torchvision` are usually preinstalled.

---

## Data Instructions

1. Download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle: `paultimothymooney/chest-xray-pneumonia`.
2. Expected folder structure:

```
data/
└─ chest_xray/
   ├─ train/
   │  ├─ NORMAL/
   │  └─ PNEUMONIA/
   ├─ test/
   │  ├─ NORMAL/
   │  └─ PNEUMONIA/
   └─ val/ (optional)
```

3. Set `DATA_DIR` in `train_inception_timm_chestxray.py` to your dataset location. The default path in the script points to `/kaggle/input/chest-xray-pneumonia/chest_xray`.

---

## How to Run (Training & Evaluation)

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. Run the training script:

```bash
python scripts/train_inception_timm_chestxray.py
```

3. The best model will be automatically saved to `./Model/best_inception_v3_timm.pth`.
4. Final evaluation metrics on the test set will be printed in the terminal.

---

## Code Explanation (train_inception_timm_chestxray.py)

* **Transforms & Augmentation:** Uses `timm.data.create_transform` for model-specific preprocessing and adds mild augmentations (90° rotation, vertical flip) to the training set.
* **Class weights:** Dynamically computed from data distribution to handle class imbalance.
* **Phased training:** Phase 1 (freeze backbone, train classifier) → Phase 2 (unfreeze selected Inception blocks for fine-tuning).
* **Numerical stability:** Applies `torch.nan_to_num` and `clamp` to prevent NaN or extreme values.
* **Mixed precision:** Enabled through `torch.cuda.amp` when GPU is available to improve speed.

---

## Mathematical Formulas (for presentation)

* **2D Convolution:**

$$
S(x, y) = (I * K)(x, y) = \sum_{a=-m}^{m} \sum_{b=-n}^{n} I(x+a, y+b) \cdot K(a, b)
$$

* **Binary Cross-Entropy Loss:**

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

* **Evaluation Metrics:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = 2 \cdot \frac{P \cdot R}{P + R} \quad ; \quad F_2 = 5 \cdot \frac{P \cdot R}{4P + R}$$

---

## Experimental Results (Python)

Final results obtained using GPU (CUDA):

* **Accuracy:** 90.54%
* **Precision:** 96.36%
* **Recall:** 88.21%
* **F2-score:** 89.72%
* **Total time:** 905.52 s

> These metrics are based on the Kaggle GPU runtime. Results may vary slightly across different environments.

---

## Practical Tips & Notes

* **GPU usage is highly recommended** — InceptionV3 training on CPU is very slow.
* **Pretrained weights:** Using ImageNet pretrained weights speeds up convergence and improves accuracy.
* **Small validation set:** If the `val/` split contains few images (e.g., 16), you can use the `test/` set as validation or apply cross-validation.
* **Reproducibility:** The script fixes random seeds and includes deterministic settings for repeatable results.

---

## License

Default license: **MIT**. You may replace or modify it based on your preference.

---

## Push to GitHub (Quick Guide)

```bash
# in repository root
git init
git add .
git commit -m "Initial commit: InceptionV3 pneumonia detection (Python)"
# create a new repo on GitHub, then link and push
git remote add origin git@github.com:<username>/pneumonia-inception-python.git
git push -u origin main
```
