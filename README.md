# 🧠 NeuroScan-AI: Brain Tumor MRI Classification

Deep learning pipeline for detecting and classifying brain tumor types from MRI scans into 4 categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

---

## 📖 Interactive Walkthrough Notebook

To understand every stage of the project with interactive code, diagrams, dataset integrity inspection, and Grad-CAM visualizations, open:
👉 **[`notebooks/01_project_walkthrough.ipynb`](notebooks/01_project_walkthrough.ipynb)**

The notebook is pre-executed with all output plots and tables rendered.

---

## 🏗️ Architecture & Pipeline Overview

```
 raw dataset folder ──► 1. discover & inspect ──► 2. find duplicates ──► 3. leakage-safe split
   (read-only)            (src/data_loader.py)      (pHash + MD5 + stem)     (70 / 15 / 15, grouped)
                                                                                  │
 6. Grad-CAM  ◄── 5. evaluate on TEST ◄── 4. train on TRAIN, select on VAL ◄──────┘
 (src/gradcam.py) (src/evaluate.py)       (src/train.py: Custom CNN | EfficientNetB0)
```

1. **Read-Only Dataset Discovery (`src/data_loader.py`)**:
   - Discovers image files across Roboflow-exported splits.
   - Extracts original image stems to detect Roboflow-augmented clones.

2. **Integrity & Duplicate Grouping (`src/data_loader.py`)**:
   - Performs MD5 byte checking and 64-bit DCT perceptual hashing (`pHash`).
   - Clusters identical and near-duplicate images using Union-Find to avoid data leakage between splits.

3. **Leakage-Safe Stratified Splitting (`src/prepare_data.py`)**:
   - Uses `StratifiedGroupKFold` (20 folds → 14 train / 3 val / 3 test: 70% / 15% / 15%).
   - Guarantees images from the same scan or duplicate cluster never cross split boundaries.

4. **Models & Training (`src/models/`, `src/train.py`)**:
   - **Custom CNN**: 4-block VGG-style CNN with BatchNormalization, Dropout, and GlobalAveragePooling trained from scratch.
   - **EfficientNetB0**: 2-stage transfer learning (Stage A: frozen backbone feature extraction; Stage B: unfreeze top blocks with frozen BatchNorm).
   - Selection is done strictly on **Validation macro-F1**.

5. **Test Evaluation & Benchmarking (`src/evaluate.py`, `src/compare_models.py`)**:
   - Evaluates once on the held-out test split.
   - Computes accuracy, macro/weighted F1, confusion matrices, model size, and inference latency.

6. **Explainability (`src/gradcam.py`)**:
   - Gradient-weighted Class Activation Mapping (Grad-CAM) to inspect convolutional feature activations and sanity-check model attention.

---

## 🚀 Quickstart Commands

### 1. Run Tests
```bash
python -m pytest -q
```

### 2. Inspect & Prepare Dataset (Generate Splits)
```bash
python -m src.explore
python -m src.prepare_data
```

### 3. Train Models
```bash
# Fast smoke test (CPU sanity check)
python -m src.train --model custom_cnn --smoke

# Full training
python -m src.train --model custom_cnn
python -m src.train --model efficientnet
```

### 4. Evaluate on Test Split
```bash
python -m src.evaluate --model efficientnet --split test
python -m src.compare_models
```

### 5. Generate Grad-CAM Visualizations
```bash
python -m src.gradcam --model efficientnet --n 8
```

---

## 📁 Repository Structure

```
.
├── notebooks/
│   └── 01_project_walkthrough.ipynb  # Complete interactive walkthrough with outputs
├── src/
│   ├── config.py                     # Central configuration & hyperparameters
│   ├── data_loader.py                # Dataset parsing, hashing & leakage-safe grouping
│   ├── preprocessing.py              # Resizing, tf.data pipeline & class weighting
│   ├── augmentation.py               # Anatomy-preserving training augmentations
│   ├── models/
│   │   ├── custom_cnn.py             # 4-block CNN architecture
│   │   └── efficientnet.py           # EfficientNetB0 transfer learning setup
│   ├── train.py                      # Training loop with callbacks & stage selection
│   ├── evaluate.py                   # Test metrics, confusion matrix & inference timing
│   ├── compare_models.py             # Markdown report generator comparing models
│   └── gradcam.py                    # Grad-CAM heatmap generation & overlays
├── tests/                            # Comprehensive pytest test suite
└── artifacts/                        # Splits, models, logs, metrics & figures
```
