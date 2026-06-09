# Pneumonia Detection from Chest X-Rays
> A deep learning project that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN).
---

## About
A CNN-based binary classifier that detects pneumonia from chest X-ray images, trained on grayscale scans with data augmentation to handle class imbalance and overfitting.

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | Chest X-Ray Images (Pneumonia) — Kaggle |
| **Classes** | `PNEUMONIA` / `NORMAL` |
| **Image Size** | Resized to 150×150 pixels (grayscale) |
| **Splits** | Train / Validation / Test |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | Model building and training |
| `OpenCV (cv2)` | Image loading and resizing |
| `NumPy`, `Pandas` | Data handling |
| `Matplotlib`, `Seaborn` | Visualization |
| `scikit-learn` | Classification report and evaluation |
| `ffmpeg` | Training animation export |

---

## Preprocessing & Augmentation

| Step | Detail |
|---|---|
| **Grayscale Conversion** | All images read as single-channel grayscale |
| **Resize** | Images resized to 150×150 pixels |
| **Normalization** | Pixel values scaled to `[0, 1]` by dividing by 255 |
| **Reshape** | Reshaped to `(samples, 150, 150, 1)` for CNN input |
| **Rotation** | Random rotation up to 30° |
| **Zoom** | Random zoom up to 20% |
| **Width / Height Shift** | Random shift up to 10% |
| **Horizontal Flip** | Randomly flipped horizontally |

---

## Model Architecture

| Layer | Details |
|---|---|
| `Conv2D` (32 filters) | 3×3 kernel, ReLU, same padding |
| `BatchNormalization` | Normalize activations |
| `MaxPool2D` | 2×2 pool, stride 2 |
| `Conv2D` (64 filters) + `Dropout(0.1)` | 3×3 kernel, ReLU |
| `BatchNormalization` + `MaxPool2D` | — |
| `Conv2D` (64 filters) | 3×3 kernel, ReLU |
| `BatchNormalization` + `MaxPool2D` | — |
| `Conv2D` (128 filters) + `Dropout(0.2)` | 3×3 kernel, ReLU |
| `BatchNormalization` + `MaxPool2D` | — |
| `Conv2D` (256 filters) + `Dropout(0.2)` | 3×3 kernel, ReLU |
| `BatchNormalization` + `MaxPool2D` | — |
| `Flatten` | — |
| `Dense` (128 units, ReLU) + `Dropout(0.2)` | Fully connected |
| `Dense` (1 unit, Sigmoid) | Binary output |

```text
Input (150×150×1)
       │
       ▼
┌─────────────────────┐
│ Conv2D (32)         │
│ BatchNorm + MaxPool │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Conv2D (64)         │
│ Dropout + MaxPool   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Conv2D (64)         │
│ BatchNorm + MaxPool │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Conv2D (128)        │
│ Dropout + MaxPool   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Conv2D (256)        │
│ Dropout + MaxPool   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Flatten             │
│ Dense(128) + Drop   │
│ Dense(1) + Sigmoid  │
└──────────┬──────────┘
           │
           ▼
  PNEUMONIA / NORMAL
```

---

## 📈 Training

| Parameter | Value |
|---|---|
| **Optimizer** | RMSProp |
| **Loss Function** | Binary Crossentropy |
| **Epochs** | 12 |
| **Batch Size** | 32 |
| **Callback** | ReduceLROnPlateau (patience=2, factor=0.3) |

---

## 📊 Evaluation

| Metric | Detail |
|---|---|
| **Accuracy** | Evaluated on test set |
| **Loss** | Binary crossentropy on test set |
| **Classification Report** | Precision, Recall, F1 for each class |
| **Correct Predictions** | Sample X-rays with correct labels visualized |
| **Incorrect Predictions** | Sample X-rays with wrong labels visualized |

---

## Project Structure
| File | Description |
|---|---|
| `Untitled1.ipynb` | Main Jupyter notebook |
| `chest_xray/` | Dataset folder (train / val / test) |
| `README.md` | Project documentation |

---

## Getting Started
| Step | Command |
| **1. Clone the repo** | `git clone https://github.com/your-username/pneumonia-detection.git` |
| **2. Install dependencies** | `pip install tensorflow keras opencv-python matplotlib seaborn scikit-learn` |
| **3. Download dataset** | [Chest X-Ray Images on Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) |
| **4. Launch notebook** | `jupyter notebook Untitled1.ipynb` |
> **Note:** Update the dataset path in the notebook to point to your local `chest_xray/` folder.
---

## License
This project is for academic and educational purposes.
