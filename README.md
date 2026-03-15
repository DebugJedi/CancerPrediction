# Breast Cancer Predictor

> **Neural network classifier for breast mass malignancy prediction — interactive Streamlit UI with real-time radar chart visualization.**  
> PyTorch · Streamlit · Plotly · scikit-learn · Wisconsin Breast Cancer Dataset

---

## What This Is

A fully interactive breast cancer prediction app built on a custom PyTorch neural network trained on the **Wisconsin Breast Cancer Dataset (UCI)**. Users adjust 30 biopsy measurements via sliders and get an instant benign/malignant prediction with probability scores — visualized on a live radar chart.

Built to demonstrate end-to-end ML: data cleaning → model architecture → training → serialization → interactive deployment.

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **97.37%** |
| Training Accuracy | 97.58% |
| Precision (Benign) | 0.97 |
| Recall (Benign) | 0.99 |
| F1 (Benign) | 0.98 |
| Precision (Malignant) | 0.98 |
| Recall (Malignant) | 0.95 |
| F1 (Malignant) | 0.97 |
| Weighted avg F1 | **0.97** |
| Test set size | 114 samples |

> Training converged smoothly from 87.9% accuracy at epoch 10 to 97.6% at epoch 100 with BCELoss dropping from 0.52 → 0.11.

---

## How It Works

```
Wisconsin Breast Cancer Dataset (569 samples · 30 features)
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing │  Drop ID/unnamed cols · encode M→1, B→0
│  StandardScaler     │  80/20 train/test split · feature scaling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Neural Network     │  Input(30) → Linear → ReLU
│  (PyTorch)          │  → Linear(54) → Sigmoid → Output(1)
│                     │  BCELoss · Adam(lr=0.001) · 100 epochs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Persistence  │  pickle → model.pkl + scaler.pkl
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit App      │  30 sliders · Plotly radar chart
│                     │  Real-time prediction + probability scores
└─────────────────────┘
```

---

## Model Architecture

```python
NeuralNet(
  (fc1):     Linear(in=30, out=54)
  (relu):    ReLU()
  (fc2):     Linear(in=54, out=1)
  (sigmoid): Sigmoid()
)
```

| Hyperparameter | Value |
|---|---|
| Input features | 30 (cell nucleus measurements) |
| Hidden layer size | 54 |
| Output | 1 (sigmoid — malignancy probability) |
| Loss function | Binary Cross Entropy (`BCELoss`) |
| Optimizer | Adam (lr = 0.001) |
| Epochs | 100 |
| Train/test split | 80/20 · random_state=32 |

---

## Dataset

**Wisconsin Breast Cancer Dataset** — 569 samples, 30 numerical features computed from digitized FNA (fine needle aspirate) images of breast masses.

Features are grouped into 3 measurement types across 10 cell nucleus characteristics:

| Group | Features |
|---|---|
| Mean | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| Standard Error | Same 10 characteristics |
| Worst | Same 10 characteristics (largest mean of 3 worst values) |

**Class distribution:** 357 Benign (62.7%) · 212 Malignant (37.3%)

---

## App Features

- **30 interactive sliders** — one per feature, min/max derived from dataset range, default at feature mean
- **Live radar chart** — three overlapping traces (Mean / Standard Error / Worst) update with every slider change using Plotly
- **Real-time prediction** — benign or malignant classification with probability score for each class
- **Min-max normalization** — slider values normalized to [0,1] before inference to match training distribution
- **GPU/CPU auto-detection** — runs on CUDA if available, falls back to CPU

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/DebugJedi/CancerPrediction.git
cd CancerPrediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run cancer_prediction.py
# → http://localhost:8501
```

---

## Train the Model Yourself

```bash
python model/main.py
```

This will:
1. Load and clean `Datasets/data.csv`
2. Scale features with `StandardScaler`
3. Train for 100 epochs, printing loss + accuracy every 10 epochs
4. Print test accuracy and full classification report
5. Save `model.pkl` and `scaler.pkl` to `resources/`

---

## Project Structure

```
CancerPrediction/
├── cancer_prediction.py    ←  Streamlit app · UI · radar chart · prediction
├── model/
│   └── main.py             ←  NeuralNet class · training · evaluation · save
├── Datasets/
│   └── data.csv            ←  Wisconsin Breast Cancer Dataset
├── resources/
│   ├── model.pkl           ←  Trained PyTorch model (serialized)
│   └── scaler.pkl          ←  Fitted StandardScaler
├── assets/
│   └── style.css           ←  Custom Streamlit styling
├── .streamlit/             ←  Streamlit config
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Neural network | PyTorch (`nn.Module`) |
| Data processing | pandas · scikit-learn (`StandardScaler`, `train_test_split`) |
| Visualization | Plotly (`Scatterpolar` radar chart) |
| UI framework | Streamlit |
| Model persistence | pickle |
| Evaluation | scikit-learn (`accuracy_score`, `classification_report`) |

---

## Roadmap

- [x] Custom PyTorch neural network
- [x] Interactive Streamlit UI with 30 feature sliders
- [x] Real-time radar chart visualization
- [x] Probability scores for each class
- [x] Model and scaler persistence
- [ ] Cross-validation and hyperparameter tuning
- [ ] ROC curve and AUC visualization
- [ ] SHAP feature importance explanations
- [ ] Streamlit Cloud deployment

---

## Author

Built and maintained by **Priyank Rao** — Data Scientist / ML Engineer  
[Portfolio](https://priyankrao.co) · [GitHub](https://github.com/DebugJedi)

---

> **Disclaimer:** This app is for educational and research purposes only. It is not a substitute for professional medical diagnosis.