# 🍎 Automated Fruit Freshness Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework and comparative study for automated binary fruit freshness detection (**Fresh** vs. **Rotten**). The study compares **MobileNetV2** (inverted residual bottlenecks) against **EfficientNetB0** (compound scaling) on 13,599 images, accompanied by a production-ready **FastAPI** backend and an interactive **Streamlit** web application.

---

## 👥 Authors & Affiliation
**Panipat Institute of Engineering and Technology (PIET), Panipat, India**  
*Department of Computer Science & Engineering*

- **Ayush Kumar** — `ayushsyntax@gmail.com`
- **Bhushan Verma** — `vermabhushan004@gmail.com`
- **Jayant Jain** — `jayantjain058@gmail.com`
- **Dr. Mitu Sehgal** — `technomitusehgal@gmail.com`

---

## 🏆 Comparative Benchmark Results

Both architectures were trained and evaluated on 13,599 multi-fruit images (8,721 train / 2,180 validation / 2,698 test):

| Metric | MobileNetV2 (Champion) | EfficientNetB0 | Margin |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **97.96%** | 49.89% | **+48.07%** |
| **Precision (Macro)** | **0.9789** | 0.7087 | **+0.2702** |
| **Recall (Macro)** | **0.9796** | 0.5584 | **+0.4212** |
| **F1-Score (Macro)** | **0.9792** | 0.4256 | **+0.5536** |
| **ROC-AUC Score** | **0.9985** | 0.7475 | **+0.2510** |
| **Pearson Correlation (Ground Truth)** | **0.9695** | 0.4110 | **+0.5585** |
| **Total Parameters** | **2,608,577** | 4,400,164 | **-40.72% (Lighter)** |
| **Model Disk Size** | **13.11 MB** | 20.11 MB | **-34.81% (More Compact)** |

---

## 📊 Evaluation Visuals

The project includes complete evaluation figures:
- `fig_dataset_samples.png`: Sample Fresh vs Rotten fruits
- `fig_architecture.png`: Deep Transfer Learning Architecture
- `fig_workflow.png`: End-to-End System Workflow
- `fig_curves.png`: Training & Validation Accuracy and Loss Curves
- `fig_confusion_matrix.png`: Side-by-Side Confusion Matrix Heatmaps
- `fig_roc_curve.png`: ROC-AUC Curves Comparison
- `fig_correlation.png`: Prediction Correlation Matrix Heatmap
- `fig_predictions.png`: Visual Predictions on Sample Test Images
- `img.png`: Master Collage Evaluation Dashboard

---

## 🚀 Quickstart Guide

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/your-username/fruit-freshness-classifier.git
cd fruit-freshness-classifier

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the FastAPI Microservice
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
- Swagger Interactive Documentation: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

### 3. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
*Note: The Streamlit app includes automatic fallback to direct local inference if the FastAPI server is offline.*

### 4. Run the Jupyter Notebook
```bash
jupyter notebook code.ipynb
```

---

## 📄 Research Paper Publication

The research paper associated with this project:
- **Title**: *Deep Learning-Based Automated Fruit Freshness Classification: A Comparative Study of MobileNetV2 and EfficientNetB0*
- **Authors**: Ayush Kumar, Bhushan Verma, Jayant Jain, and Dr. Mitu Sehgal
- **Affiliation**: Panipat Institute of Engineering and Technology (PIET)
- **Conference Format**: IEEE 2-Column Standard (5 Pages)

