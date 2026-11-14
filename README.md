# 🧬 Quantum ML for Pneumonia Classification
### *Exploring the intersection of Quantum Computing, Machine Learning & Medical Imaging*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Quantum%20ML-Qiskit-6929C4" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Experiment%20Tracking-MLflow-0194E2?logo=mlflow" />
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white" />
</p>

---

## 🌟 01. Project Vision

Can quantum computing contribute to the future of medical image classification?

This project explores that question by developing a machine learning pipeline for classifying chest X-ray images into two categories:

- 🟢 **NORMAL** — X-ray images without the target pneumonia classification.
- 🔴 **PNEUMONIA** — X-ray images classified as showing pneumonia.

The project combines classical machine learning experiments with an exploratory Quantum Machine Learning (QML) approach.

It also includes an API-based prediction system, experiment tracking, automated testing, containerization, and cloud deployment.

> **Our objective:** Explore different learning approaches for medical image classification while building a reproducible and deployable ML pipeline.

## 🎯 02. Project Highlights

| Feature | Description |
|---|---|
| 🩻 Image Classification | Classifies chest X-rays into NORMAL and PNEUMONIA |
| 🧠 Classical ML | SVM and Random Forest experiments |
| ⚛️ Quantum ML | Exploratory QSVC-based experiment |
| 📉 Dimensionality Reduction | PCA for feature reduction |
| 📊 Experiment Tracking | MLflow for model metrics and runs |
| 🚀 Prediction API | FastAPI-based image prediction |
| 🐳 Containerization | Docker-based deployment |
| 🧪 Automated Testing | Pytest-based API tests |
| ☁️ Cloud Deployment | Hosted application on Render |

## 🏗️ 03. System Architecture

The project follows a modular machine learning workflow:

```text
             CHEST X-RAY DATASET
                      |
                      v
             IMAGE PREPROCESSING
          Resize | Grayscale | Normalize
                      |
                      v
             FEATURE EXTRACTION
                  PCA (6)
                      |
              +-------+-------+
              |               |
              v               v
        CLASSICAL ML      QUANTUM ML
              |               |
         SVM / RF           QSVC
              |               |
              +-------+-------+
                      |
                      v
              MODEL EVALUATION
                      |
                      v
              MLFLOW TRACKING
                      |
                      v
             SAVED MODEL PIPELINE
                      |
                      v
                FASTAPI APP
                      |
                      v
              X-RAY PREDICTION
```

## 🩻 04. Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

**Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset contains chest X-ray images organized into NORMAL and PNEUMONIA categories.

### Preprocessing pipeline

1. Load chest X-ray images.
2. Convert images to grayscale.
3. Resize images to 64 × 64 pixels.
4. Normalize pixel values.
5. Flatten images into feature vectors.
6. Apply StandardScaler.
7. Apply PCA with 6 components.

## ⚙️ 05. Machine Learning Methodology

### Classical Machine Learning

Three classical models were evaluated:

| Model | Configuration |
|---|---|
| SVM | RBF kernel, C = 1 |
| SVM | RBF kernel, C = 10 |
| Random Forest | 100 estimators, max depth = 15 |

### Quantum Machine Learning

An exploratory Quantum Support Vector Classifier (QSVC) experiment was also conducted.

| Component | Configuration |
|---|---|
| Algorithm | QSVC |
| Feature Map | ZZFeatureMap |
| Qubits | 6 |
| Repetitions | 2 |
| Training Sample | 80 images |
| Validation Sample | 40 images |

The quantum experiment used a smaller sample and should be interpreted separately from the classical model evaluation.

## 📊 06. Experimental Results

### Classical Model Comparison

The following results were obtained from the MLflow-tracked validation runs.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| 🥇 SVM (C=1) | **90.05%** | 92.26% | 89.86% | **91.04%** |
| SVM (C=10) | 89.23% | **92.40%** | 88.12% | 90.21% |
| Random Forest | 88.74% | 89.88% | **90.14%** | 90.01% |

### Key Observations

- SVM with C=1 achieved the highest validation accuracy and F1-score among the three classical models.
- SVM with C=10 achieved the highest precision.
- Random Forest achieved the highest recall.
- Model selection should consider multiple metrics rather than accuracy alone.

### Quantum ML Experiment

The earlier QSVC experiment produced the following results on its smaller validation sample:

| Metric | QSVC Result |
|---|---:|
| Accuracy | 55.00% |
| Precision | 57.58% |
| Recall | 82.61% |
| F1-score | 67.86% |

**Important:** The QSVC and classical results were obtained using different evaluation sample sizes. They are not a controlled, directly comparable benchmark.

## 📈 07. Experiment Tracking with MLflow

MLflow is used to record machine learning experiments.

Tracked information includes:

- Model parameters
- Accuracy
- Precision
- Recall
- F1-score
- Model artifacts

Start the MLflow interface locally:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

`http://127.0.0.1:5000`

## 🚀 08. Application & API

The trained model is integrated into a FastAPI application.

### Live Demo

🌐 **[Open Deployed Application](https://quantum-ml-medical-image-classification.onrender.com/)**

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Application homepage |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/predict` | POST | X-ray image prediction |
| `/predict-page` | GET | Prediction page |

### Prediction Workflow

1. Upload a chest X-ray image.
2. The application preprocesses the image.
3. The saved scaler and PCA transform the image features.
4. The trained model generates a prediction.
5. The API returns the predicted class and confidence.

## 🧪 09. Testing & Quality Assurance

The application includes automated tests using Pytest.

The test suite checks:

- Homepage availability.
- API health endpoint.
- Valid image prediction.
- Missing image input.
- Invalid file content type.

Run the tests:

```bash
python -m pytest test_app.py -v
```

A separate synthetic training smoke test is also available:

```bash
python ci_train.py
```

This checks the training components independently of the actual chest X-ray dataset.

## 🐳 10. Docker Deployment

Build the Docker image:

```bash
docker build -t quantum-ml-pneumonia .
```

Run the container:

```bash
docker run -d -p 10000:10000 --name quantum-pneumonia quantum-ml-pneumonia
```

Open:

`http://localhost:10000`

## 💻 11. Run Locally

### Step 1 — Clone the repository

```bash
git clone https://github.com/ParulRana2907/Quantum-ML-Medical-Image-Classification.git
cd Quantum-ML-Medical-Image-Classification
```

### Step 2 — Install dependencies

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

### Step 3 — Prepare the dataset

Download the dataset from Kaggle and organize it as:

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/
```

### Step 4 — Train the models

```bash
python train.py
```

### Step 5 — Start the API

```bash
uvicorn app:app --reload
```

Open:

`http://127.0.0.1:8000`

## 🛠️ 12. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Classical ML | Scikit-learn |
| Quantum ML | Qiskit, QSVC |
| Dimensionality Reduction | PCA |
| Experiment Tracking | MLflow |
| Backend | FastAPI |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Version Control | Git & GitHub |

## 🔮 13. Future Scope

Potential future improvements include:

- Testing larger and more diverse datasets.
- Evaluating additional quantum feature maps and classifiers.
- Conducting controlled comparisons between classical and quantum approaches.
- Exploring transfer learning and deep learning architectures.
- Improving model interpretability using explainable AI.
- Expanding automated experiment tracking and evaluation.

## ⚠️ 14. Limitations & Responsible Use

- This is an academic research and demonstration project.
- Predictions are not a substitute for professional medical diagnosis.
- Dataset quality, class distribution, and preprocessing can influence performance.
- The quantum experiment uses a smaller sample and requires further evaluation.
- The model should not be used to make clinical decisions.

## 📁 15. Repository Structure

```text
Quantum-ML-Medical-Image-Classification/
│
├── app.py
├── train.py
├── ci_train.py
├── test_app.py
├── requirements.txt
├── requirements-serve.txt
├── Dockerfile
├── README.md
│
├── model/
│   └── best_model_pipeline.joblib
│
├── static/
│   └── index.html
│
├── screenshots/
│
└── .github/
    └── workflows/
        └── tests.yml
```

## 👩‍💻 16. Project Information

**Project:** Quantum ML for Pneumonia Classification Using Chest X-Ray Images

**Domain:** Artificial Intelligence | Machine Learning | Quantum Computing | Healthcare

**Developer:** Parul Rana

**GitHub Repository:** [View Source Code](https://github.com/ParulRana2907/Quantum-ML-Medical-Image-Classification)

**Live Application:** [Try the Deployed Project](https://quantum-ml-medical-image-classification.onrender.com/)

---

<p align="center">
  <b>⚛️ Exploring intelligent healthcare through Classical & Quantum Machine Learning.</b>
  <br/>
  <i>Built as an academic exploration of emerging AI technologies.</i>
</p>