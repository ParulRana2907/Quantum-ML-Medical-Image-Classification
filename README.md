# Quantum Machine Learning for Pneumonia Classification Using Chest X-Ray Images

## 1. Project Overview

This project explores the application of Quantum Machine Learning (QML) for medical image classification, specifically identifying pneumonia from chest X-ray images.

The project classifies images into two categories:

- **NORMAL:** Chest X-ray images without pneumonia.
- **PNEUMONIA:** Chest X-ray images showing pneumonia-related patterns.

The project investigates quantum machine learning techniques and compares their performance with classical machine learning algorithms.

It also includes an end-to-end machine learning workflow with experiment tracking, model deployment, API development, automated testing and containerization.

**Note:** This project is an academic research and demonstration project. Its predictions are not intended for clinical diagnosis.

## 2. Problem Statement

Pneumonia is a respiratory infection that can be associated with visible patterns in chest X-ray images.

The objective of this project is to develop and evaluate machine learning approaches that classify chest X-ray images into NORMAL and PNEUMONIA categories.

The project also investigates whether quantum machine learning approaches can provide useful classification performance compared with classical machine learning algorithms.

## 3. Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

**Source:** Kaggle

**Dataset link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains chest X-ray images organized into the following classes:

- NORMAL
- PNEUMONIA

The dataset is organized into training, testing and validation directories.

Expected directory structure:

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

The dataset is not included in the GitHub repository. Users must obtain it from Kaggle and place it in the expected directory structure.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy and Pandas | Data processing |
| OpenCV and Pillow | Image preprocessing |
| Scikit-learn | Classical machine learning |
| Qiskit | Quantum machine learning experiments |
| MLflow | Experiment tracking |
| FastAPI | Model serving and API development |
| Pytest | Automated testing |
| Docker | Application containerization |
| GitHub Actions | Continuous integration |

## 5. Machine Learning Workflow

The project follows these major stages:

1. Load chest X-ray images from the dataset.
2. Convert images to grayscale.
3. Resize images to 64 × 64 pixels.
4. Normalize pixel values.
5. Flatten images into feature vectors.
6. Apply StandardScaler for feature standardization.
7. Apply Principal Component Analysis (PCA) for dimensionality reduction.
8. Train classical machine learning algorithms.
9. Evaluate a quantum machine learning classifier.
10. Compare model performance using evaluation metrics.
11. Track experiments using MLflow.
12. Save the selected trained model.
13. Integrate the trained model with FastAPI.
14. Test and deploy the application.

## 6. Machine Learning Models

### Classical Machine Learning

The following classical algorithms are evaluated:

- Support Vector Machine (SVM) with RBF kernel and C = 1.
- Support Vector Machine (SVM) with RBF kernel and C = 10.
- Random Forest Classifier.

### Quantum Machine Learning

A quantum Support Vector Classifier (QSVC) is explored using Qiskit.

The quantum experiment uses:

- Six qubits.
- ZZFeatureMap.
- Two feature-map repetitions.
- PCA-reduced input features.

The quantum model is evaluated as an experimental comparison with the classical approaches.

**Evaluation note:** The quantum experiment uses a smaller image sample because quantum kernel evaluation is computationally expensive. Its results should therefore be interpreted separately from the full-data classical experiments.

## 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

### Classical Model Comparison

## Model Performance Comparison

The following table presents the validation performance
of the three classical machine learning models trained
during the experiment.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SVM (C=1) | 90.05% | 92.26% | 89.86% | 91.04% |
| SVM (C=10) | 89.23% | 92.40% | 88.12% | 90.21% |
| Random Forest | 88.74% | 89.88% | 90.14% | 90.01% |

Based on validation F1-score and accuracy, SVM (C=1)
achieved the best overall performance among these
three classical models.

### Quantum Model Results

Add the quantum model's accuracy, precision, recall and F1-score from the notebook experiment.

Clearly mention that it was evaluated on a smaller sample.

### Best Model Selection

The final classical model is selected based on validation performance, particularly the F1-score.

The selected model, preprocessing scaler and PCA transformation are saved together in:

```text
model/best_model_pipeline.joblib
```

## 8. Experiment Tracking with MLflow

MLflow is used to record and compare machine learning experiments.

The tracked information includes:

- Algorithm names
- Model hyperparameters
- Evaluation metrics
- Model artifacts

To launch the MLflow interface:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

http://127.0.0.1:5000

## 9. Application and API Development

FastAPI is used to create an application that accepts chest X-ray images and returns model predictions.

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Application homepage |
| `/health` | GET | Application health check |
| `/docs` | GET | Interactive API documentation |
| `/predict` | POST | Image classification |

The prediction API returns the predicted class and confidence information.

### Live Application

https://quantum-ml-medical-image-classification.onrender.com/

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/ParulRana2907/Quantum-ML-Medical-Image-Classification.git
cd Quantum-ML-Medical-Image-Classification
```

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3: Download the dataset

Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle and place it under:

```text
data/chest_xray/
```

### Step 4: Train the models

```bash
python train.py
```

### Step 5: Run the application

```bash
python -m uvicorn app:app --reload
```

Open:

http://127.0.0.1:8000/

## 11. Docker Implementation

The application is containerized using Docker.

### Build the Docker image

```bash
docker build -t quantum-ml-pneumonia .
```

### Run the Docker container

```bash
docker run -d -p 10000:10000 --name quantum-pneumonia quantum-ml-pneumonia
```

### Access the application

```text
http://localhost:10000/
```

The serving container uses `requirements-serve.txt` to install its runtime dependencies.

## 12. Testing and Continuous Integration

Pytest is used to verify the application.

Run the test suite:

```bash
python -m pytest test_app.py -v
```

The application tests cover:

- Homepage availability
- Health endpoint
- Valid image prediction
- Missing image input
- Invalid file type

The project also includes a GitHub Actions workflow that performs a synthetic-data training smoke test and runs the automated application tests.

The synthetic CI training test is separate from the actual chest X-ray model training.

## 13. GitHub Repository

Project source code:

https://github.com/ParulRana2907/Quantum-ML-Medical-Image-Classification

## 14. Screenshots

The `screenshots/` directory contains screenshots of the project implementation and results.

Recommended screenshots include:

- Dataset structure
- Model training output
- MLflow experiment comparison
- FastAPI homepage
- API documentation
- Image prediction results
- Docker container running
- GitHub Actions successful workflow
- Live Render application

## 15. Limitations

- Quantum machine learning experiments use a smaller dataset sample because of computational limitations.
- The quantum and classical model results are not directly comparable when evaluated on different sample sizes.
- Model performance depends on dataset quality and preprocessing.
- The application is an academic demonstration and is not clinically validated.
- Predictions should not be used as a substitute for professional medical evaluation.

## 16. Future Scope

- Evaluate additional quantum machine learning algorithms.
- Explore quantum feature maps and different circuit configurations.
- Improve model generalization using larger datasets.
- Investigate hybrid quantum-classical architectures.
- Perform more extensive evaluation on independent datasets.
- Improve experiment reproducibility and model monitoring.

## 17. Conclusion

This project explores Quantum Machine Learning for chest X-ray image classification and compares it with classical machine learning algorithms.

It combines image preprocessing, dimensionality reduction, model training, quantum experimentation, MLflow tracking, FastAPI development, Docker containerization and automated testing.

The project provides an academic foundation for further exploration of quantum machine learning in medical image classification.
