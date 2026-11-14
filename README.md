
# Quantum Machine Learning for Pneumonia Classification

## 1. Project Overview

This project explores Quantum Machine Learning for classifying chest X-ray images into two categories:
- NORMAL
- PNEUMONIA

The project also compares quantum and classical machine learning approaches.

## 2. Dataset

Dataset: Chest X-Ray Images (Pneumonia)

Source: Kaggle

The dataset contains chest X-ray images organized into NORMAL and PNEUMONIA classes.

## 3. Technologies Used

- Python
- NumPy and Pandas
- OpenCV and Pillow
- Scikit-learn
- Qiskit
- MLflow
- FastAPI
- Pytest
- Docker
- GitHub Actions

## 4. Machine Learning Workflow

1. Load chest X-ray images.
2. Resize and normalize images.
3. Perform exploratory data analysis.
4. Extract features using PCA.
5. Train classical machine learning models.
6. Train a quantum machine learning model.
7. Compare model performance.
8. Save the selected model.

## 5. Experiment Tracking

MLflow is used to track model experiments and evaluation metrics.

## 6. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| / | GET | Homepage |
| /health | GET | Health check |
| /predict | POST | Image classification |

## 7. Docker

Build:

docker build -t quantum-ml-pneumonia .

Run:

docker run -d -p 8000:8000 --name quantum-ml-container quantum-ml-pneumonia

## 8. Testing

Run:

python -m pytest test_app.py -v

## 9. Limitations

This project is an academic machine learning demonstration.
Its predictions are not intended for clinical diagnosis.
