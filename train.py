
import argparse
import logging
from pathlib import Path

import cv2
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

MODELS = {
    "SVM_C1": lambda: SVC(
        kernel="rbf",
        C=1,
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "SVM_C10": lambda: SVC(
        kernel="rbf",
        C=10,
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Random_Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. LOAD AND PREPROCESS IMAGES
# --------------------------------------------------

def load_dataset(split, data_dir, image_size):

    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):

        folder = data_dir / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        image_paths = sorted(folder.glob("*"))

        logger.info(
            "Loading %s images from %s",
            class_name,
            folder
        )

        for image_path in image_paths:

            if not image_path.is_file():
                continue

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                logger.warning(
                    "Skipping unreadable image: %s",
                    image_path
                )
                continue

            image = cv2.resize(
                image,
                (image_size, image_size)
            )

            image = image.astype(np.float32) / 255.0

            images.append(image.flatten())
            labels.append(label)

    if not images:
        raise ValueError(
            f"No valid images found in {data_dir / split}"
        )

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


# --------------------------------------------------
# 3. EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# --------------------------------------------------
# 4. MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train classical ML models for pneumonia classification"
    )

    parser.add_argument(
        "--data-dir",
        default="data/chest_xray",
        help="Path to the chest X-ray dataset"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=6
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    logger.info("Starting training pipeline")
    logger.info("Dataset directory: %s", data_dir)

    # ----------------------------------------------
    # Load training and official test datasets
    # ----------------------------------------------

    X_train_full, y_train_full = load_dataset(
        "train",
        data_dir,
        args.image_size
    )

    X_test, y_test = load_dataset(
        "test",
        data_dir,
        args.image_size
    )

    logger.info("Training data shape: %s", X_train_full.shape)
    logger.info("Test data shape: %s", X_test.shape)

    # ----------------------------------------------
    # Training-validation split
    # ----------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_train_full
    )

    logger.info("Training samples: %d", len(y_train))
    logger.info("Validation samples: %d", len(y_val))

    # ----------------------------------------------
    # Standardization
    # ----------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------------
    # PCA
    # ----------------------------------------------

    if args.pca_components > min(X_train_scaled.shape):
        raise ValueError(
            "PCA components cannot exceed the available "
            "training samples or features."
        )

    pca = PCA(
        n_components=args.pca_components,
        random_state=args.random_state
    )

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    logger.info("PCA completed successfully")

    # ----------------------------------------------
    # MLflow configuration
    # ----------------------------------------------

    mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(
        "Quantum_ML_Pneumonia_Classification"
    )

    # ----------------------------------------------
    # Train and evaluate models
    # ----------------------------------------------

    results = []
    trained_models = {}

    for model_name, model_factory in MODELS.items():

        logger.info("Training model: %s", model_name)

        model = model_factory()

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train_pca, y_train)

            val_predictions = model.predict(X_val_pca)

            metrics = calculate_metrics(
                y_val,
                val_predictions
            )

            # Log parameters
            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("image_size", args.image_size)
            mlflow.log_param(
                "pca_components",
                args.pca_components
            )
            mlflow.log_param("random_state", args.random_state)

            if model_name.startswith("SVM"):
                mlflow.log_param("kernel", "rbf")
                mlflow.log_param("C", model.C)
            else:
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 15)

            # Log validation metrics
            mlflow.log_metrics(metrics)

            # Save model in MLflow
            if model_name == "Random_Forest":

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ]
                )

            else:

               mlflow.sklearn.log_model(
                   sk_model=model,
                   name="model"
            )

            results.append({
                "name": model_name,
                **metrics
            })

            trained_models[model_name] = model

            logger.info(
                "%s validation metrics: %s",
                model_name,
                metrics
            )

    # ----------------------------------------------
    # Select best model using validation F1-score
    # ----------------------------------------------

    best_result = max(
        results,
        key=lambda result: result["f1_score"]
    )

    best_model_name = best_result["name"]
    best_model = trained_models[best_model_name]

    logger.info("Best model: %s", best_model_name)
    logger.info(
        "Best validation F1-score: %.4f",
        best_result["f1_score"]
    )

    # ----------------------------------------------
    # Evaluate selected model on official test set
    # ----------------------------------------------

    test_predictions = best_model.predict(X_test_pca)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    logger.info("Final test metrics: %s", test_metrics)

    # ----------------------------------------------
    # Save complete model bundle
    # ----------------------------------------------

    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": best_model,
        "scaler": scaler,
        "pca": pca,
        "image_size": args.image_size,
        "class_names": CLASS_NAMES
    }

    model_path = model_dir / "best_model_pipeline.joblib"

    joblib.dump(model_bundle, model_path)

    logger.info("Saved model to: %s", model_path)

    print("\n========== MODEL COMPARISON ==========")

    for result in results:
        print(
            f"{result['name']}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1_score']:.4f}"
        )

    print("\n========== BEST MODEL ==========")
    print("Model:", best_model_name)
    print("Validation F1:", round(best_result["f1_score"], 4))

    print("\n========== OFFICIAL TEST RESULTS ==========")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
