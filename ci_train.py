
"""Small synthetic-data training smoke test for GitHub Actions."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from pathlib import Path

# Generate deterministic synthetic data for CI only
X, y = make_classification(
    n_samples=100,
    n_features=16,
    n_informative=8,
    n_redundant=2,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=6)),
    ("classifier", SVC(kernel="rbf", probability=True))
])

pipeline.fit(X, y)

predictions = pipeline.predict(X)
assert len(predictions) == len(y)

print("CI synthetic training smoke test passed.")
