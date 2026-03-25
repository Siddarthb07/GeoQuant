from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


@dataclass
class TrainResult:
    model: MLPClassifier
    accuracy: float
    auc: float


def train_binary_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 240,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> TrainResult:
    model = MLPClassifier(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=epochs,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_val, preds))
    try:
        auc = float(roc_auc_score(y_val, probs))
    except ValueError:
        auc = 0.5

    return TrainResult(model=model, accuracy=accuracy, auc=auc)


def predict_proba(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]
