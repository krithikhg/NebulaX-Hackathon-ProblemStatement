"""Evaluation metrics (generic across PS3 subsystems)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def summary(y_true: np.ndarray, y_pred: np.ndarray, names=("Normal", "Side I", "Side II")) -> Dict:
    cm = confusion(y_true, y_pred)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_f1": dict(zip(names, np.round(per_class_f1(y_true, y_pred), 4).tolist())),
        "confusion": cm.tolist(),
    }
