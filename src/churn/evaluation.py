from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(list(y_true))
    y_score_array = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score_array >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true_array, y_score_array)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision_positive": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
    }


def select_best_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    metric_name: str = "f1",
    candidate_thresholds: Iterable[float] | None = None,
) -> tuple[float, dict[str, float]]:
    thresholds = list(candidate_thresholds or np.linspace(0.05, 0.95, 19))
    best_threshold = thresholds[0]
    best_metrics = compute_binary_metrics(y_true, y_score, threshold=best_threshold)

    for threshold in thresholds[1:]:
        metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
        if metrics[metric_name] > best_metrics[metric_name]:
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics
