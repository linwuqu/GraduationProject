from __future__ import annotations

from typing import Sequence

import numpy as np


def rank_features_by_importance(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
) -> list[tuple[str, float]]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = list(zip(feature_names, mean_abs.tolist(), strict=True))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking


def perturb_numeric_feature(
    matrix: np.ndarray,
    column_index: int,
    delta: float,
) -> np.ndarray:
    perturbed = matrix.copy()
    perturbed[:, column_index] = perturbed[:, column_index] + delta
    return perturbed
