from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


@dataclass
class EnsembleConfig:
    n_bags: int = 5
    random_state: int = 42
    class_weight: str | None = None


@dataclass
class LayerModel:
    bag_index: int
    layer_index: int
    model: object


class LayerWiseBaggingEnsemble:
    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.members: list[LayerModel] = []

    def fit(
        self,
        layer_features: list[pd.DataFrame],
        target: pd.Series,
        base_estimator: object | None = None,
    ) -> None:
        estimator_template = base_estimator or LogisticRegression(
            solver="liblinear",
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )
        self.members = []
        y = target.to_numpy()
        positive_idx = np.where(y == 1)[0]
        negative_idx = np.where(y == 0)[0]

        if len(positive_idx) == 0 or len(negative_idx) == 0:
            raise ValueError("Both classes must be present to train the ensemble.")

        for bag_idx in range(self.config.n_bags):
            sampled_negative = resample(
                negative_idx,
                replace=False,
                n_samples=min(len(negative_idx), len(positive_idx)),
                random_state=self.config.random_state + bag_idx,
            )
            bag_indices = np.concatenate([positive_idx, sampled_negative])
            for layer_idx, features in enumerate(layer_features):
                estimator = clone(estimator_template)
                estimator.fit(features.iloc[bag_indices], target.iloc[bag_indices])
                self.members.append(
                    LayerModel(
                        bag_index=bag_idx,
                        layer_index=layer_idx,
                        model=estimator,
                    )
                )

    def predict_proba(self, layer_features: list[pd.DataFrame]) -> np.ndarray:
        if not self.members:
            raise RuntimeError("Ensemble must be fitted before predict_proba.")
        probs: list[np.ndarray] = []
        for member in self.members:
            layer_matrix = layer_features[member.layer_index]
            scores = member.model.predict_proba(layer_matrix)[:, 1]
            probs.append(scores)
        return np.mean(np.vstack(probs), axis=0)
