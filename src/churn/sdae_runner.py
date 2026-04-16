from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from churn.evaluation import compute_binary_metrics, select_best_threshold
from churn.models.ensemble import EnsembleConfig, LayerWiseBaggingEnsemble
from churn.models.sdae import SDAEConfig, SDAEFeatureExtractor
from churn.pipeline import ModelingDataset


@dataclass(frozen=True)
class SDAEExperimentConfig:
    name: str
    one_class_label: int | None = None
    n_bags: int = 5
    use_smote: bool = False
    class_weight: str | None = None
    random_state: int = 42


@dataclass
class SDAEExperimentResult:
    name: str
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_scores: np.ndarray
    test_scores: np.ndarray


def run_sdae_experiment(
    dataset: ModelingDataset,
    config: SDAEExperimentConfig,
) -> SDAEExperimentResult:
    train_X = dataset.X_train.copy()
    train_y = dataset.y_train.copy()

    if config.use_smote:
        sampler = SMOTE(random_state=config.random_state)
        sampled_X, sampled_y = sampler.fit_resample(train_X, train_y)
        train_X = pd.DataFrame(sampled_X, columns=dataset.X_train.columns)
        train_y = pd.Series(sampled_y, name=dataset.y_train.name)

    extractor = SDAEFeatureExtractor(
        SDAEConfig(
            random_state=config.random_state,
            one_class_label=config.one_class_label,
        )
    )
    _, train_layer_features = extractor.fit_transform(train_X, target=train_y)
    _, val_layer_features = extractor.transform(dataset.X_val)
    _, test_layer_features = extractor.transform(dataset.X_test)

    ensemble = LayerWiseBaggingEnsemble(
        EnsembleConfig(
            n_bags=config.n_bags,
            random_state=config.random_state,
            class_weight=config.class_weight,
        )
    )
    ensemble.fit(train_layer_features, train_y)

    val_scores = ensemble.predict_proba(val_layer_features)
    threshold, validation_metrics = select_best_threshold(dataset.y_val, val_scores)

    test_scores = ensemble.predict_proba(test_layer_features)
    test_metrics = compute_binary_metrics(dataset.y_test, test_scores, threshold=threshold)

    return SDAEExperimentResult(
        name=config.name,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        val_scores=val_scores,
        test_scores=test_scores,
    )
