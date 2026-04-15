from __future__ import annotations

from typing import Sequence

import pandas as pd

from churn.evaluation import compute_binary_metrics, select_best_threshold
from churn.models.baselines import BaselineModelSpec
from churn.pipeline import ModelingDataset


def run_baseline_models(
    dataset: ModelingDataset,
    specs: Sequence[BaselineModelSpec],
    test_evaluation_enabled: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in specs:
        try:
            estimator = spec.create_estimator()
        except ModuleNotFoundError:
            rows.append(
                {
                    "model_name": spec.name,
                    "status": "missing_dependency",
                    "selected_threshold": None,
                    "validation_auc": None,
                    "validation_f1": None,
                    "validation_recall_positive": None,
                    "test_auc": None,
                    "test_f1": None,
                    "test_recall_positive": None,
                }
            )
            continue

        estimator.fit(dataset.X_train, dataset.y_train)
        validation_scores = _predict_scores(estimator, dataset.X_val)
        selected_threshold, validation_metrics = select_best_threshold(
            dataset.y_val,
            validation_scores,
        )

        row: dict[str, object] = {
            "model_name": spec.name,
            "status": "ok",
            "selected_threshold": selected_threshold,
            "validation_auc": validation_metrics["auc"],
            "validation_f1": validation_metrics["f1"],
            "validation_recall_positive": validation_metrics["recall_positive"],
            "test_auc": None,
            "test_f1": None,
            "test_recall_positive": None,
        }

        if test_evaluation_enabled:
            test_scores = _predict_scores(estimator, dataset.X_test)
            test_metrics = compute_binary_metrics(
                dataset.y_test,
                test_scores,
                threshold=selected_threshold,
            )
            row["test_auc"] = test_metrics["auc"]
            row["test_f1"] = test_metrics["f1"]
            row["test_recall_positive"] = test_metrics["recall_positive"]

        rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty and "validation_auc" in results.columns:
        status_order = {"ok": 0, "missing_dependency": 1}
        results = results.assign(
            _status_rank=results["status"].map(status_order).fillna(99)
        )
        results = results.sort_values(
            by=["_status_rank", "validation_auc"],
            ascending=[True, False],
            na_position="last",
        ).drop(columns=["_status_rank"]).reset_index(drop=True)
    return results


def _predict_scores(estimator: object, features: pd.DataFrame) -> pd.Series:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)[:, 1]
        return pd.Series(scores, index=features.index, dtype=float)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
        return pd.Series(scores, index=features.index, dtype=float)
    scores = estimator.predict(features)
    return pd.Series(scores, index=features.index, dtype=float)
