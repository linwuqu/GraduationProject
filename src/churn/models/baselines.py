from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class BaselineModelSpec:
    name: str
    builder: Callable[[], object]
    dependency: str | None = None

    def create_estimator(self) -> object:
        return self.builder()


def _build_xgboost(class_ratio: float) -> object:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=class_ratio,
        random_state=42,
        eval_metric="auc",
        n_jobs=-1,
    )


def _build_lightgbm() -> object:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _build_catboost() -> object:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_state=42,
        verbose=0,
        allow_writing_files=False,
    )


def build_baseline_models(class_ratio: float) -> list[BaselineModelSpec]:
    return [
        BaselineModelSpec(
            name="logistic_regression",
            builder=lambda: LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                random_state=42,
            ),
        ),
        BaselineModelSpec(
            name="xgboost",
            builder=lambda: _build_xgboost(class_ratio),
            dependency="xgboost",
        ),
        BaselineModelSpec(
            name="lightgbm",
            builder=_build_lightgbm,
            dependency="lightgbm",
        ),
        BaselineModelSpec(
            name="catboost",
            builder=_build_catboost,
            dependency="catboost",
        ),
    ]
