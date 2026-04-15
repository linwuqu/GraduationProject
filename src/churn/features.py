from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_columns: list[str]
    ranking: dict[str, int]


class BorutaSelector:
    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 100,
        perc: int = 100,
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.perc = perc
        self.selected_columns: list[str] = []
        self.ranking: dict[str, int] = {}
        self._selector = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> FeatureSelectionResult:
        from boruta import BorutaPy

        forest = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced",
            max_depth=5,
            random_state=self.random_state,
        )
        selector = BorutaPy(
            estimator=forest,
            n_estimators="auto",
            verbose=0,
            random_state=self.random_state,
            perc=self.perc,
            max_iter=self.max_iter,
        )
        selector.fit(features.to_numpy(), target.to_numpy())
        self._selector = selector
        self.ranking = dict(zip(features.columns, selector.ranking_.tolist(), strict=True))
        self.selected_columns = [
            column
            for column, keep in zip(features.columns, selector.support_.tolist(), strict=True)
            if keep
        ]
        return FeatureSelectionResult(
            selected_columns=self.selected_columns.copy(),
            ranking=self.ranking.copy(),
        )

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            raise RuntimeError("BorutaSelector must be fitted before transform.")
        return features.loc[:, self.selected_columns].copy()

    def fit_transform(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, FeatureSelectionResult]:
        result = self.fit(features, target)
        return self.transform(features), result
