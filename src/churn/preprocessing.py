from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import StandardScaler

from churn.schemas import BINARY_MAPPINGS, NUMERIC_COLUMNS


@dataclass
class ChurnPreprocessor:
    target_column: str
    numeric_columns: list[str] = field(default_factory=lambda: NUMERIC_COLUMNS.copy())
    binary_mappings: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            column: mapping.copy() for column, mapping in BINARY_MAPPINGS.items()
        }
    )
    scaler: StandardScaler | None = None
    fitted_numeric_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> None:
        working = self._encode(frame.drop(columns=[self.target_column]))
        self.feature_columns = working.columns.tolist()
        self.fitted_numeric_columns = [
            column for column in self.numeric_columns if column in working.columns
        ]
        if self.fitted_numeric_columns:
            self.scaler = StandardScaler().fit(working[self.fitted_numeric_columns])
        else:
            self.scaler = None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        features = self._encode(frame.drop(columns=[self.target_column]))
        features = features.reindex(columns=self.feature_columns, fill_value=0)
        if self.scaler is not None and self.fitted_numeric_columns:
            features = features.astype(
                {column: float for column in self.fitted_numeric_columns},
                copy=False,
            )
            features.loc[:, self.fitted_numeric_columns] = self.scaler.transform(
                features[self.fitted_numeric_columns]
            )

        result = features.copy()
        result[self.target_column] = frame[self.target_column].to_numpy()
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        for column, mapping in self.binary_mappings.items():
            if column in encoded.columns:
                encoded[column] = encoded[column].map(mapping).fillna(encoded[column])
        return pd.get_dummies(encoded, drop_first=False, dtype=int)
