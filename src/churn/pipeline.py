from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from churn.config import SplitConfig
from churn.features import BorutaSelector, FeatureSelectionResult
from churn.preprocessing import ChurnPreprocessor
from churn.schemas import DROP_COLUMNS, NUMERIC_COLUMNS, RAW_TO_CN_RENAME_MAP
from churn.splitter import DataSplit, split_train_val_test


@dataclass
class ModelingDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    selected_columns: list[str]
    preprocessor: ChurnPreprocessor | None
    split: DataSplit | None
    feature_selection_result: FeatureSelectionResult | None


def canonicalize_telco_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    rename_map = {
        source: target
        for source, target in RAW_TO_CN_RENAME_MAP.items()
        if source in working.columns and target not in working.columns
    }
    working = working.rename(columns=rename_map)
    working = working.drop(columns=[column for column in DROP_COLUMNS if column in working.columns])
    working = working.apply(
        lambda column: column.mask(column.astype(str).str.fullmatch(r"\s*"))
    )

    for column in NUMERIC_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            working[column] = working[column].fillna(working[column].median())

    for column in working.columns:
        if working[column].isna().any():
            if working[column].dtype == object:
                mode = working[column].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                working[column] = working[column].fillna(fill_value)
            else:
                working[column] = working[column].fillna(working[column].median())

    return working


def build_modeling_dataset(
    frame: pd.DataFrame,
    split_config: SplitConfig | None = None,
    feature_columns: Sequence[str] | None = None,
    use_boruta: bool = False,
) -> ModelingDataset:
    config = split_config or SplitConfig()
    canonical_frame = canonicalize_telco_frame(frame)

    requested_feature_columns = [
        column for column in (feature_columns or []) if column in canonical_frame.columns
    ]
    if requested_feature_columns:
        selected_feature_columns = requested_feature_columns
    else:
        selected_feature_columns = [
            column for column in canonical_frame.columns if column != config.target_column
        ]

    modeling_frame = canonical_frame.loc[
        :,
        [*selected_feature_columns, config.target_column],
    ].copy()
    split = split_train_val_test(
        modeling_frame,
        target_column=config.target_column,
        random_state=config.random_state,
    )

    preprocessor = ChurnPreprocessor(target_column=config.target_column)
    train_processed = preprocessor.fit_transform(split.train)
    val_processed = preprocessor.transform(split.val)
    test_processed = preprocessor.transform(split.test)

    X_train = train_processed.drop(columns=[config.target_column])
    y_train = train_processed[config.target_column]
    X_val = val_processed.drop(columns=[config.target_column])
    y_val = val_processed[config.target_column]
    X_test = test_processed.drop(columns=[config.target_column])
    y_test = test_processed[config.target_column]

    feature_selection_result = None
    final_columns = X_train.columns.tolist()

    if use_boruta:
        selector = BorutaSelector(random_state=config.random_state)
        feature_selection_result = selector.fit(X_train, y_train)
        final_columns = feature_selection_result.selected_columns
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)

    return ModelingDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_columns=X_train.columns.tolist(),
        selected_columns=final_columns,
        preprocessor=preprocessor,
        split=split,
        feature_selection_result=feature_selection_result,
    )
