from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_train_val_test(
    frame: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> DataSplit:
    train_val, test = train_test_split(
        frame,
        test_size=0.1,
        random_state=random_state,
        stratify=frame[target_column],
    )
    train, val = train_test_split(
        train_val,
        test_size=2 / 9,
        random_state=random_state,
        stratify=train_val[target_column],
    )
    return DataSplit(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
