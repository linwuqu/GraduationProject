import pandas as pd

from churn.config import SplitConfig
from churn.pipeline import build_modeling_dataset, canonicalize_telco_frame
from churn.schemas import COUNTRY_CN, CUSTOMER_ID_CN, GENDER_CN, TARGET_CN, TENURE_MONTHS_CN


def _build_raw_telco_frame(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": [f"000{i}" for i in range(rows)],
            "Country": ["USA"] * rows,
            "Gender": ["Male" if i % 2 == 0 else "Female" for i in range(rows)],
            "Tenure Months": list(range(1, rows + 1)),
            "Monthly Charges": [50.0 + i for i in range(rows)],
            "Total Charges": ["" if i == 0 else 100.0 + i * 10 for i in range(rows)],
            "Contract": ["Month-to-month" if i % 3 == 0 else "One year" for i in range(rows)],
            "Churn Value": [0, 1] * (rows // 2),
            "CLTV": [2000 + i * 100 for i in range(rows)],
        }
    )


def test_canonicalize_telco_frame_renames_raw_columns_and_drops_metadata() -> None:
    frame = canonicalize_telco_frame(_build_raw_telco_frame())

    assert TARGET_CN in frame.columns
    assert GENDER_CN in frame.columns
    assert COUNTRY_CN not in frame.columns
    assert CUSTOMER_ID_CN not in frame.columns
    assert frame[TENURE_MONTHS_CN].dtype.kind in {"i", "u", "f"}
    assert frame.isna().sum().sum() == 0


def test_build_modeling_dataset_returns_leakage_free_splits() -> None:
    dataset = build_modeling_dataset(
        _build_raw_telco_frame(),
        split_config=SplitConfig(random_state=7),
    )

    assert len(dataset.X_train) == 14
    assert len(dataset.X_val) == 4
    assert len(dataset.X_test) == 2
    assert TARGET_CN not in dataset.X_train.columns
    assert dataset.X_train.columns.tolist() == dataset.X_val.columns.tolist()
    assert dataset.X_train.columns.tolist() == dataset.X_test.columns.tolist()
    assert dataset.feature_columns == dataset.X_train.columns.tolist()
