import pandas as pd

from churn.preprocessing import ChurnPreprocessor
from churn.schemas import GENDER_CN, MONTHLY_CHARGES_CN, TARGET_CN


def test_preprocessor_fits_on_train_and_transforms_val_without_target_leakage() -> None:
    train = pd.DataFrame(
        {
            GENDER_CN: ["Male", "Female", "Female", "Male"],
            MONTHLY_CHARGES_CN: [10.0, 20.0, 30.0, 40.0],
            TARGET_CN: [0, 1, 0, 1],
        }
    )
    val = pd.DataFrame(
        {
            GENDER_CN: ["Female", "Male"],
            MONTHLY_CHARGES_CN: [15.0, 35.0],
            TARGET_CN: [1, 0],
        }
    )

    preprocessor = ChurnPreprocessor(
        target_column=TARGET_CN,
        numeric_columns=[MONTHLY_CHARGES_CN],
        binary_mappings={GENDER_CN: {"Male": 1, "Female": 0}},
    )
    preprocessor.fit(train)
    transformed = preprocessor.transform(val)

    assert TARGET_CN in transformed.columns
    assert transformed[GENDER_CN].isin([0, 1]).all()
    assert transformed.shape[0] == 2


def test_preprocessor_reuses_training_feature_space_for_unseen_categories() -> None:
    train = pd.DataFrame(
        {
            "plan": ["A", "A", "B", "B"],
            TARGET_CN: [0, 0, 1, 1],
        }
    )
    val = pd.DataFrame(
        {
            "plan": ["A", "C"],
            TARGET_CN: [0, 1],
        }
    )

    preprocessor = ChurnPreprocessor(
        target_column=TARGET_CN,
        numeric_columns=[],
        binary_mappings={},
    )
    preprocessor.fit(train)
    transformed = preprocessor.transform(val)

    assert transformed.columns.tolist() == ["plan_A", "plan_B", TARGET_CN]
    assert transformed.loc[1, "plan_A"] == 0
    assert transformed.loc[1, "plan_B"] == 0
