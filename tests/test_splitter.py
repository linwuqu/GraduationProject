import pandas as pd

from churn.schemas import TARGET_CN
from churn.splitter import split_train_val_test


def test_split_train_val_test_keeps_class_ratio() -> None:
    frame = pd.DataFrame(
        {
            "feature": range(100),
            TARGET_CN: [0] * 74 + [1] * 26,
        }
    )

    split = split_train_val_test(frame, target_column=TARGET_CN, random_state=42)

    assert len(split.train) == 70
    assert len(split.val) == 20
    assert len(split.test) == 10
    assert abs(split.train[TARGET_CN].mean() - 0.26) < 0.05
    assert abs(split.val[TARGET_CN].mean() - 0.26) < 0.10
    assert abs(split.test[TARGET_CN].mean() - 0.26) < 0.10
