from dataclasses import dataclass

from churn.schemas import TARGET_CN


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_state: int = 42
    target_column: str = TARGET_CN
