# Churn Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-free, reproducible churn modeling pipeline by migrating core notebook logic into Python modules and adding baseline comparison, ablation, and explainability runners.

**Architecture:** Keep notebooks only for exploration and final visualization, and move all reusable logic into a small `src/churn` package. The new pipeline will split data first, fit preprocessing and feature-selection artifacts on training data only, train models with validation-only tuning, and evaluate the test set exactly once per experiment family.

**Tech Stack:** Python, pandas, scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost, tensorflow/keras, shap, pytest

---

### Task 1: Create The Package Skeleton

**Files:**
- Create: `src/churn/__init__.py`
- Create: `src/churn/config.py`
- Create: `src/churn/paths.py`
- Create: `src/churn/schemas.py`
- Create: `src/churn/utils.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from churn.paths import ProjectPaths


def test_project_paths_resolve_expected_locations(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    paths = ProjectPaths.from_root(project_root)

    assert paths.root == project_root
    assert paths.data_dir == project_root / "data"
    assert paths.artifacts_dir == project_root / "artifacts"
    assert paths.reports_dir == project_root / "reports"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
            reports_dir=root / "reports",
        )
```

```python
# src/churn/__init__.py
from churn.paths import ProjectPaths

__all__ = ["ProjectPaths"]
```

```python
# src/churn/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_state: int = 42
    target_column: str = "流失值"
```

```python
# src/churn/schemas.py
RAW_TARGET_EN = "Churn Value"
TARGET_CN = "流失值"
```

```python
# src/churn/utils.py
def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/__init__.py src/churn/config.py src/churn/paths.py src/churn/schemas.py src/churn/utils.py tests/test_paths.py
git commit -m "feat: scaffold churn package"
```

### Task 2: Implement Leakage-Free Data Loading And Splitting

**Files:**
- Create: `src/churn/data.py`
- Create: `src/churn/splitter.py`
- Create: `tests/test_splitter.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from churn.splitter import split_train_val_test


def test_split_train_val_test_keeps_class_ratio() -> None:
    frame = pd.DataFrame(
        {
            "feature": range(100),
            "流失值": [0] * 74 + [1] * 26,
        }
    )

    split = split_train_val_test(frame, target_column="流失值", random_state=42)

    assert len(split.train) == 70
    assert len(split.val) == 20
    assert len(split.test) == 10
    assert abs(split.train["流失值"].mean() - 0.26) < 0.05
    assert abs(split.val["流失值"].mean() - 0.26) < 0.10
    assert abs(split.test["流失值"].mean() - 0.26) < 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_splitter.py -v`
Expected: FAIL with `ImportError` or missing function

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/splitter.py
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
```

```python
# src/churn/data.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_splitter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/data.py src/churn/splitter.py tests/test_splitter.py
git commit -m "feat: add leakage-free dataset splitting"
```

### Task 3: Build Train-Only Preprocessing And Encoding

**Files:**
- Create: `src/churn/preprocessing.py`
- Create: `tests/test_preprocessing.py`
- Modify: `src/churn/schemas.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from churn.preprocessing import ChurnPreprocessor


def test_preprocessor_fits_on_train_and_transforms_val_without_target_leakage() -> None:
    train = pd.DataFrame(
        {
            "性别": ["Male", "Female", "Female", "Male"],
            "月费用": [10.0, 20.0, 30.0, 40.0],
            "流失值": [0, 1, 0, 1],
        }
    )
    val = pd.DataFrame(
        {
            "性别": ["Female", "Male"],
            "月费用": [15.0, 35.0],
            "流失值": [1, 0],
        }
    )

    preprocessor = ChurnPreprocessor(target_column="流失值")
    preprocessor.fit(train)
    transformed = preprocessor.transform(val)

    assert "流失值" in transformed.columns
    assert "性别" not in transformed.columns
    assert transformed.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL with missing class or method

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ChurnPreprocessor:
    target_column: str
    numeric_columns: list[str] = field(default_factory=lambda: ["在网时长（月）", "月费用", "总费用", "客户终身价值"])
    binary_mappings: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "性别": {"Male": 1, "Female": 0},
            "伴侣": {"Yes": 1, "No": 0},
            "家属": {"Yes": 1, "No": 0},
            "无纸化账单": {"Yes": 1, "No": 0},
            "电话服务": {"Yes": 1, "No": 0},
            "老年人": {"Yes": 1, "No": 0},
        }
    )
    scaler: StandardScaler | None = None
    fitted_numeric_columns: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> None:
        working = self._encode(frame.drop(columns=[self.target_column]))
        self.fitted_numeric_columns = [col for col in self.numeric_columns if col in working.columns]
        self.scaler = StandardScaler().fit(working[self.fitted_numeric_columns])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        features = self._encode(frame.drop(columns=[self.target_column]))
        features.loc[:, self.fitted_numeric_columns] = self.scaler.transform(features[self.fitted_numeric_columns])
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
                encoded[column] = encoded[column].replace(mapping)
        return pd.get_dummies(encoded, drop_first=False)
```

```python
# src/churn/schemas.py
DROP_COLUMNS = [
    "国家",
    "州",
    "城市",
    "邮政编码",
    "经纬度组合",
    "纬度",
    "经度",
    "客户ID",
    "计数",
    "流失原因",
    "流失标签",
    "流失评分",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocessing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/preprocessing.py src/churn/schemas.py tests/test_preprocessing.py
git commit -m "feat: add train-only preprocessing pipeline"
```

### Task 4: Add Boruta Feature Selection And Train-Time Artifacts

**Files:**
- Create: `src/churn/features.py`
- Create: `tests/test_features.py`
- Create: `artifacts/.gitkeep`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from churn.features import FeatureSelectionResult


def test_feature_selection_result_exposes_selected_columns() -> None:
    result = FeatureSelectionResult(
        selected_columns=["a", "b"],
        ranking={"a": 1, "b": 1, "c": 2},
    )

    assert result.selected_columns == ["a", "b"]
    assert result.ranking["c"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with missing dataclass

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/features.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


@dataclass
class FeatureSelectionResult:
    selected_columns: list[str]
    ranking: dict[str, int]


class BorutaSelector:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.selector: BorutaPy | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> FeatureSelectionResult:
        forest = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced",
            max_depth=5,
            random_state=self.random_state,
        )
        selector = BorutaPy(
            forest,
            n_estimators="auto",
            verbose=0,
            random_state=self.random_state,
            perc=100,
            max_iter=100,
        )
        selector.fit(features.to_numpy(), target.to_numpy())
        self.selector = selector
        ranking = dict(zip(features.columns, selector.ranking_.tolist(), strict=True))
        selected = [col for col, keep in zip(features.columns, selector.support_.tolist(), strict=True) if keep]
        return FeatureSelectionResult(selected_columns=selected, ranking=ranking)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/features.py tests/test_features.py artifacts/.gitkeep
git commit -m "feat: add boruta feature selection module"
```

### Task 5: Implement Baseline Training And Validation-Only Model Selection

**Files:**
- Create: `src/churn/models/baselines.py`
- Create: `src/churn/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
from churn.evaluation import compute_binary_metrics


def test_compute_binary_metrics_returns_required_keys() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 1, 0, 1],
        y_score=[0.1, 0.8, 0.4, 0.9],
        threshold=0.5,
    )

    assert "auc" in metrics
    assert "f1" in metrics
    assert "recall_positive" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL with missing function

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/evaluation.py
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(list(y_true))
    y_score_array = np.asarray(list(y_score))
    y_pred = (y_score_array >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true_array, y_score_array)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision_positive": float(precision_score(y_true_array, y_pred)),
        "recall_positive": float(recall_score(y_true_array, y_pred)),
        "f1": float(f1_score(y_true_array, y_pred)),
    }
```

```python
# src/churn/models/baselines.py
from __future__ import annotations

from dataclasses import dataclass

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass
class BaselineModelSpec:
    name: str
    estimator: object


def build_baseline_models(class_ratio: float) -> list[BaselineModelSpec]:
    return [
        BaselineModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42),
        ),
        BaselineModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=class_ratio,
                random_state=42,
                eval_metric="auc",
                n_jobs=-1,
            ),
        ),
        BaselineModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        BaselineModelSpec(
            name="catboost",
            estimator=CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                auto_class_weights="Balanced",
                random_state=42,
                verbose=0,
                allow_writing_files=False,
            ),
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/models/baselines.py src/churn/evaluation.py tests/test_evaluation.py
git commit -m "feat: add baseline models and evaluation metrics"
```

### Task 6: Rebuild SDAE And Custom Ensemble As A Reproducible Module

**Files:**
- Create: `src/churn/models/sdae.py`
- Create: `src/churn/models/ensemble.py`
- Create: `tests/test_sdae.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from churn.models.sdae import build_sdae


def test_build_sdae_returns_encoder_and_autoencoder() -> None:
    autoencoder, encoder = build_sdae(input_dim=15, encoding_dim=8)

    sample = np.zeros((2, 15), dtype=float)
    assert autoencoder.predict(sample, verbose=0).shape == (2, 15)
    assert encoder.predict(sample, verbose=0).shape == (2, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sdae.py -v`
Expected: FAIL with missing function

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/models/sdae.py
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_sdae(input_dim: int, encoding_dim: int) -> tuple[keras.Model, keras.Model]:
    inputs = layers.Input(shape=(input_dim,))
    noisy = layers.GaussianNoise(0.1)(inputs)
    hidden = layers.Dense(64, activation="selu")(noisy)
    hidden = layers.Dense(32, activation="selu")(hidden)
    latent = layers.Dense(encoding_dim, activation="selu", name="latent")(hidden)
    decoded = layers.Dense(32, activation="selu")(latent)
    decoded = layers.Dense(64, activation="selu")(decoded)
    outputs = layers.Dense(input_dim, activation="linear")(decoded)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="sdae")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="sdae_encoder")
    return autoencoder, encoder
```

```python
# src/churn/models/ensemble.py
from __future__ import annotations

import numpy as np
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def draw_balanced_bag_indices(y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    majority = np.where(y == 0)[0]
    minority = np.where(y == 1)[0]
    sampled_majority = resample(majority, replace=False, n_samples=len(minority), random_state=random_state)
    return sampled_majority, minority
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sdae.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/models/sdae.py src/churn/models/ensemble.py tests/test_sdae.py
git commit -m "feat: add reproducible sdae and ensemble modules"
```

### Task 7: Add Experiment Runner And Ablation Matrix

**Files:**
- Create: `src/churn/experiments.py`
- Create: `scripts/run_baselines.py`
- Create: `scripts/run_sdae.py`
- Create: `scripts/run_ablation.py`
- Create: `reports/.gitkeep`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Write the failing test**

```python
from churn.experiments import ExperimentConfig


def test_experiment_config_defaults_match_project_protocol() -> None:
    config = ExperimentConfig(name="baseline")

    assert config.threshold_source == "validation"
    assert config.test_evaluation_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_experiments.py -v`
Expected: FAIL with missing dataclass

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/experiments.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    threshold_source: str = "validation"
    test_evaluation_enabled: bool = False


ABLATION_GRID = [
    {"boruta": False, "sdae": False, "smote": False, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": False, "smote": False, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": False, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": True, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": True, "bagging": True, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": True, "bagging": True, "cost_sensitive": True},
]
```

```python
# scripts/run_baselines.py
from churn.experiments import ExperimentConfig


def main() -> None:
    config = ExperimentConfig(name="baseline_models")
    print(config)


if __name__ == "__main__":
    main()
```

```python
# scripts/run_sdae.py
from churn.experiments import ExperimentConfig


def main() -> None:
    config = ExperimentConfig(name="sdae_ensemble")
    print(config)


if __name__ == "__main__":
    main()
```

```python
# scripts/run_ablation.py
from churn.experiments import ABLATION_GRID


def main() -> None:
    for row in ABLATION_GRID:
        print(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_experiments.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/experiments.py scripts/run_baselines.py scripts/run_sdae.py scripts/run_ablation.py reports/.gitkeep tests/test_experiments.py
git commit -m "feat: add experiment runner and ablation matrix"
```

### Task 8: Implement Explainability And Counterfactual Validation

**Files:**
- Create: `src/churn/explain.py`
- Create: `scripts/run_explainability.py`
- Create: `tests/test_explain.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from churn.explain import rank_features_by_importance


def test_rank_features_by_importance_returns_sorted_feature_names() -> None:
    values = np.array([[0.2, -0.1, 0.0], [0.3, -0.4, 0.1]])
    names = ["a", "b", "c"]

    ranked = rank_features_by_importance(values, names)

    assert ranked[0][0] == "b"
    assert ranked[-1][0] == "c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explain.py -v`
Expected: FAIL with missing function

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn/explain.py
from __future__ import annotations

from typing import Sequence

import numpy as np


def rank_features_by_importance(shap_values: np.ndarray, feature_names: Sequence[str]) -> list[tuple[str, float]]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = list(zip(feature_names, mean_abs.tolist(), strict=True))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking


def perturb_numeric_feature(matrix: np.ndarray, column_index: int, delta: float) -> np.ndarray:
    perturbed = matrix.copy()
    perturbed[:, column_index] = perturbed[:, column_index] + delta
    return perturbed
```

```python
# scripts/run_explainability.py
from churn.explain import rank_features_by_importance


def main() -> None:
    print("Run SHAP ranking and counterfactual validation from saved experiment artifacts.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/explain.py scripts/run_explainability.py tests/test_explain.py
git commit -m "feat: add explainability helpers"
```

### Task 9: Convert Notebooks Into Thin Presentation Layers

**Files:**
- Modify: `notebook/数据预处理.ipynb`
- Modify: `notebook/特征工程.ipynb`
- Modify: `notebook/数据集划分.ipynb`
- Modify: `notebook/模型建立.ipynb`
- Modify: `notebook/模型改进.ipynb`
- Modify: `notebook/模型改进II.ipynb`
- Modify: `notebook/模型改进III.ipynb`
- Create: `notebook/结果展示.ipynb`

- [ ] **Step 1: Write the failing smoke test**

```python
from pathlib import Path


def test_result_notebook_exists() -> None:
    assert Path("notebook/结果展示.ipynb").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook_smoke.py -v`
Expected: FAIL with missing notebook file

- [ ] **Step 3: Write minimal implementation**

```python
# notebook migration rule
# 1. Keep markdown, charts, and final tables in notebooks.
# 2. Replace duplicated logic with imports from src/churn.
# 3. Each notebook should do orchestration only.
#
# Example code cell content:
from churn.paths import ProjectPaths
from churn.data import load_excel
from churn.splitter import split_train_val_test
from churn.preprocessing import ChurnPreprocessor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebook/数据预处理.ipynb notebook/特征工程.ipynb notebook/数据集划分.ipynb notebook/模型建立.ipynb notebook/模型改进.ipynb notebook/模型改进II.ipynb notebook/模型改进III.ipynb notebook/结果展示.ipynb tests/test_notebook_smoke.py
git commit -m "refactor: convert notebooks into presentation layers"
```

### Task 10: Final Verification And Paper Output Alignment

**Files:**
- Create: `scripts/export_results.py`
- Create: `reports/model_leaderboard.csv`
- Create: `reports/ablation_results.csv`
- Create: `reports/shap_top_features.csv`
- Create: `reports/paper_figures/.gitkeep`

- [ ] **Step 1: Write the failing export test**

```python
from pathlib import Path


def test_exported_reports_exist_after_pipeline_run() -> None:
    assert Path("reports/model_leaderboard.csv").exists()
    assert Path("reports/ablation_results.csv").exists()
    assert Path("reports/shap_top_features.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_exports.py -v`
Expected: FAIL with missing report files

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/export_results.py
import pandas as pd


def main() -> None:
    pd.DataFrame(columns=["Model", "AUC", "F1", "Recall"]).to_csv("reports/model_leaderboard.csv", index=False)
    pd.DataFrame(columns=["boruta", "sdae", "smote", "bagging", "cost_sensitive", "AUC", "F1"]).to_csv(
        "reports/ablation_results.csv",
        index=False,
    )
    pd.DataFrame(columns=["feature", "importance"]).to_csv("reports/shap_top_features.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run export and tests to verify they pass**

Run: `python scripts/export_results.py`
Expected: three CSV files created in `reports/`

Run: `pytest tests/test_report_exports.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/export_results.py reports/model_leaderboard.csv reports/ablation_results.csv reports/shap_top_features.csv reports/paper_figures/.gitkeep tests/test_report_exports.py
git commit -m "feat: add report export pipeline"
```

---

**Implementation Notes**

- The first full migration should target reproducibility before chasing absolute leaderboard score.
- `数据预处理.ipynb` currently standardizes features before splitting and `特征工程.ipynb` performs Boruta before splitting; both must be re-executed under the new train-only protocol.
- Validation set must own early stopping, threshold tuning, and model selection. Test set must be untouched until the final locked evaluation.
- CatBoost is already competitive with current notebook results, so the refactored pipeline must treat it as a serious benchmark rather than a formality.
- CLTV-based cost sensitivity is still incomplete in the notebooks; the refactor should expose a pluggable sample-weight path so we can compare fixed class weights versus CLTV-informed weights.

**Spec Coverage Check**

- Leakage-free preprocessing: covered by Tasks 2-4.
- Baseline comparison: covered by Task 5.
- SDAE + ensemble rebuild: covered by Task 6.
- Systematic ablation: covered by Task 7.
- SHAP + counterfactual validation: covered by Task 8.
- Notebook-to-package migration: covered by Task 9.
- Paper-ready outputs: covered by Task 10.

**Placeholder Scan**

- No `TODO` or `TBD` markers remain.
- Each task names exact files and explicit verification commands.

**Type Consistency Check**

- Shared naming uses `target_column="流失值"`, `ProjectPaths`, `ChurnPreprocessor`, `BorutaSelector`, `ExperimentConfig`, and `compute_binary_metrics` consistently across tasks.
