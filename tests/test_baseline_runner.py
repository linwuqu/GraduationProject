import pandas as pd

from churn.baseline_runner import run_baseline_models
from churn.models.baselines import BaselineModelSpec
from churn.pipeline import ModelingDataset


def test_run_baseline_models_returns_validation_metrics() -> None:
    dataset = ModelingDataset(
        X_train=pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "x2": [1.0, 0.9, 0.2, 0.1]}),
        y_train=pd.Series([0, 0, 1, 1], name="target"),
        X_val=pd.DataFrame({"x1": [0.15, 0.85], "x2": [0.95, 0.15]}),
        y_val=pd.Series([0, 1], name="target"),
        X_test=pd.DataFrame({"x1": [0.05, 0.95], "x2": [0.98, 0.05]}),
        y_test=pd.Series([0, 1], name="target"),
        feature_columns=["x1", "x2"],
        selected_columns=["x1", "x2"],
        preprocessor=None,
        split=None,
        feature_selection_result=None,
    )
    specs = [
        BaselineModelSpec(
            name="tiny_logistic",
            builder=lambda: __import__("sklearn.linear_model").linear_model.LogisticRegression(
                solver="liblinear",
                random_state=42,
            ),
        )
    ]

    results = run_baseline_models(dataset, specs, test_evaluation_enabled=True)

    assert results.shape[0] == 1
    assert results.loc[0, "model_name"] == "tiny_logistic"
    assert results.loc[0, "status"] == "ok"
    assert 0.0 <= results.loc[0, "validation_auc"] <= 1.0
    assert 0.0 <= results.loc[0, "selected_threshold"] <= 1.0
    assert 0.0 <= results.loc[0, "test_auc"] <= 1.0
