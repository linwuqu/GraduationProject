from sklearn.linear_model import LogisticRegression

from churn.experiments import ABLATION_GRID, ExperimentConfig
from churn.models.baselines import build_baseline_models


def test_experiment_config_defaults_match_project_protocol() -> None:
    config = ExperimentConfig(name="baseline")

    assert config.threshold_source == "validation"
    assert config.test_evaluation_enabled is False


def test_build_baseline_models_exposes_logistic_regression_without_optional_imports() -> None:
    specs = build_baseline_models(class_ratio=3.0)

    assert [spec.name for spec in specs] == [
        "logistic_regression",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
    assert isinstance(specs[0].create_estimator(), LogisticRegression)
    assert ABLATION_GRID[-1]["cost_sensitive"] is True
