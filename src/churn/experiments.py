from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    threshold_source: str = "validation"
    test_evaluation_enabled: bool = False
    random_state: int = 42


ABLATION_GRID = [
    {"boruta": False, "sdae": False, "smote": False, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": False, "smote": False, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": False, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": True, "bagging": False, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": True, "bagging": True, "cost_sensitive": False},
    {"boruta": True, "sdae": True, "smote": True, "bagging": True, "cost_sensitive": True},
]
