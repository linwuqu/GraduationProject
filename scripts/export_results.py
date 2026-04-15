from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
LEADERBOARD_COLUMNS = [
    "model_name",
    "status",
    "selected_threshold",
    "validation_auc",
    "validation_f1",
    "validation_recall_positive",
    "test_auc",
    "test_f1",
    "test_recall_positive",
]
ABLATION_COLUMNS = ["boruta", "sdae", "smote", "bagging", "cost_sensitive", "AUC", "F1"]
SHAP_COLUMNS = ["feature", "importance"]


def _ensure_report_exists(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    _ensure_report_exists(REPORTS_DIR / "model_leaderboard.csv", LEADERBOARD_COLUMNS)
    _ensure_report_exists(REPORTS_DIR / "ablation_results.csv", ABLATION_COLUMNS)
    _ensure_report_exists(REPORTS_DIR / "shap_top_features.csv", SHAP_COLUMNS)


if __name__ == "__main__":
    main()