from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from churn.baseline_runner import run_baseline_models
    from churn.data import load_excel
    from churn.models.baselines import build_baseline_models
    from churn.paths import ProjectPaths
    from churn.pipeline import build_modeling_dataset
    from churn.schemas import FINAL_MODEL_FEATURE_COLUMNS

    project_paths = ProjectPaths.from_root(PROJECT_ROOT)
    raw_frame = load_excel(project_paths.data_dir / "Telco_customer_churn.xlsx")
    dataset = build_modeling_dataset(raw_frame, feature_columns=FINAL_MODEL_FEATURE_COLUMNS)
    positive_count = int(dataset.y_train.sum())
    negative_count = int(len(dataset.y_train) - positive_count)
    class_ratio = negative_count / max(positive_count, 1)
    results = run_baseline_models(
        dataset,
        build_baseline_models(class_ratio=class_ratio),
        test_evaluation_enabled=True,
    )
    project_paths.reports_dir.mkdir(exist_ok=True)
    output_path = project_paths.reports_dir / "model_leaderboard.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved leaderboard to {output_path}")


if __name__ == "__main__":
    main()
