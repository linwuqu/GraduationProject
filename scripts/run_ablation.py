from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from churn.data import load_excel
    from churn.experiments import ABLATION_GRID
    from churn.paths import ProjectPaths
    from churn.pipeline import build_modeling_dataset
    from churn.schemas import FINAL_MODEL_FEATURE_COLUMNS
    from churn.sdae_runner import SDAEExperimentConfig, run_sdae_experiment

    project_paths = ProjectPaths.from_root(PROJECT_ROOT)
    raw_frame = load_excel(project_paths.data_dir / "Telco_customer_churn.xlsx")

    rows: list[dict[str, object]] = []
    for idx, config in enumerate(ABLATION_GRID, start=1):
        dataset = build_modeling_dataset(
            raw_frame,
            feature_columns=FINAL_MODEL_FEATURE_COLUMNS,
            use_boruta=config["boruta"],
        )
        if config["sdae"]:
            result = run_sdae_experiment(
                dataset,
                SDAEExperimentConfig(
                    name=f"ablation_{idx}",
                    one_class_label=0 if config["cost_sensitive"] else None,
                    n_bags=5 if config["bagging"] else 1,
                    use_smote=config["smote"],
                    class_weight="balanced" if config["cost_sensitive"] else None,
                    random_state=42 + idx,
                ),
            )
            auc = result.test_metrics["auc"]
            f1 = result.test_metrics["f1"]
            recall = result.test_metrics["recall_positive"]
        else:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
            model.fit(dataset.X_train, dataset.y_train)
            scores = model.predict_proba(dataset.X_test)[:, 1]
            from churn.evaluation import compute_binary_metrics

            metrics = compute_binary_metrics(dataset.y_test, scores)
            auc = metrics["auc"]
            f1 = metrics["f1"]
            recall = metrics["recall_positive"]

        rows.append({**config, "AUC": auc, "F1": f1, "Recall": recall})

    result_frame = pd.DataFrame(rows)
    project_paths.reports_dir.mkdir(exist_ok=True)
    output_path = project_paths.reports_dir / "ablation_results.csv"
    result_frame.to_csv(output_path, index=False)
    print(result_frame.to_string(index=False))
    print(f"\nSaved ablation results to {output_path}")


if __name__ == "__main__":
    main()
