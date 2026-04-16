from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from churn.data import load_excel
    from churn.evaluation import compute_binary_metrics
    from churn.paths import ProjectPaths
    from churn.pipeline import build_modeling_dataset
    from churn.schemas import FINAL_MODEL_FEATURE_COLUMNS

    project_paths = ProjectPaths.from_root(PROJECT_ROOT)
    raw_frame = load_excel(project_paths.data_dir / "Telco_customer_churn.xlsx")
    dataset = build_modeling_dataset(raw_frame, feature_columns=FINAL_MODEL_FEATURE_COLUMNS)

    model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
    model.fit(dataset.X_train, dataset.y_train)
    test_scores = model.predict_proba(dataset.X_test)[:, 1]
    baseline_metrics = compute_binary_metrics(dataset.y_test, test_scores)

    explainer = shap.Explainer(model, dataset.X_train)
    shap_values = explainer(dataset.X_test)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranking = (
        pd.DataFrame(
            {
                "feature": dataset.X_test.columns,
                "importance": mean_abs,
                "direction_sign": np.sign(shap_values.values.mean(axis=0)),
                "model_name": "logistic_regression",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_features_path = project_paths.reports_dir / "shap_top_features.csv"
    ranking.to_csv(top_features_path, index=False)

    threshold = 0.6
    perturbed = dataset.X_test.copy()
    top_feature = ranking.loc[0, "feature"]
    perturbed[top_feature] = perturbed[top_feature] + perturbed[top_feature].std()
    perturbed_scores = model.predict_proba(perturbed)[:, 1]
    flipped = ((test_scores < threshold) & (perturbed_scores >= threshold)).sum()
    one_class_validation = pd.DataFrame(
        [
            {
                "model_variant": "logistic_regression",
                "threshold": threshold,
                "flip_count_after_top_feature_shift": int(flipped),
                "original_auc": baseline_metrics["auc"],
                "notes": f"top_feature={top_feature}",
            }
        ]
    )
    validation_path = project_paths.reports_dir / "one_class_validation.csv"
    one_class_validation.to_csv(validation_path, index=False)
    print(ranking.head(10).to_string(index=False))
    print(f"\nSaved SHAP ranking to {top_features_path}")
    print(f"Saved perturbation validation to {validation_path}")


if __name__ == "__main__":
    main()
