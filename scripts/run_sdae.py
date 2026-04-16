from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run() -> pd.DataFrame:
    from churn.data import load_excel
    from churn.evaluation import (
        build_threshold_business_curve,
        build_topk_business_curve,
        summarize_best_business_actions,
    )
    from churn.paths import ProjectPaths
    from churn.pipeline import build_modeling_dataset
    from churn.schemas import CLTV_CN, FINAL_MODEL_FEATURE_COLUMNS, MONTHLY_CHARGES_CN
    from churn.sdae_runner import SDAEExperimentConfig, run_sdae_experiment

    project_paths = ProjectPaths.from_root(PROJECT_ROOT)
    raw_frame = load_excel(project_paths.data_dir / "Telco_customer_churn.xlsx")
    dataset = build_modeling_dataset(raw_frame, feature_columns=FINAL_MODEL_FEATURE_COLUMNS)

    configs = [
        SDAEExperimentConfig(name="sdae_standard", one_class_label=None, n_bags=5),
        SDAEExperimentConfig(name="sdae_one_class", one_class_label=0, n_bags=10),
    ]
    rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    for config in configs:
        result = run_sdae_experiment(dataset, config)
        rows.append(
            {
                "model_name": config.name,
                "status": "ok",
                "selected_threshold": result.threshold,
                "validation_auc": result.validation_metrics["auc"],
                "validation_f1": result.validation_metrics["f1"],
                "validation_recall_positive": result.validation_metrics["recall_positive"],
                "test_auc": result.test_metrics["auc"],
                "test_f1": result.test_metrics["f1"],
                "test_recall_positive": result.test_metrics["recall_positive"],
            }
        )
        score_frames.append(
            pd.DataFrame(
                {
                    "model_name": config.name,
                    "y_true": dataset.y_test.to_numpy(),
                    "y_score": result.test_scores,
                    "cltv": dataset.split.test[CLTV_CN].to_numpy(),
                    "monthly_charge": dataset.split.test[MONTHLY_CHARGES_CN].to_numpy(),
                }
            )
        )

    leaderboard = pd.DataFrame(rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
    score_table = pd.concat(score_frames, ignore_index=True)

    project_paths.reports_dir.mkdir(exist_ok=True)
    leaderboard_path = project_paths.reports_dir / "sdae_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    business_rows: list[pd.DataFrame] = []
    for model_name, group in score_table.groupby("model_name"):
        cltv = group["cltv"].astype(float)
        monthly = group["monthly_charge"].astype(float)
        for scenario_name, uplift_rate in [("pessimistic", 0.08), ("neutral", 0.15), ("optimistic", 0.25)]:
            for coupon_ratio, coupon_name in [(0.25, "quarter"), (0.5, "half"), (1.0, "full")]:
                coupon_cost = monthly * coupon_ratio
                threshold_curve = build_threshold_business_curve(
                    y_true=group["y_true"],
                    y_score=group["y_score"],
                    customer_value=cltv,
                    coupon_cost=coupon_cost,
                    uplift_rate=uplift_rate,
                ).assign(
                    model_name=model_name,
                    policy="threshold_all",
                    scenario=scenario_name,
                    coupon_tier=coupon_name,
                )
                topk_curve = build_topk_business_curve(
                    y_true=group["y_true"],
                    y_score=group["y_score"],
                    customer_value=cltv,
                    coupon_cost=coupon_cost,
                    uplift_rate=uplift_rate,
                ).assign(
                    model_name=model_name,
                    policy="topk",
                    scenario=scenario_name,
                    coupon_tier=coupon_name,
                )
                business_rows.extend([threshold_curve, topk_curve])

    business_curve = pd.concat(business_rows, ignore_index=True)
    business_path = project_paths.reports_dir / "business_impact_curve.csv"
    business_curve.to_csv(business_path, index=False)
    half_coupon_summary = summarize_best_business_actions(business_curve, coupon_tier="half")
    summary_path = project_paths.reports_dir / "business_insights_summary.csv"
    half_coupon_summary.to_csv(summary_path, index=False)

    best_overall = (
        half_coupon_summary.sort_values("net_gain", ascending=False).head(1)
        if not half_coupon_summary.empty
        else pd.DataFrame()
    )
    print(leaderboard.to_string(index=False))
    if not half_coupon_summary.empty:
        print("\n=== Half-Coupon (0.5x monthly charge) Best Actions ===")
        print(half_coupon_summary.to_string(index=False))
    if not best_overall.empty:
        print("\n=== Best Overall Business Action (Half-Coupon View) ===")
        print(best_overall.to_string(index=False))
    print(f"\nSaved SDAE leaderboard to {leaderboard_path}")
    print(f"Saved business impact curve to {business_path}")
    print(f"Saved business insights summary to {summary_path}")
    return leaderboard


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
