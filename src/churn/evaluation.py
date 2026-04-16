from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(list(y_true))
    y_score_array = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score_array >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true_array, y_score_array)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision_positive": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
    }


def select_best_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    metric_name: str = "f1",
    candidate_thresholds: Iterable[float] | None = None,
) -> tuple[float, dict[str, float]]:
    thresholds = list(candidate_thresholds or np.linspace(0.05, 0.95, 19))
    best_threshold = thresholds[0]
    best_metrics = compute_binary_metrics(y_true, y_score, threshold=best_threshold)

    for threshold in thresholds[1:]:
        metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
        if metrics[metric_name] > best_metrics[metric_name]:
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics


def build_threshold_business_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    customer_value: Iterable[float],
    coupon_cost: Iterable[float],
    uplift_rate: float,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_score_array = np.asarray(list(y_score), dtype=float)
    value_array = np.asarray(list(customer_value), dtype=float)
    cost_array = np.asarray(list(coupon_cost), dtype=float)

    for threshold in list(thresholds or np.linspace(0.05, 0.95, 19)):
        mask = y_score_array >= threshold
        intervened = int(mask.sum())
        if intervened == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "intervened": 0,
                    "expected_saved_customers": 0.0,
                    "expected_saved_value": 0.0,
                    "coupon_cost_total": 0.0,
                    "net_gain": 0.0,
                    "roi": 0.0,
                }
            )
            continue

        churners = y_true_array[mask] == 1
        expected_saved_customers = float(churners.sum() * uplift_rate)
        expected_saved_value = float((value_array[mask] * churners).sum() * uplift_rate)
        coupon_cost_total = float(cost_array[mask].sum())
        net_gain = expected_saved_value - coupon_cost_total
        roi = net_gain / coupon_cost_total if coupon_cost_total > 0 else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "intervened": intervened,
                "expected_saved_customers": expected_saved_customers,
                "expected_saved_value": expected_saved_value,
                "coupon_cost_total": coupon_cost_total,
                "net_gain": float(net_gain),
                "roi": float(roi),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def build_topk_business_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    customer_value: Iterable[float],
    coupon_cost: Iterable[float],
    uplift_rate: float,
    topk_ratios: Iterable[float] | None = None,
) -> pd.DataFrame:
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_score_array = np.asarray(list(y_score), dtype=float)
    value_array = np.asarray(list(customer_value), dtype=float)
    cost_array = np.asarray(list(coupon_cost), dtype=float)
    order = np.argsort(-y_score_array)
    rows: list[dict[str, float]] = []

    for ratio in list(topk_ratios or [0.05, 0.1, 0.2, 0.3, 0.5]):
        top_n = max(1, int(len(order) * ratio))
        picked = order[:top_n]
        churners = y_true_array[picked] == 1
        expected_saved_customers = float(churners.sum() * uplift_rate)
        expected_saved_value = float((value_array[picked] * churners).sum() * uplift_rate)
        coupon_cost_total = float(cost_array[picked].sum())
        net_gain = expected_saved_value - coupon_cost_total
        roi = net_gain / coupon_cost_total if coupon_cost_total > 0 else 0.0
        rows.append(
            {
                "topk_ratio": float(ratio),
                "intervened": int(top_n),
                "expected_saved_customers": expected_saved_customers,
                "expected_saved_value": expected_saved_value,
                "coupon_cost_total": coupon_cost_total,
                "net_gain": float(net_gain),
                "roi": float(roi),
            }
        )

    return pd.DataFrame(rows).sort_values("topk_ratio").reset_index(drop=True)


def summarize_best_business_actions(
    business_curve: pd.DataFrame,
    coupon_tier: str | None = None,
) -> pd.DataFrame:
    if business_curve.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "scenario",
                "policy",
                "coupon_tier",
                "decision_value",
                "intervened",
                "expected_saved_customers",
                "expected_saved_value",
                "coupon_cost_total",
                "net_gain",
                "roi",
            ]
        )

    working = business_curve.copy()
    if coupon_tier is not None:
        working = working[working["coupon_tier"] == coupon_tier]

    rows: list[dict[str, float | str]] = []
    for (model_name, scenario, policy), group in working.groupby(
        ["model_name", "scenario", "policy"],
        dropna=False,
    ):
        best_idx = group["net_gain"].astype(float).idxmax()
        best_row = group.loc[best_idx]
        decision_value = (
            best_row["threshold"]
            if policy == "threshold_all"
            else best_row["topk_ratio"]
        )
        rows.append(
            {
                "model_name": model_name,
                "scenario": scenario,
                "policy": policy,
                "coupon_tier": best_row["coupon_tier"],
                "decision_value": float(decision_value),
                "intervened": int(best_row["intervened"]),
                "expected_saved_customers": float(best_row["expected_saved_customers"]),
                "expected_saved_value": float(best_row["expected_saved_value"]),
                "coupon_cost_total": float(best_row["coupon_cost_total"]),
                "net_gain": float(best_row["net_gain"]),
                "roi": float(best_row["roi"]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["scenario", "model_name", "policy"]
    ).reset_index(drop=True)
