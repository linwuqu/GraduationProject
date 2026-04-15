from churn.evaluation import compute_binary_metrics, select_best_threshold


def test_compute_binary_metrics_returns_required_keys() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 1, 0, 1],
        y_score=[0.1, 0.8, 0.4, 0.9],
        threshold=0.5,
    )

    assert "auc" in metrics
    assert "f1" in metrics
    assert "recall_positive" in metrics


def test_select_best_threshold_prefers_the_stronger_f1_candidate() -> None:
    threshold, metrics = select_best_threshold(
        y_true=[0, 1, 0, 1],
        y_score=[0.2, 0.55, 0.45, 0.9],
        candidate_thresholds=[0.4, 0.5, 0.6],
    )

    assert threshold == 0.5
    assert metrics["f1"] >= 0.79
