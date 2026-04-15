from churn.features import FeatureSelectionResult


def test_feature_selection_result_exposes_selected_columns() -> None:
    result = FeatureSelectionResult(
        selected_columns=["a", "b"],
        ranking={"a": 1, "b": 1, "c": 2},
    )

    assert result.selected_columns == ["a", "b"]
    assert result.ranking["c"] == 2
