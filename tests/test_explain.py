import numpy as np

from churn.explain import perturb_numeric_feature, rank_features_by_importance


def test_rank_features_by_importance_returns_sorted_feature_names() -> None:
    values = np.array([[0.2, -0.5, 0.0], [0.3, -0.4, 0.1]])
    names = ["a", "b", "c"]

    ranked = rank_features_by_importance(values, names)

    assert ranked[0][0] == "b"
    assert ranked[-1][0] == "c"


def test_perturb_numeric_feature_changes_only_the_selected_column() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    perturbed = perturb_numeric_feature(matrix, column_index=1, delta=0.5)

    assert np.allclose(perturbed[:, 0], matrix[:, 0])
    assert np.allclose(perturbed[:, 1], np.array([2.5, 4.5]))
