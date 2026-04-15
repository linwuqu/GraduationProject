from pathlib import Path

from churn.paths import ProjectPaths


def test_project_paths_resolve_expected_locations(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = ProjectPaths.from_root(project_root)

    assert paths.root == project_root
    assert paths.data_dir == project_root / "data"
    assert paths.artifacts_dir == project_root / "artifacts"
    assert paths.reports_dir == project_root / "reports"
