from pathlib import Path

import pandas as pd

from scripts.export_results import main


def test_exported_reports_exist_after_pipeline_run() -> None:
    main()

    assert Path("reports/model_leaderboard.csv").exists()
    assert Path("reports/ablation_results.csv").exists()
    assert Path("reports/shap_top_features.csv").exists()


def test_export_results_preserves_existing_report_contents(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    existing = pd.DataFrame(
        [{"model_name": "catboost", "validation_auc": 0.86, "test_auc": 0.85}]
    )
    existing.to_csv(reports_dir / "model_leaderboard.csv", index=False)

    main()

    preserved = pd.read_csv(reports_dir / "model_leaderboard.csv")
    assert preserved.to_dict(orient="records") == existing.to_dict(orient="records")