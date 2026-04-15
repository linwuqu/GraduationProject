from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
            reports_dir=root / "reports",
        )
