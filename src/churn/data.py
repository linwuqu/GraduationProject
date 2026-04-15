from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)
