"""Default state and sensor initialization policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DefaultPolicy:
    """Default-value paths and first-day initialization metadata."""

    action_default_path: Path
    sensor_default_path: Path | None
    first_day: str = "20260318"
    first_livedata_ts: str = "2026-03-18 11:14:23.321"


def load_action_defaults(_path: str | Path):
    """Load action_default.xlsx into initial state_1..state_122 values."""
    path = Path(_path)
    defaults = np.zeros(122, dtype=np.float32)
    if not path.exists():
        return defaults

    df = pd.read_excel(path)
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    index_col = normalized_cols.get("index", df.columns[0])
    default_col = normalized_cols.get("default", df.columns[1] if len(df.columns) > 1 else df.columns[0])
    for _, row in df.iterrows():
        if pd.isna(row[index_col]):
            continue
        idx = int(row[index_col])
        if 1 <= idx <= len(defaults):
            defaults[idx - 1] = 0.0 if pd.isna(row[default_col]) else float(row[default_col])
    return defaults


def load_sensor_defaults(_path: str | Path | None):
    """Load sensor defaults, or return the unified default if no table exists."""
    defaults = np.zeros(150, dtype=np.float32)
    if _path is None:
        return defaults
    path = Path(_path)
    if not path.exists():
        return defaults

    df = pd.read_excel(path)
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    index_col = normalized_cols.get("index")
    default_col = normalized_cols.get("default")
    if index_col is None or default_col is None:
        return defaults
    for _, row in df.iterrows():
        if pd.isna(row[index_col]):
            continue
        idx = int(row[index_col])
        if 1 <= idx <= len(defaults):
            defaults[idx - 1] = 0.0 if pd.isna(row[default_col]) else float(row[default_col])
    return defaults


def is_first_day_initialization_window(ts: str, policy: DefaultPolicy) -> bool:
    """Return whether a timestamp belongs to the 20260318 pre-livedata window."""
    return ts[:10].replace("-", "") == policy.first_day and ts < policy.first_livedata_ts
