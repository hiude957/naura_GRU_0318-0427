"""Readers for raw log, APC, and livedata files.

This module owns source-file parsing only. It should not decide alignment,
model features, or train/eval policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [str(i) for i in range(1, 151)]
MASK_COLUMNS = [f"mask_{i}" for i in range(1, 151)]


@dataclass(frozen=True)
class DayFiles:
    """Raw files discovered for one YYYYMMDD dataset directory."""

    date: str
    day_dir: Path
    log_path: Path | None
    livedata_paths: tuple[Path, ...]
    apc_paths: tuple[Path, ...]


def discover_date_dirs(input_root: str | Path) -> list[Path]:
    """Return sorted YYYYMMDD date directories under the dataset root."""
    root = Path(input_root)
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def discover_day_files(input_root: str | Path, date: str) -> DayFiles:
    """Discover raw files for one date, tolerating empty/missing subdirectories."""
    day_dir = Path(input_root) / date
    log_dir = day_dir / "log"
    sensor_dir = day_dir / "sensor"
    log_paths = sorted(log_dir.glob("*_log_anonymized.txt")) if log_dir.exists() else []
    livedata_paths = sorted(sensor_dir.glob("livedata*.txt")) if sensor_dir.exists() else []
    apc_paths = sorted(sensor_dir.glob("apc_*_anonymized.txt")) if sensor_dir.exists() else []
    return DayFiles(
        date=date,
        day_dir=day_dir,
        log_path=log_paths[0] if log_paths else None,
        livedata_paths=tuple(livedata_paths),
        apc_paths=tuple(apc_paths),
    )


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": pd.Series(dtype="int64"),
            "io_id": pd.Series(dtype="int16"),
            "io_value": pd.Series(dtype="float32"),
        }
    )


def _empty_sensor() -> pd.DataFrame:
    columns = {
        "ts_ms": pd.Series(dtype="int64"),
        **{col: pd.Series(dtype="float32") for col in SENSOR_COLUMNS},
        **{col: pd.Series(dtype="int8") for col in MASK_COLUMNS},
    }
    return pd.DataFrame(columns)


def _datetime_to_ms(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype="datetime64[ms]").astype("int64")


def _parse_log_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y/%m/%d %H:%M:%S.%f")


def _parse_livedata_time(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.replace(r":(\d{1,6})$", r".\1", regex=True)
    return pd.to_datetime(normalized, format="%Y/%m/%d %H:%M:%S.%f")


def _parse_process_start(value: str) -> pd.Timestamp:
    date_part, time_part = value.strip().split(maxsplit=1)
    if time_part.count(":") >= 3:
        head, frac = time_part.rsplit(":", 1)
        normalized = f"{date_part} {head}.{frac}"
    else:
        normalized = f"{date_part} {time_part}"
    fmt = "%Y/%m/%d %H:%M:%S.%f" if "." in normalized else "%Y/%m/%d %H:%M:%S"
    return pd.Timestamp(pd.to_datetime(normalized, format=fmt))


def _ensure_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in SENSOR_COLUMNS:
        if col not in df:
            df[col] = 0.0
    for col in MASK_COLUMNS:
        if col not in df:
            df[col] = 0
    df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].astype("float32")
    df[MASK_COLUMNS] = df[MASK_COLUMNS].fillna(0).astype("int8")
    return df[["ts_ms", *SENSOR_COLUMNS, *MASK_COLUMNS]]


def read_log_day(path_or_day_dir: str | Path) -> pd.DataFrame:
    """Read one day's log/action file.

    Output columns: ts_ms, io_id, io_value.
    """
    path = Path(path_or_day_dir)
    if path.is_dir():
        matches = sorted((path / "log").glob("*_log_anonymized.txt"))
        if not matches:
            return _empty_log()
        path = matches[0]
    if not path.exists():
        return _empty_log()

    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return _empty_log()
    df["ts_ms"] = _datetime_to_ms(_parse_log_time(df["timestamp"]))
    out = df[["ts_ms", "io_id", "io_value"]].copy()
    out["io_id"] = out["io_id"].astype("int16")
    out["io_value"] = out["io_value"].astype("float32")
    return out.sort_values("ts_ms", kind="mergesort").reset_index(drop=True)


def read_livedata_day(path_or_day_dir: str | Path) -> pd.DataFrame:
    """Read one day's livedata file with absolute timestamps."""
    path = Path(path_or_day_dir)
    if path.is_dir():
        paths = sorted((path / "sensor").glob("livedata*.txt"))
    else:
        paths = [path]
    frames: list[pd.DataFrame] = []
    for livedata_path in paths:
        if not livedata_path.exists():
            continue
        df = pd.read_csv(livedata_path, sep="\t")
        if df.empty:
            continue
        ts_ms = _datetime_to_ms(_parse_livedata_time(df["Time"]))
        df = df.drop(columns=["Time"], errors="ignore").copy()
        df.insert(0, "ts_ms", ts_ms)
        frames.append(_ensure_sensor_columns(df))
    if not frames:
        return _empty_sensor()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("ts_ms", kind="mergesort").drop_duplicates("ts_ms", keep="last")
    return out.reset_index(drop=True)


def read_apc_file(path: str | Path) -> pd.DataFrame:
    """Read one APC file and convert Process Start Time + Time to absolute time."""
    path = Path(path)
    if not path.exists():
        return _empty_sensor()

    process_start = None
    header_line = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            line = line.rstrip("\n")
            if line.startswith("Process Start Time:"):
                value = line.split(":", 1)[1].strip()
                process_start = _parse_process_start(value)
            if line.startswith("Time\t"):
                header_line = line_no
                break
    if process_start is None or header_line is None:
        return _empty_sensor()

    df = pd.read_csv(path, sep="\t", skiprows=header_line)
    if df.empty:
        return _empty_sensor()
    start_ms = process_start.value // 1_000_000
    ts_ms = start_ms + np.rint(df["Time"].astype("float64").to_numpy() * 1000).astype("int64")
    df = df.drop(columns=["Time"], errors="ignore").copy()
    df.insert(0, "ts_ms", ts_ms)
    return _ensure_sensor_columns(df).sort_values("ts_ms", kind="mergesort").reset_index(drop=True)


def read_apc_day(path_or_day_dir: str | Path) -> pd.DataFrame:
    """Read all APC files for one day and return one sorted sensor frame."""
    path = Path(path_or_day_dir)
    if path.is_dir():
        paths = sorted((path / "sensor").glob("apc_*_anonymized.txt"))
    else:
        paths = [path]
    frames = [read_apc_file(apc_path) for apc_path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_sensor()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("ts_ms", kind="mergesort").drop_duplicates("ts_ms", keep="last")
    return out.reset_index(drop=True)
