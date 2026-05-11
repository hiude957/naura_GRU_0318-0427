"""Fixed 110 ms alignment rules for raw sensor and action data."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from naura_gru.data.defaults import load_action_defaults, load_sensor_defaults
from naura_gru.data.raw_readers import (
    MASK_COLUMNS,
    SENSOR_COLUMNS,
    discover_day_files,
    read_apc_day,
    read_livedata_day,
    read_log_day,
)

GRID_STEP_MS = 110

SOURCE_CODES = {
    "apc_grid": 1,
    "livedata_grid": 2,
    "carried_sensor": 3,
    "sensor_default": 4,
}


def _date_to_datetime(day: str) -> datetime:
    return datetime.strptime(day, "%Y%m%d")


def _datetime_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _timestamp_text_to_ms(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000)


def _iter_dates(start: str, end: str) -> list[str]:
    current = _date_to_datetime(start)
    stop = _date_to_datetime(end)
    dates = []
    while current <= stop:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def build_day_grid(day: str, grid_step_ms: int = GRID_STEP_MS) -> np.ndarray:
    """Build one natural-day 24h grid anchored at 00:00:00.000."""
    start = _datetime_to_ms(_date_to_datetime(day))
    end = _datetime_to_ms(_date_to_datetime(day) + timedelta(days=1))
    return np.arange(start, end, grid_step_ms, dtype=np.int64)


def _valid_interpolation(
    raw_ts: np.ndarray,
    grid_ts: np.ndarray,
    max_gap_ms: int,
    nearest_ms: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.zeros(len(grid_ts), dtype=bool)
    left = np.full(len(grid_ts), -1, dtype=np.int64)
    right = np.full(len(grid_ts), -1, dtype=np.int64)
    ratio = np.zeros(len(grid_ts), dtype=np.float32)
    if len(raw_ts) == 0:
        return valid, left, right, ratio

    pos = np.searchsorted(raw_ts, grid_ts, side="left")
    exact = (pos < len(raw_ts)) & (raw_ts[np.minimum(pos, len(raw_ts) - 1)] == grid_ts)
    valid[exact] = True
    left[exact] = pos[exact]
    right[exact] = pos[exact]

    between = ~exact & (pos > 0) & (pos < len(raw_ts))
    between_idx = np.flatnonzero(between)
    if len(between_idx):
        lidx = pos[between_idx] - 1
        ridx = pos[between_idx]
        gap = raw_ts[ridx] - raw_ts[lidx]
        nearest = np.minimum(grid_ts[between_idx] - raw_ts[lidx], raw_ts[ridx] - grid_ts[between_idx])
        ok = gap <= max_gap_ms
        if nearest_ms is not None:
            ok &= nearest <= nearest_ms
        ok_idx = between_idx[ok]
        valid[ok_idx] = True
        left[ok_idx] = lidx[ok]
        right[ok_idx] = ridx[ok]
        ratio[ok_idx] = (
            (grid_ts[ok_idx] - raw_ts[left[ok_idx]]) / (raw_ts[right[ok_idx]] - raw_ts[left[ok_idx]])
        ).astype(np.float32)
    return valid, left, right, ratio


def _assign_interpolated_source(
    sensor_values: np.ndarray,
    masks: np.ndarray,
    source_code: np.ndarray,
    grid_ts: np.ndarray,
    raw_rows,
    code: int,
    max_gap_ms: int,
    nearest_ms: int | None,
    allowed_rows: np.ndarray,
) -> int:
    if raw_rows.empty:
        return 0
    raw_ts = raw_rows["ts_ms"].to_numpy(dtype=np.int64)
    valid, left, right, ratio = _valid_interpolation(raw_ts, grid_ts, max_gap_ms, nearest_ms)
    valid &= allowed_rows
    valid &= source_code != SOURCE_CODES["apc_grid"]
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return 0

    raw_sensor = raw_rows[SENSOR_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    raw_masks = raw_rows[MASK_COLUMNS].to_numpy(dtype=np.int8, copy=False)
    left_values = raw_sensor[left[idx]]
    right_values = raw_sensor[right[idx]]
    row_ratio = ratio[idx, None]
    sensor_values[idx] = left_values + row_ratio * (right_values - left_values)
    masks[idx] = raw_masks[left[idx]]
    source_code[idx] = code
    return len(idx)


def _apply_causal_sensor_fill(
    sensor_values: np.ndarray,
    masks: np.ndarray,
    source_code: np.ndarray,
    previous_sensor: np.ndarray | None,
    sensor_defaults: np.ndarray,
) -> np.ndarray | None:
    real = (source_code == SOURCE_CODES["apc_grid"]) | (source_code == SOURCE_CODES["livedata_grid"])
    invalid = ~real
    masks[invalid] = 0
    valid_idx = np.flatnonzero(real)

    if len(valid_idx) == 0:
        if previous_sensor is not None:
            sensor_values[:] = previous_sensor
            source_code[:] = SOURCE_CODES["carried_sensor"]
            return previous_sensor.copy()
        sensor_values[:] = sensor_defaults
        source_code[:] = SOURCE_CODES["sensor_default"]
        return None

    first_valid = valid_idx[0]
    if first_valid > 0:
        if previous_sensor is not None:
            sensor_values[:first_valid] = previous_sensor
            source_code[:first_valid] = SOURCE_CODES["carried_sensor"]
        else:
            sensor_values[:first_valid] = sensor_defaults
            source_code[:first_valid] = SOURCE_CODES["sensor_default"]

    last_valid_idx = np.where(real, np.arange(len(real)), -1)
    np.maximum.accumulate(last_valid_idx, out=last_valid_idx)
    fill_rows = invalid & (last_valid_idx >= 0)
    sensor_values[fill_rows] = sensor_values[last_valid_idx[fill_rows]]
    source_code[fill_rows] = SOURCE_CODES["carried_sensor"]
    return sensor_values[valid_idx[-1]].copy()


def align_sensor_to_grid(
    grid: np.ndarray,
    apc_rows,
    livedata_rows,
    config: dict,
    previous_sensor: np.ndarray | None = None,
):
    """Apply apc_grid > livedata_grid > carried_sensor > sensor_default."""
    sensor_defaults = load_sensor_defaults(config.get("sensor_default_path"))
    sensor_values = np.empty((len(grid), len(SENSOR_COLUMNS)), dtype=np.float32)
    sensor_values[:] = sensor_defaults
    masks = np.zeros((len(grid), len(MASK_COLUMNS)), dtype=np.int8)
    source_code = np.full(len(grid), SOURCE_CODES["sensor_default"], dtype=np.int8)

    allowed_rows = np.ones(len(grid), dtype=bool)
    day = datetime.fromtimestamp(int(grid[0]) / 1000).strftime("%Y%m%d")
    if day == str(config.get("first_day", "")):
        allowed_rows &= grid >= _timestamp_text_to_ms(config["first_livedata_ts"])

    apc_assigned = _assign_interpolated_source(
        sensor_values,
        masks,
        source_code,
        grid,
        apc_rows,
        SOURCE_CODES["apc_grid"],
        int(config.get("apc_gap_ms", 250)),
        125,
        allowed_rows,
    )
    livedata_allowed = allowed_rows & (source_code != SOURCE_CODES["apc_grid"])
    livedata_assigned = _assign_interpolated_source(
        sensor_values,
        masks,
        source_code,
        grid,
        livedata_rows,
        SOURCE_CODES["livedata_grid"],
        int(config.get("livedata_gap_ms", 2000)),
        None,
        livedata_allowed,
    )
    final_sensor = _apply_causal_sensor_fill(
        sensor_values, masks, source_code, previous_sensor, sensor_defaults
    )
    return sensor_values, masks, source_code, final_sensor, {
        "apc_grid_rows": int(apc_assigned),
        "livedata_grid_rows": int(livedata_assigned),
    }


def aggregate_actions_to_grid(
    grid: np.ndarray,
    log_rows,
    state_defaults: np.ndarray,
    grid_step_ms: int = GRID_STEP_MS,
):
    """Aggregate actions into grid-level evt_* and state_* columns."""
    row_count = len(grid)
    io_count = len(state_defaults)
    evt = np.zeros((row_count, io_count), dtype=np.int8)
    state = np.empty((row_count, io_count), dtype=np.float32)
    action_count = np.zeros(row_count, dtype=np.int16)

    if log_rows.empty:
        state[:] = state_defaults
        return evt, state, action_count, state_defaults.copy()

    day_start = int(grid[0])
    day_end = day_start + 86_400_000
    actions = log_rows[(log_rows["ts_ms"] >= day_start) & (log_rows["ts_ms"] < day_end)].copy()
    if actions.empty:
        state[:] = state_defaults
        return evt, state, action_count, state_defaults.copy()

    actions["grid_id"] = ((actions["ts_ms"] - day_start) // grid_step_ms).astype("int64")
    actions = actions[(actions["grid_id"] >= 0) & (actions["grid_id"] < row_count)]
    actions = actions[(actions["io_id"] >= 1) & (actions["io_id"] <= io_count)]
    if actions.empty:
        state[:] = state_defaults
        return evt, state, action_count, state_defaults.copy()

    action_count += np.bincount(actions["grid_id"].to_numpy(), minlength=row_count).astype(np.int16)
    actions = actions.sort_values(["ts_ms"], kind="mergesort")
    last_actions = actions.drop_duplicates(["grid_id", "io_id"], keep="last")

    updates_by_io: list[tuple[np.ndarray, np.ndarray]] = []
    final_state = state_defaults.copy()
    for io_idx in range(1, io_count + 1):
        io_actions = last_actions[last_actions["io_id"] == io_idx]
        if io_actions.empty:
            updates_by_io.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            continue
        grouped = io_actions.sort_values("grid_id", kind="mergesort")
        grids = grouped["grid_id"].to_numpy(dtype=np.int64)
        values = grouped["io_value"].to_numpy(dtype=np.float32)
        evt[grids, io_idx - 1] = 1
        updates_by_io.append((grids, values))
        final_state[io_idx - 1] = values[-1]

    for col_idx, (grids, values) in enumerate(updates_by_io):
        col = state[:, col_idx]
        current = state_defaults[col_idx]
        start = 0
        for grid_id, value in zip(grids, values, strict=False):
            col[start:grid_id] = current
            current = value
            start = grid_id
        col[start:] = current

    return evt, state, action_count, final_state


def _split_for_date(date: str, config: dict) -> str:
    boundary = str(config.get("split_boundary", "20260416"))
    return "val" if date >= boundary else "train"


def _write_aligned_parquet(
    output_path: Path,
    grid: np.ndarray,
    sensor_values: np.ndarray,
    masks: np.ndarray,
    source_code: np.ndarray,
    evt: np.ndarray,
    state: np.ndarray,
    action_count: np.ndarray,
) -> None:
    arrays: dict[str, pa.Array] = {
        "timestamp": pa.array(grid, type=pa.timestamp("ms")),
        "ts_ms": pa.array(grid, type=pa.int64()),
        "grid_index": pa.array(np.arange(len(grid), dtype=np.int32)),
        "source_code": pa.array(source_code, type=pa.int8()),
        "has_action": pa.array((action_count > 0).astype(np.int8), type=pa.int8()),
        "action_count": pa.array(action_count, type=pa.int16()),
    }
    for idx, col in enumerate(SENSOR_COLUMNS):
        arrays[col] = pa.array(sensor_values[:, idx], type=pa.float32())
    for idx, col in enumerate(MASK_COLUMNS):
        arrays[col] = pa.array(masks[:, idx], type=pa.int8())
    for idx in range(evt.shape[1]):
        arrays[f"evt_{idx + 1}"] = pa.array(evt[:, idx], type=pa.int8())
    for idx in range(state.shape[1]):
        arrays[f"state_{idx + 1}"] = pa.array(state[:, idx], type=pa.float32())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(arrays)
    pq.write_table(table, output_path, compression="zstd", use_dictionary=False)


def build_aligned_day(
    date: str,
    config: dict,
    previous_state: np.ndarray | None = None,
    previous_sensor: np.ndarray | None = None,
):
    """Build one day's aligned table."""
    input_root = Path(config.get("input_root", "dataset"))
    output_root = Path(config.get("output_root", "outputs/aligned"))
    day_files = discover_day_files(input_root, date)
    grid = build_day_grid(date, int(config.get("grid_step_ms", GRID_STEP_MS)))
    state_defaults = (
        previous_state.copy()
        if previous_state is not None
        else load_action_defaults(config.get("action_default_path", "dataset/action_default.xlsx"))
    )

    log_rows = read_log_day(day_files.day_dir)
    livedata_rows = read_livedata_day(day_files.day_dir)
    apc_rows = read_apc_day(day_files.day_dir)
    sensor_values, masks, source_code, final_sensor, sensor_stats = align_sensor_to_grid(
        grid, apc_rows, livedata_rows, config, previous_sensor
    )
    evt, state, action_count, final_state = aggregate_actions_to_grid(
        grid, log_rows, state_defaults, int(config.get("grid_step_ms", GRID_STEP_MS))
    )

    output_path = output_root / date / f"{date}_aligned.parquet"
    _write_aligned_parquet(output_path, grid, sensor_values, masks, source_code, evt, state, action_count)
    unique_sources, source_counts = np.unique(source_code, return_counts=True)
    source_count_map = {str(int(k)): int(v) for k, v in zip(unique_sources, source_counts, strict=False)}
    info = {
        "date": date,
        "split": _split_for_date(date, config),
        "path": str(output_path),
        "row_count": int(len(grid)),
        "source_counts": source_count_map,
        "sensor_stats": sensor_stats,
        "input_files": {
            "log": str(day_files.log_path) if day_files.log_path else None,
            "livedata": [str(path) for path in day_files.livedata_paths],
            "apc_count": len(day_files.apc_paths),
        },
        "log_rows": int(len(log_rows)),
        "livedata_rows": int(len(livedata_rows)),
        "apc_rows": int(len(apc_rows)),
        "action_rows": int(action_count.sum()),
    }
    return info, final_state, final_sensor


def build_aligned_range(config: dict) -> list[dict]:
    """Build aligned Parquet files for the configured date range and write a manifest."""
    output_root = Path(config.get("output_root", "outputs/aligned"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    previous_state = None
    previous_sensor = None
    for date in _iter_dates(str(config["date_start"]), str(config["date_end"])):
        print(f"[align] building {date}", flush=True)
        info, previous_state, previous_sensor = build_aligned_day(
            date, config, previous_state=previous_state, previous_sensor=previous_sensor
        )
        print(
            f"[align] wrote {info['path']} rows={info['row_count']} split={info['split']}",
            flush=True,
        )
        manifest.append(info)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "date_start": config["date_start"],
                "date_end": config["date_end"],
                "split_boundary": config.get("split_boundary", "20260416"),
                "grid_step_ms": config.get("grid_step_ms", GRID_STEP_MS),
                "source_codes": SOURCE_CODES,
                "dates": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest
