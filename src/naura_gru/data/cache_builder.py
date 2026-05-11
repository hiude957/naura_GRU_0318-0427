"""Build model cache arrays from aligned fixed-grid tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from naura_gru.data.raw_readers import MASK_COLUMNS, SENSOR_COLUMNS

EVT_COLUMNS = [f"evt_{i}" for i in range(1, 123)]
STATE_COLUMNS = [f"state_{i}" for i in range(1, 123)]


def _stack_columns(table, columns: list[str], dtype) -> np.ndarray:
    return np.column_stack(
        [table[column].to_numpy(zero_copy_only=False) for column in columns]
    ).astype(dtype, copy=False)


def _since_event(ts_ms: np.ndarray, event_mask: np.ndarray, cap_ms: int) -> np.ndarray:
    event_ts = np.where(event_mask, ts_ms, -1)
    np.maximum.accumulate(event_ts, out=event_ts)
    since = np.where(event_ts >= 0, ts_ms - event_ts, cap_ms)
    return np.clip(since, 0, cap_ms).astype(np.float32) / float(cap_ms)


def build_cache_for_day(aligned_path, cache_dir, config):
    """Convert one aligned day into feature, target, mask, and metadata arrays."""
    aligned_path = Path(aligned_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(aligned_path)

    sensors = _stack_columns(table, SENSOR_COLUMNS, np.float32)
    masks = _stack_columns(table, MASK_COLUMNS, np.int8)
    evt = _stack_columns(table, EVT_COLUMNS, np.int8)
    state = _stack_columns(table, STATE_COLUMNS, np.float32)
    ts_ms = table["ts_ms"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    source_code = table["source_code"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    has_action = table["has_action"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)

    source_onehot = np.zeros((len(ts_ms), 4), dtype=np.float32)
    valid_source = (source_code >= 1) & (source_code <= 4)
    source_onehot[np.flatnonzero(valid_source), source_code[valid_source] - 1] = 1.0

    cap_ms = int(config.get("time_feature_cap_ms", 86_400_000))
    delta_t = np.diff(ts_ms, prepend=ts_ms[0]).astype(np.float32)
    delta_t = np.clip(delta_t, 0, cap_ms) / float(cap_ms)
    real_codes = set(config.get("source_codes", {}).get("real_sensor", [1, 2]))
    real_sensor = np.isin(source_code, list(real_codes))
    time_features = np.column_stack(
        [
            delta_t,
            _since_event(ts_ms, has_action != 0, cap_ms),
            _since_event(ts_ms, real_sensor, cap_ms),
        ]
    ).astype(np.float32)

    features = np.concatenate(
        [
            sensors,
            masks.astype(np.float32),
            evt.astype(np.float32),
            state,
            source_onehot,
            time_features,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    real_indices = np.flatnonzero(real_sensor)
    row_indices = np.arange(len(ts_ms))
    next_pos = np.searchsorted(real_indices, row_indices + 1)
    has_future = next_pos < len(real_indices)
    future_idx = np.full(len(ts_ms), -1, dtype=np.int64)
    future_idx[has_future] = real_indices[next_pos[has_future]]
    future_gap = np.zeros(len(ts_ms), dtype=np.int64)
    future_gap[has_future] = ts_ms[future_idx[has_future]] - ts_ms[has_future]
    within_gap = has_future & (future_gap <= int(config.get("max_target_dt_ms", 600_000)))

    target_delta = np.zeros_like(sensors, dtype=np.float32)
    target_source = np.zeros(len(ts_ms), dtype=np.int8)
    target_log_dt = np.zeros(len(ts_ms), dtype=np.float32)
    loss_mask = np.zeros_like(masks, dtype=np.int8)
    idx = np.flatnonzero(within_gap)
    if len(idx):
        fut = future_idx[idx]
        target_delta[idx] = sensors[fut] - sensors[idx]
        target_source[idx] = source_code[fut]
        target_log_dt[idx] = np.log1p(future_gap[idx]).astype(np.float32)
        loss_mask[idx] = (masks[idx].astype(bool) & masks[fut].astype(bool)).astype(np.int8)
    valid_target = loss_mask.any(axis=1)
    sample_weight = valid_target.astype(np.float32)

    np.save(cache_dir / "features.npy", features)
    np.save(cache_dir / "target_delta.npy", target_delta)
    np.save(cache_dir / "target_log_dt.npy", target_log_dt)
    np.save(cache_dir / "loss_mask.npy", loss_mask)
    np.save(cache_dir / "sample_weight.npy", sample_weight)
    np.save(cache_dir / "valid_target.npy", valid_target)
    np.save(cache_dir / "ts_ms.npy", ts_ms)
    np.save(cache_dir / "source_code.npy", source_code)
    np.save(cache_dir / "target_source.npy", target_source)
    np.save(cache_dir / "has_action.npy", has_action)

    meta = {
        "aligned_path": str(aligned_path),
        "row_count": int(len(ts_ms)),
        "feature_dim": int(features.shape[1]),
        "valid_target_count": int(valid_target.sum()),
        "real_sensor_count": int(real_sensor.sum()),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def build_cache(config):
    """Build cache arrays and manifest for a configured date range."""
    aligned_root = Path(config.get("aligned_root", "outputs/aligned"))
    cache_root = Path(config.get("cache_root", "outputs/cache")) / config.get(
        "cache_name", "fixed_110ms_v1"
    )
    manifest_path = aligned_root / "manifest.json"
    aligned_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for day_info in aligned_manifest["dates"]:
        date = day_info["date"]
        day_cache_dir = cache_root / date
        print(f"[cache] building {date}", flush=True)
        meta = build_cache_for_day(day_info["path"], day_cache_dir, config)
        print(
            f"[cache] wrote {day_cache_dir} rows={meta['row_count']} split={day_info['split']}",
            flush=True,
        )
        entries.append(
            {
                "date": date,
                "split": day_info["split"],
                "cache_dir": str(day_cache_dir),
                **meta,
            }
        )

    manifest = {
        "cache_name": config.get("cache_name", "fixed_110ms_v1"),
        "aligned_manifest": str(manifest_path),
        "dates": entries,
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
