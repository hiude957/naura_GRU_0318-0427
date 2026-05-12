"""Closed-loop rolling evaluation without real sensor correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from naura_gru.evaluation.metrics import normalized_accuracy_torch, split_sensor_metrics_torch
from naura_gru.models.gru import SensorGRU
from naura_gru.training.dataset import SENSOR_DIM, CachedDay
from naura_gru.training.losses import DEFAULT_BINARY_SENSOR_INDICES_1BASED, masked_sensor_loss
from naura_gru.training.rollout import SINCE_REAL_OFFSET, make_rollout_input, sensor_feedback_from_prediction


def _resolve_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    if run_dir.exists():
        return run_dir.resolve()
    latest_txt = run_dir.parent / "latest.txt"
    if run_dir.name == "latest" and latest_txt.exists():
        return Path(latest_txt.read_text(encoding="utf-8").strip()).resolve()
    raise FileNotFoundError(f"Run directory not found: {run_dir}")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[SensorGRU, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint["config"]
    model_config = train_config.get("model", {})
    model = SensorGRU(
        input_size=int(model_config.get("input_size", 551)),
        hidden_size=int(model_config.get("hidden_size", 512)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
        output_size=int(model_config.get("output_size", 150)),
        head_layer_norm=bool(model_config.get("head_layer_norm", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, train_config


def _manifest_entries(cache_root: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["dates"] if entry["split"] == split]
    if not entries:
        raise ValueError(f"No cache days found for split={split!r}")
    return entries


def _entry_date(entry: dict[str, Any]) -> str:
    return str(entry.get("date") or Path(entry["cache_dir"]).name)


def _selected_day_indices(
    entries: list[dict[str, Any]], date_start: str | None, date_end: str | None
) -> list[int]:
    selected = []
    for index, entry in enumerate(entries):
        date = _entry_date(entry)
        if date_start is not None and date < date_start:
            continue
        if date_end is not None and date > date_end:
            continue
        selected.append(index)
    if not selected:
        raise ValueError(f"No cache days found for date range {date_start!r} to {date_end!r}")
    return selected


def _empty_day_metrics() -> dict[str, float | int]:
    return {
        "correct": 0,
        "total": 0,
        "legacy_correct": 0,
        "legacy_total": 0,
        "continuous_correct": 0,
        "continuous_total": 0,
        "continuous_mae_sum": 0.0,
        "binary_correct": 0,
        "binary_total": 0,
        "binary_mae_sum": 0.0,
        "loss_sum": 0.0,
        "loss_batches": 0,
        "steps": 0,
        "valid_steps": 0,
        "start_row": 0,
        "end_row": 0,
    }


def _finalize_day_metrics(metrics: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    out = {}
    for date, values in metrics.items():
        total = int(values["total"])
        legacy_total = int(values["legacy_total"])
        continuous_total = int(values["continuous_total"])
        binary_total = int(values["binary_total"])
        loss_batches = int(values["loss_batches"])
        out[date] = {
            "steps": int(values["steps"]),
            "valid_steps": int(values["valid_steps"]),
            "start_row": int(values["start_row"]),
            "end_row": int(values["end_row"]),
            "loss": float(values["loss_sum"] / loss_batches) if loss_batches else 0.0,
            "accuracy": float(values["correct"] / total) if total else 0.0,
            "correct": int(values["correct"]),
            "total": total,
            "legacy_accuracy": float(values["legacy_correct"] / legacy_total)
            if legacy_total
            else 0.0,
            "legacy_correct": int(values["legacy_correct"]),
            "legacy_total": legacy_total,
            "continuous_accuracy": float(values["continuous_correct"] / continuous_total)
            if continuous_total
            else 0.0,
            "continuous_correct": int(values["continuous_correct"]),
            "continuous_total": continuous_total,
            "continuous_mae": float(values["continuous_mae_sum"] / continuous_total)
            if continuous_total
            else 0.0,
            "binary_accuracy": float(values["binary_correct"] / binary_total)
            if binary_total
            else 0.0,
            "binary_correct": int(values["binary_correct"]),
            "binary_total": binary_total,
            "binary_mae": float(values["binary_mae_sum"] / binary_total) if binary_total else 0.0,
        }
    return out


def _add_split_metrics(
    totals: dict[str, float | int],
    metrics: dict[str, float | int],
) -> None:
    totals["correct"] = int(totals["correct"]) + int(metrics["split_correct"])
    totals["total"] = int(totals["total"]) + int(metrics["split_total"])
    totals["continuous_correct"] = int(totals["continuous_correct"]) + int(
        metrics["continuous_correct"]
    )
    totals["continuous_total"] = int(totals["continuous_total"]) + int(
        metrics["continuous_total"]
    )
    totals["continuous_mae_sum"] = float(totals["continuous_mae_sum"]) + float(
        metrics["continuous_mae_sum"]
    )
    totals["binary_correct"] = int(totals["binary_correct"]) + int(metrics["binary_correct"])
    totals["binary_total"] = int(totals["binary_total"]) + int(metrics["binary_total"])
    totals["binary_mae_sum"] = float(totals["binary_mae_sum"]) + float(
        metrics["binary_mae_sum"]
    )


def _real_sensor_codes(config: dict[str, Any], train_config: dict[str, Any]) -> list[int]:
    return list(
        config.get(
            "real_sensor_source_codes",
            config.get(
                "metrics",
                {},
            ).get(
                "valid_sources",
                train_config.get("source_codes", {}).get("real_sensor", [1, 2]),
            ),
        )
    )


def _first_real_sensor_row(day: CachedDay, real_sensor_codes: list[int]) -> int:
    source_code = np.load(day.cache_dir / "source_code.npy", mmap_mode="r")
    real_mask = np.isin(source_code, real_sensor_codes)
    rows = np.flatnonzero(real_mask)
    if len(rows) == 0:
        raise ValueError(f"No real sensor rows found in {day.cache_dir}")
    return int(rows[0])


def _context_before_row(
    days: list[CachedDay],
    entries: list[dict[str, Any]],
    day_index: int,
    row_index: int,
    context_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    if row_index >= context_len:
        context = days[day_index].features[row_index - context_len : row_index]
        return torch.from_numpy(context.copy()).unsqueeze(0).to(device), _entry_date(entries[day_index])

    if day_index > 0:
        previous_day = days[day_index - 1]
        previous_needed = context_len - row_index
        if len(previous_day) < previous_needed:
            raise ValueError("Previous warmup day is shorter than required context")
        previous_context = previous_day.features[-previous_needed:]
        current_context = days[day_index].features[:row_index]
        context = np.concatenate([previous_context, current_context], axis=0)
        return torch.from_numpy(context.copy()).unsqueeze(0).to(device), _entry_date(entries[day_index - 1])

    first_day = days[day_index]
    if len(first_day) <= context_len:
        raise ValueError("First evaluation day is shorter than context_len")
    context = first_day.features[:context_len]
    return torch.from_numpy(context.copy()).unsqueeze(0).to(device), _entry_date(entries[day_index])


def _load_day_tensors(day: CachedDay, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.array(day.features, dtype=np.float32, copy=True)).to(device),
        "target": torch.from_numpy(day.target_slice(0, len(day))).to(device),
        "target_mask": torch.from_numpy(
            np.array(day.target_mask, dtype=np.float32, copy=True)
        ).to(device),
        "sample_weight": torch.from_numpy(
            np.array(day.sample_weight, dtype=np.float32, copy=True)
        ).to(device),
    }


def evaluate_closed_loop(config: dict[str, Any]):
    """Run long-horizon closed-loop evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _resolve_run_dir(config.get("run_dir", "runs/latest"))
    checkpoint_path = Path(config.get("checkpoint_path", run_dir / "checkpoints" / "best.pt"))
    model, train_config = _load_model(checkpoint_path, device)

    cache_root = Path(
        config.get("cache_dir")
        or train_config.get("data", {}).get("cache_dir")
        or train_config.get("cache_dir")
    )
    split = str(config.get("split", "val"))
    entries = _manifest_entries(cache_root, split)
    days = [CachedDay(entry["cache_dir"]) for entry in entries]
    rollout_config = config.get("rollout", {})
    context_len = int(
        rollout_config.get("context_len", train_config.get("train", {}).get("seq_len", 1024))
    )
    date_start = config.get("date_start")
    date_end = config.get("date_end")
    selected_indices = _selected_day_indices(entries, date_start, date_end)
    first_eval_day_index = selected_indices[0]
    last_eval_day_index = selected_indices[-1]

    binary_indices = train_config.get("target", {}).get(
        "binary_sensor_indices_1based", DEFAULT_BINARY_SENSOR_INDICES_1BASED
    )

    start_at_first_real_sensor = bool(rollout_config.get("start_at_first_real_sensor", False))
    first_real_sensor_row = None
    warmup_from_previous_day = bool(rollout_config.get("warmup_from_previous_day", True))
    if start_at_first_real_sensor:
        real_sensor_codes = _real_sensor_codes(config, train_config)
        first_real_sensor_row = _first_real_sensor_row(days[first_eval_day_index], real_sensor_codes)
        context_end = first_real_sensor_row + 1
        context, warmup_date = _context_before_row(
            days,
            entries,
            first_eval_day_index,
            context_end,
            context_len,
            device,
        )
        day_index = first_eval_day_index
        row_index = context_end
    elif warmup_from_previous_day and first_eval_day_index > 0:
        warmup_day_index = first_eval_day_index - 1
        warmup_day = days[warmup_day_index]
        if len(warmup_day) < context_len:
            raise ValueError("Warmup day is shorter than context_len")
        context = torch.from_numpy(warmup_day.features[-context_len:].copy()).unsqueeze(0).to(device)
        day_index = first_eval_day_index
        row_index = 0
        warmup_date = _entry_date(entries[warmup_day_index])
    else:
        warmup_day_index = first_eval_day_index
        first_day = days[first_eval_day_index]
        if len(first_day) <= context_len:
            raise ValueError("First evaluation day is shorter than context_len")
        context = torch.from_numpy(first_day.features[:context_len].copy()).unsqueeze(0).to(device)
        day_index = first_eval_day_index
        row_index = context_len
        warmup_date = _entry_date(entries[warmup_day_index])

    rows_per_day = len(days[first_eval_day_index])
    horizon_steps = rollout_config.get("horizon_steps")
    if horizon_steps is None and date_start is None and date_end is None:
        horizon_steps = int(float(rollout_config.get("horizon_days", 30)) * rows_per_day)
    elif horizon_steps is None:
        horizon_steps = len(days[first_eval_day_index]) - row_index
        horizon_steps += sum(len(days[index]) for index in selected_indices[1:])
    horizon_steps = int(horizon_steps)

    totals = _empty_day_metrics()
    loss_sum = 0.0
    loss_batches = 0
    actual_steps = 0
    day_metrics: dict[str, dict[str, float | int]] = {
        _entry_date(entries[index]): _empty_day_metrics() for index in selected_indices
    }
    first_date = _entry_date(entries[first_eval_day_index])
    day_metrics[first_date]["start_row"] = row_index

    with torch.no_grad():
        _context_pred, hidden = model(context)
        previous_sensor = context[:, -1, :SENSOR_DIM]
        previous_since_real = context[:, -1, SINCE_REAL_OFFSET]
        current_loaded_day_index = None
        current_day_tensors: dict[str, torch.Tensor] | None = None

        progress = tqdm(
            range(horizon_steps),
            desc="closed-loop",
            leave=False,
            disable=not bool(config.get("progress", True)),
        )
        for _ in progress:
            if day_index > last_eval_day_index:
                break
            day = days[day_index]
            if row_index >= len(day):
                day_index += 1
                row_index = 0
                if day_index > last_eval_day_index:
                    break
                next_date = _entry_date(entries[day_index])
                day_metrics[next_date]["start_row"] = row_index
                continue

            date = _entry_date(entries[day_index])
            if current_loaded_day_index != day_index:
                current_day_tensors = _load_day_tensors(day, device)
                current_loaded_day_index = day_index
            if current_day_tensors is None:
                raise RuntimeError("Day tensors were not loaded")

            base = current_day_tensors["features"][row_index : row_index + 1].unsqueeze(0)
            step_features, previous_since_real = make_rollout_input(
                base, previous_sensor, previous_since_real
            )
            pred, hidden = model(step_features, hidden)
            raw_pred = pred[:, 0, :]
            metric_pred = sensor_feedback_from_prediction(raw_pred, binary_indices)

            current_day_metrics = day_metrics[date]
            current_day_metrics["steps"] = int(current_day_metrics["steps"]) + 1
            current_day_metrics["end_row"] = row_index
            if bool(day.valid_target[row_index]):
                target = current_day_tensors["target"][row_index : row_index + 1].unsqueeze(0)
                mask = current_day_tensors["target_mask"][row_index : row_index + 1].unsqueeze(0)
                weight = current_day_tensors["sample_weight"][row_index : row_index + 1].unsqueeze(0)

                losses = masked_sensor_loss(
                    raw_pred.unsqueeze(1),
                    target,
                    mask,
                    weight,
                    binary_sensor_indices_1based=binary_indices,
                )
                legacy_acc = normalized_accuracy_torch(metric_pred.unsqueeze(1), target, mask)
                split_acc = split_sensor_metrics_torch(
                    metric_pred.unsqueeze(1),
                    target,
                    mask,
                    binary_sensor_indices_1based=binary_indices,
                )
                totals["legacy_correct"] = int(totals["legacy_correct"]) + int(
                    legacy_acc["correct"]
                )
                totals["legacy_total"] = int(totals["legacy_total"]) + int(legacy_acc["total"])
                _add_split_metrics(totals, split_acc)
                loss_sum += float(losses["loss"].item())
                loss_batches += 1
                current_day_metrics["legacy_correct"] = int(
                    current_day_metrics["legacy_correct"]
                ) + int(legacy_acc["correct"])
                current_day_metrics["legacy_total"] = int(
                    current_day_metrics["legacy_total"]
                ) + int(legacy_acc["total"])
                _add_split_metrics(current_day_metrics, split_acc)
                current_day_metrics["loss_sum"] = float(current_day_metrics["loss_sum"]) + float(
                    losses["loss"].item()
                )
                current_day_metrics["loss_batches"] = int(current_day_metrics["loss_batches"]) + 1
                current_day_metrics["valid_steps"] = int(current_day_metrics["valid_steps"]) + 1

            previous_sensor = metric_pred.detach()
            row_index += 1
            actual_steps += 1
            progress.set_postfix(
                accuracy=float(int(totals["correct"]) / int(totals["total"]))
                if int(totals["total"])
                else 0.0
            )

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "cache_dir": str(cache_root),
        "split": split,
        "date_start": date_start,
        "date_end": date_end,
        "evaluated_dates": [_entry_date(entries[index]) for index in selected_indices],
        "warmup_date": warmup_date,
        "warmup_from_previous_day": warmup_from_previous_day,
        "start_at_first_real_sensor": start_at_first_real_sensor,
        "first_real_sensor_row": first_real_sensor_row,
        "first_rollout_row": day_metrics[first_date]["start_row"],
        "context_len": context_len,
        "requested_horizon_steps": horizon_steps,
        "actual_steps": actual_steps,
        "loss": float(loss_sum / loss_batches) if loss_batches else 0.0,
        "accuracy": float(int(totals["correct"]) / int(totals["total"]))
        if int(totals["total"])
        else 0.0,
        "correct": int(totals["correct"]),
        "total": int(totals["total"]),
        "legacy_accuracy": float(int(totals["legacy_correct"]) / int(totals["legacy_total"]))
        if int(totals["legacy_total"])
        else 0.0,
        "legacy_correct": int(totals["legacy_correct"]),
        "legacy_total": int(totals["legacy_total"]),
        "continuous_accuracy": float(
            int(totals["continuous_correct"]) / int(totals["continuous_total"])
        )
        if int(totals["continuous_total"])
        else 0.0,
        "continuous_correct": int(totals["continuous_correct"]),
        "continuous_total": int(totals["continuous_total"]),
        "continuous_mae": float(totals["continuous_mae_sum"] / int(totals["continuous_total"]))
        if int(totals["continuous_total"])
        else 0.0,
        "binary_accuracy": float(int(totals["binary_correct"]) / int(totals["binary_total"]))
        if int(totals["binary_total"])
        else 0.0,
        "binary_correct": int(totals["binary_correct"]),
        "binary_total": int(totals["binary_total"]),
        "binary_mae": float(totals["binary_mae_sum"] / int(totals["binary_total"]))
        if int(totals["binary_total"])
        else 0.0,
        "per_day": _finalize_day_metrics(day_metrics),
    }
    output_dir = Path(config.get("output_root", "outputs/eval")) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = config.get("output_name")
    if output_name is None:
        if date_start or date_end:
            output_name = f"closed_loop_{date_start or 'start'}_{date_end or 'end'}.json"
        else:
            output_name = "closed_loop_metrics.json"
    output_path = output_dir / str(output_name)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {output_path}", flush=True)
    return metrics
