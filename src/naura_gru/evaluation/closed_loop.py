"""Closed-loop rolling evaluation without real sensor correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from naura_gru.evaluation.metrics import normalized_accuracy_torch
from naura_gru.models.gru import SensorGRU
from naura_gru.training.dataset import CachedDay
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, train_config


def _manifest_days(cache_root: Path, split: str) -> list[CachedDay]:
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["dates"] if entry["split"] == split]
    if not entries:
        raise ValueError(f"No cache days found for split={split!r}")
    return [CachedDay(entry["cache_dir"]) for entry in entries]


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
    days = _manifest_days(cache_root, split)
    rollout_config = config.get("rollout", {})
    context_len = int(
        rollout_config.get("context_len", train_config.get("train", {}).get("seq_len", 1024))
    )
    rows_per_day = len(days[0])
    horizon_steps = rollout_config.get("horizon_steps")
    if horizon_steps is None:
        horizon_steps = int(float(rollout_config.get("horizon_days", 30)) * rows_per_day)
    horizon_steps = int(horizon_steps)
    binary_indices = train_config.get("target", {}).get(
        "binary_sensor_indices_1based", DEFAULT_BINARY_SENSOR_INDICES_1BASED
    )

    first_day = days[0]
    if len(first_day) <= context_len:
        raise ValueError("First evaluation day is shorter than context_len")
    context = torch.from_numpy(first_day.features[:context_len].copy()).unsqueeze(0).to(device)

    correct = 0
    total = 0
    loss_sum = 0.0
    loss_batches = 0
    actual_steps = 0
    day_index = 0
    row_index = context_len

    with torch.no_grad():
        context_pred, hidden = model(context)
        previous_sensor = sensor_feedback_from_prediction(context_pred[:, -1, :], binary_indices)
        previous_since_real = context[:, -1, SINCE_REAL_OFFSET]

        progress = tqdm(range(horizon_steps), desc="closed-loop", leave=False)
        for _ in progress:
            if day_index >= len(days):
                break
            day = days[day_index]
            if row_index >= len(day):
                day_index += 1
                row_index = 0
                continue

            base = torch.from_numpy(day.features[row_index : row_index + 1].copy()).unsqueeze(0)
            base = base.to(device)
            step_features, previous_since_real = make_rollout_input(
                base, previous_sensor, previous_since_real
            )
            pred, hidden = model(step_features, hidden)
            raw_pred = pred[:, 0, :]
            metric_pred = sensor_feedback_from_prediction(raw_pred, binary_indices)

            target = torch.from_numpy(day.target_slice(row_index, row_index + 1)).unsqueeze(0)
            target = target.to(device)
            mask = torch.from_numpy(day.target_mask[row_index : row_index + 1].copy()).unsqueeze(0)
            mask = mask.to(device=device, dtype=torch.float32)
            weight = torch.from_numpy(day.sample_weight[row_index : row_index + 1].copy()).unsqueeze(0)
            weight = weight.to(device=device, dtype=torch.float32)

            losses = masked_sensor_loss(
                raw_pred.unsqueeze(1),
                target,
                mask,
                weight,
                binary_sensor_indices_1based=binary_indices,
            )
            acc = normalized_accuracy_torch(metric_pred.unsqueeze(1), target, mask)
            correct += int(acc["correct"])
            total += int(acc["total"])
            loss_sum += float(losses["loss"].item())
            loss_batches += 1

            previous_sensor = metric_pred.detach()
            row_index += 1
            actual_steps += 1
            progress.set_postfix(accuracy=float(correct / total) if total else 0.0)

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "cache_dir": str(cache_root),
        "split": split,
        "context_len": context_len,
        "requested_horizon_steps": horizon_steps,
        "actual_steps": actual_steps,
        "loss": float(loss_sum / loss_batches) if loss_batches else 0.0,
        "accuracy": float(correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }
    output_dir = Path(config.get("output_root", "outputs/eval")) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "closed_loop_metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {output_path}", flush=True)
    return metrics
