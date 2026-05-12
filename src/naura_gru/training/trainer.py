"""Training loop, checkpointing, and metrics emission."""

from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from tqdm import tqdm

from naura_gru.evaluation.metrics import normalized_accuracy_torch
from naura_gru.models.gru import SensorGRU
from naura_gru.training.dataset import make_dataloaders
from naura_gru.training.losses import (
    DEFAULT_BINARY_SENSOR_INDICES_1BASED,
    masked_sensor_loss,
)
from naura_gru.training.rollout import SINCE_REAL_OFFSET, make_rollout_input, sensor_feedback_from_prediction
from naura_gru.utils.seed import seed_everything


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _binary_indices(config: dict[str, Any]) -> list[int]:
    return list(
        config.get("target", {}).get(
            "binary_sensor_indices_1based", DEFAULT_BINARY_SENSOR_INDICES_1BASED
        )
    )


def _build_model(config: dict[str, Any]) -> SensorGRU:
    model_config = config.get("model", {})
    return SensorGRU(
        input_size=int(model_config.get("input_size", 551)),
        hidden_size=int(model_config.get("hidden_size", 512)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
        output_size=int(model_config.get("output_size", 150)),
    )


def _optimizer(model: nn.Module, config: dict[str, Any], stage: dict[str, Any]):
    train_config = config.get("train", {})
    lr = float(stage.get("lr", train_config.get("lr", 1e-3)))
    weight_decay = float(stage.get("weight_decay", train_config.get("weight_decay", 1e-4)))
    optimizer_name = str(stage.get("optimizer", train_config.get("optimizer", "adamw"))).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _loss_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    target_config = config.get("target", {})
    return {
        "binary_sensor_indices_1based": _binary_indices(config),
        "continuous_weight": float(target_config.get("continuous_loss_weight", 1.0)),
        "binary_weight": float(target_config.get("binary_loss_weight", 1.0)),
    }


def _teacher_forcing_step(
    model: SensorGRU,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    pred, _hidden = model(batch["features"])
    losses = masked_sensor_loss(
        pred,
        batch["target"],
        batch["target_mask"],
        batch.get("sample_weight"),
        **_loss_kwargs(config),
    )
    metric_pred = sensor_feedback_from_prediction(pred, _binary_indices(config))
    return losses["loss"], losses, metric_pred


def _rollout_step(
    model: SensorGRU,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    context_len = int(stage.get("context_len", config.get("train", {}).get("seq_len", 1024)))
    rollout_steps = int(stage.get("rollout_steps", 128))
    rollout_weight = float(stage.get("rollout_loss_weight", 0.2))
    features = batch["features"]
    if features.shape[1] < context_len + rollout_steps:
        raise ValueError("rollout batch sequence is shorter than context_len + rollout_steps")

    context = features[:, :context_len]
    context_pred, hidden = model(context)
    teacher_losses = masked_sensor_loss(
        context_pred,
        batch["target"][:, :context_len],
        batch["target_mask"][:, :context_len],
        batch["sample_weight"][:, :context_len],
        **_loss_kwargs(config),
    )

    previous_sensor = sensor_feedback_from_prediction(
        context_pred[:, -1, :], _binary_indices(config)
    ).detach()
    previous_since_real = context[:, -1, SINCE_REAL_OFFSET]
    rollout_preds: list[torch.Tensor] = []

    for step in range(rollout_steps):
        base_features = features[:, context_len + step : context_len + step + 1]
        step_features, previous_since_real = make_rollout_input(
            base_features, previous_sensor, previous_since_real
        )
        step_pred, hidden = model(step_features, hidden)
        raw_pred = step_pred[:, 0, :]
        rollout_preds.append(raw_pred)
        previous_sensor = sensor_feedback_from_prediction(raw_pred, _binary_indices(config)).detach()

    rollout_pred = torch.stack(rollout_preds, dim=1)
    rollout_target = batch["target"][:, context_len : context_len + rollout_steps]
    rollout_mask = batch["target_mask"][:, context_len : context_len + rollout_steps]
    rollout_weight_tensor = batch["sample_weight"][:, context_len : context_len + rollout_steps]
    rollout_losses = masked_sensor_loss(
        rollout_pred,
        rollout_target,
        rollout_mask,
        rollout_weight_tensor,
        **_loss_kwargs(config),
    )
    loss = teacher_losses["loss"] + rollout_weight * rollout_losses["loss"]
    metrics = {
        "loss": loss,
        "teacher_loss": teacher_losses["loss"].detach(),
        "rollout_loss": rollout_losses["loss"].detach(),
        "continuous_loss": rollout_losses["continuous_loss"],
        "binary_loss": rollout_losses["binary_loss"],
    }
    metric_pred = sensor_feedback_from_prediction(rollout_pred, _binary_indices(config))
    return loss, metrics, metric_pred, rollout_target, rollout_mask


def _run_epoch(
    model: SensorGRU,
    loader,
    config: dict[str, Any],
    stage: dict[str, Any],
    device: torch.device,
    optimizer=None,
) -> dict[str, float | int]:
    train_mode = optimizer is not None
    model.train(train_mode)
    precision = str(stage.get("precision", config.get("train", {}).get("precision", "bf16")))
    max_batches = stage.get("max_batches_per_epoch")
    metric_totals: dict[str, float] = {}
    accuracy_correct = 0
    accuracy_total = 0
    batch_count = 0

    progress = tqdm(loader, leave=False, disable=not bool(stage.get("progress", True)))
    for batch in progress:
        if max_batches is not None and batch_count >= int(max_batches):
            break
        batch = _move_batch(batch, device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, precision):
            if stage.get("mode", "teacher_forcing") == "rollout":
                loss, losses, metric_pred, metric_target, metric_mask = _rollout_step(
                    model, batch, config, stage
                )
            else:
                loss, losses, metric_pred = _teacher_forcing_step(model, batch, config)
                metric_target = batch["target"]
                metric_mask = batch["target_mask"]

        if train_mode:
            loss.backward()
            clip = float(stage.get("gradient_clip", config.get("train", {}).get("gradient_clip", 0)))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        with torch.no_grad():
            for key, value in losses.items():
                if torch.is_tensor(value):
                    metric_totals[key] = metric_totals.get(key, 0.0) + float(value.detach().item())
            acc = normalized_accuracy_torch(
                metric_pred.detach().float(),
                metric_target.detach().float(),
                metric_mask.detach(),
                near_zero_threshold=float(config.get("metrics", {}).get("near_zero_threshold", 0.005)),
                relative_threshold=float(config.get("metrics", {}).get("relative_error_threshold", 0.20)),
            )
            accuracy_correct += int(acc["correct"])
            accuracy_total += int(acc["total"])

        batch_count += 1
        progress.set_postfix(loss=float(loss.detach().item()))

    if batch_count == 0:
        raise ValueError("No batches were processed")

    out = {key: value / batch_count for key, value in metric_totals.items()}
    out["accuracy"] = float(accuracy_correct / accuracy_total) if accuracy_total else 0.0
    out["accuracy_total"] = accuracy_total
    out["batches"] = batch_count
    return out


def _save_checkpoint(
    path: Path,
    model: SensorGRU,
    optimizer,
    config: dict[str, Any],
    stage: dict[str, Any],
    epoch: int,
    metrics: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "config": config,
            "stage": stage,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def _load_checkpoint(model: SensorGRU, path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])


def _run_dir(config: dict[str, Any]) -> Path:
    run_root = Path(config.get("run_root", "runs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_root / f"{config.get('run_name', 'gru_run')}_{timestamp}"


def _write_latest_pointer(run_dir: Path) -> None:
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(run_dir.name)
    except OSError:
        (run_dir.parent / "latest.txt").write_text(str(run_dir), encoding="utf-8")


def train(config: dict[str, Any]):
    """Train the configured GRU model and write a run directory."""
    seed_everything(int(config.get("train", {}).get("seed", 20260511)))
    device = _device()
    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    model = _build_model(config).to(device)
    stages = config.get("stages") or [
        {
            "name": "stage1_teacher_forcing",
            "mode": "teacher_forcing",
            **config.get("train", {}),
        }
    ]
    metrics_path = run_dir / "metrics.jsonl"
    best_checkpoint: Path | None = None

    for stage_index, stage in enumerate(stages, start=1):
        stage = dict(stage)
        stage_name = stage.get("name", f"stage{stage_index}")
        if stage.get("init_from") and best_checkpoint is not None:
            _load_checkpoint(model, best_checkpoint, device)
        elif stage.get("checkpoint"):
            _load_checkpoint(model, stage["checkpoint"], device)

        train_loader, val_loader = make_dataloaders(config, stage)
        optimizer = _optimizer(model, config, stage)
        epochs = int(stage.get("epochs", config.get("train", {}).get("epochs", 1)))
        best_metric = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(model, train_loader, config, stage, device, optimizer)
            with torch.no_grad():
                val_metrics = _run_epoch(model, val_loader, config, stage, device, optimizer=None)

            record = {
                "stage": stage_name,
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[train] {stage_name} epoch={epoch} "
                f"train_loss={train_metrics.get('loss', 0):.6f} "
                f"val_loss={val_metrics.get('loss', 0):.6f} "
                f"val_acc={val_metrics.get('accuracy', 0):.6f}",
                flush=True,
            )

            checkpoint_dir = run_dir / "checkpoints"
            _save_checkpoint(
                checkpoint_dir / f"{stage_name}_last.pt",
                model,
                optimizer,
                config,
                stage,
                epoch,
                val_metrics,
            )
            monitor = str(config.get("checkpoint", {}).get("monitor", "val_loss"))
            current = float(val_metrics.get(monitor.replace("val_", ""), val_metrics.get("loss", 0.0)))
            if current < best_metric:
                best_metric = current
                best_checkpoint = checkpoint_dir / f"{stage_name}_best.pt"
                _save_checkpoint(best_checkpoint, model, optimizer, config, stage, epoch, val_metrics)

        if bool(config.get("checkpoint", {}).get("save_last", True)):
            shutil.copy2(run_dir / "checkpoints" / f"{stage_name}_last.pt", run_dir / "checkpoints" / "last.pt")
        if best_checkpoint is not None and bool(config.get("checkpoint", {}).get("save_best", True)):
            shutil.copy2(best_checkpoint, run_dir / "checkpoints" / "best.pt")

    _write_latest_pointer(run_dir)
    print(f"[train] run_dir={run_dir}", flush=True)
    return run_dir
