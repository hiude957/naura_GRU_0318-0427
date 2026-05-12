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
from naura_gru.training.dataset import SENSOR_DIM, make_dataloaders
from naura_gru.training.losses import (
    DEFAULT_BINARY_SENSOR_INDICES_1BASED,
    masked_sensor_loss,
)
from naura_gru.training.rollout import (
    SINCE_ACTION_OFFSET,
    SINCE_REAL_OFFSET,
    make_rollout_input,
    sensor_feedback_from_prediction,
)
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
        head_layer_norm=bool(model_config.get("head_layer_norm", False)),
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


def _time_feature_cap_ms(config: dict[str, Any], stage: dict[str, Any]) -> float:
    data_config = config.get("data", {})
    return float(
        stage.get(
            "time_feature_cap_ms",
            data_config.get("time_feature_cap_ms", config.get("time_feature_cap_ms", 86_400_000)),
        )
    )


def _continuous_action_weight_tensors(
    features: torch.Tensor,
    config: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    target_config = config.get("target", {})
    action_config = dict(target_config.get("continuous_action_weight", {}))
    action_config.update(stage.get("continuous_action_weight", {}))
    if not bool(action_config.get("enabled", False)):
        return None, None

    window_ms = float(action_config.get("window_ms", 10_000))
    weight = float(action_config.get("weight", 2.0))
    if weight <= 0:
        raise ValueError("continuous_action_weight.weight must be positive")
    threshold = window_ms / _time_feature_cap_ms(config, stage)
    action_mask = features[..., SINCE_ACTION_OFFSET] <= threshold
    weight_tensor = torch.where(
        action_mask,
        features.new_full(action_mask.shape, weight),
        features.new_ones(action_mask.shape),
    )
    return weight_tensor, action_mask


def _continuous_change_weight_tensors(
    features: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
    target_config = config.get("target", {})
    change_config = dict(target_config.get("continuous_change_weight", {}))
    change_config.update(stage.get("continuous_change_weight", {}))
    if not bool(change_config.get("enabled", False)):
        return None, None, None

    change_threshold = float(change_config.get("change_threshold", 0.005))
    active_threshold = float(change_config.get("active_threshold", 0.01))
    change_weight = float(change_config.get("change_loss_weight", 3.0))
    max_weight = float(change_config.get("max_loss_weight", 5.0))
    if change_weight <= 0:
        raise ValueError("continuous_change_weight.change_loss_weight must be positive")

    current_sensor = features[..., :SENSOR_DIM]
    change_mask = (target - current_sensor).abs() >= change_threshold
    if active_threshold > 0:
        active_mask = (target.abs() >= active_threshold) | (current_sensor.abs() >= active_threshold)
        change_mask = change_mask & active_mask
    weight_tensor = torch.where(
        change_mask,
        features.new_full(change_mask.shape, change_weight),
        features.new_ones(change_mask.shape),
    )
    if max_weight > 0:
        weight_tensor = weight_tensor.clamp_max(max_weight)
    return weight_tensor, change_mask, max_weight if max_weight > 0 else None


def _continuous_loss_weight_tensors(
    features: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    action_weight, action_mask = _continuous_action_weight_tensors(features, config, stage)
    change_weight, change_mask, max_weight = _continuous_change_weight_tensors(
        features, target, config, stage
    )
    if action_weight is None:
        combined_weight = change_weight
    elif change_weight is None:
        combined_weight = action_weight
    else:
        combined_weight = action_weight.unsqueeze(-1) * change_weight
    if combined_weight is not None and max_weight is not None:
        combined_weight = combined_weight.clamp_max(max_weight)
    return combined_weight, action_mask, change_mask


def _teacher_forcing_step(
    model: SensorGRU,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    continuous_sample_weight, continuous_action_mask, continuous_change_mask = (
        _continuous_loss_weight_tensors(batch["features"], batch["target"], config, stage)
    )
    pred, _hidden = model(batch["features"])
    losses = masked_sensor_loss(
        pred,
        batch["target"],
        batch["target_mask"],
        batch.get("sample_weight"),
        continuous_sample_weight=continuous_sample_weight,
        continuous_action_mask=continuous_action_mask,
        continuous_change_mask=continuous_change_mask,
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
    mode = str(stage.get("mode", "teacher_forcing"))
    rollout_only = mode == "rollout_only"
    context_len = int(stage.get("context_len", config.get("train", {}).get("seq_len", 1024)))
    rollout_steps = int(stage.get("rollout_steps", 128))
    rollout_weight = float(stage.get("rollout_loss_weight", 0.2))
    features = batch["features"]
    if features.shape[1] < context_len + rollout_steps:
        raise ValueError("rollout batch sequence is shorter than context_len + rollout_steps")

    context = features[:, :context_len]
    context_pred, hidden = model(context)
    teacher_losses = None
    if not rollout_only:
        context_target = batch["target"][:, :context_len]
        context_weight, context_action_mask, context_change_mask = _continuous_loss_weight_tensors(
            context,
            context_target,
            config,
            stage,
        )
        teacher_losses = masked_sensor_loss(
            context_pred,
            context_target,
            batch["target_mask"][:, :context_len],
            batch["sample_weight"][:, :context_len],
            continuous_sample_weight=context_weight,
            continuous_action_mask=context_action_mask,
            continuous_change_mask=context_change_mask,
            **_loss_kwargs(config),
        )

    if rollout_only:
        previous_sensor = context[:, -1, :SENSOR_DIM].detach()
    else:
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
    continuous_sample_weight, continuous_action_mask, continuous_change_mask = (
        _continuous_loss_weight_tensors(
            features[:, context_len : context_len + rollout_steps],
            rollout_target,
            config,
            stage,
        )
    )
    rollout_losses = masked_sensor_loss(
        rollout_pred,
        rollout_target,
        rollout_mask,
        rollout_weight_tensor,
        continuous_sample_weight=continuous_sample_weight,
        continuous_action_mask=continuous_action_mask,
        continuous_change_mask=continuous_change_mask,
        **_loss_kwargs(config),
    )
    if rollout_only:
        loss = rollout_losses["loss"]
    else:
        loss = teacher_losses["loss"] + rollout_weight * rollout_losses["loss"]
    metrics = {
        "loss": loss,
        "rollout_loss": rollout_losses["loss"].detach(),
        "continuous_loss": rollout_losses["continuous_loss"],
        "binary_loss": rollout_losses["binary_loss"],
        "continuous_action_loss": rollout_losses["continuous_action_loss"],
        "continuous_action_points": rollout_losses["continuous_action_points"],
        "continuous_change_loss": rollout_losses["continuous_change_loss"],
        "continuous_change_points": rollout_losses["continuous_change_points"],
    }
    if teacher_losses is not None:
        metrics["teacher_loss"] = teacher_losses["loss"].detach()
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
            if stage.get("mode", "teacher_forcing") in {"rollout", "rollout_only"}:
                loss, losses, metric_pred, metric_target, metric_mask = _rollout_step(
                    model, batch, config, stage
                )
            else:
                loss, losses, metric_pred = _teacher_forcing_step(model, batch, config, stage)
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
        best_accuracy = float("-inf")

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
            current_accuracy = float(val_metrics.get("accuracy", 0.0))
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                _save_checkpoint(
                    checkpoint_dir / f"{stage_name}_best_acc.pt",
                    model,
                    optimizer,
                    config,
                    stage,
                    epoch,
                    val_metrics,
                )

        if bool(config.get("checkpoint", {}).get("save_last", True)):
            shutil.copy2(run_dir / "checkpoints" / f"{stage_name}_last.pt", run_dir / "checkpoints" / "last.pt")
        if best_checkpoint is not None and bool(config.get("checkpoint", {}).get("save_best", True)):
            shutil.copy2(best_checkpoint, run_dir / "checkpoints" / "best.pt")
        best_acc_path = run_dir / "checkpoints" / f"{stage_name}_best_acc.pt"
        if best_acc_path.exists() and bool(config.get("checkpoint", {}).get("save_best_accuracy", True)):
            shutil.copy2(best_acc_path, run_dir / "checkpoints" / "best_acc.pt")

    _write_latest_pointer(run_dir)
    print(f"[train] run_dir={run_dir}", flush=True)
    return run_dir
