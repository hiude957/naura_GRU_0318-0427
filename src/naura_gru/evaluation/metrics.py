"""Project accuracy metrics for normalized sensor predictions."""

from __future__ import annotations

import numpy as np
import torch

from naura_gru.training.losses import DEFAULT_BINARY_SENSOR_INDICES_1BASED, sensor_index_sets


def normalized_accuracy(
    pred, true, valid_mask, near_zero_threshold=0.005, relative_threshold=0.20
) -> dict[str, float | int]:
    """Compute project accuracy for normalized sensor values."""
    pred_arr = np.asarray(pred)
    true_arr = np.asarray(true)
    valid_arr = np.asarray(valid_mask).astype(bool)
    near_zero = np.abs(true_arr) <= near_zero_threshold
    correct = np.where(
        near_zero,
        np.abs(pred_arr) < near_zero_threshold,
        np.abs(pred_arr - true_arr) / np.maximum(np.abs(true_arr), near_zero_threshold)
        < relative_threshold,
    )
    valid_correct = correct & valid_arr
    total = int(valid_arr.sum())
    count = int(valid_correct.sum())
    return {
        "accuracy": float(count / total) if total else 0.0,
        "correct": count,
        "total": total,
    }


def normalized_accuracy_torch(
    pred: torch.Tensor,
    true: torch.Tensor,
    valid_mask: torch.Tensor,
    near_zero_threshold: float = 0.005,
    relative_threshold: float = 0.20,
) -> dict[str, float | int]:
    """Torch version used during validation without copying large tensors to NumPy."""
    valid = valid_mask.to(dtype=torch.bool)
    near_zero = true.abs() <= near_zero_threshold
    rel_error = (pred - true).abs() / true.abs().clamp_min(near_zero_threshold)
    correct = torch.where(near_zero, pred.abs() < near_zero_threshold, rel_error < relative_threshold)
    correct = correct & valid
    total = int(valid.sum().item())
    count = int(correct.sum().item())
    return {
        "accuracy": float(count / total) if total else 0.0,
        "correct": count,
        "total": total,
    }


def split_sensor_metrics_torch(
    pred: torch.Tensor,
    true: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    binary_sensor_indices_1based=None,
    near_zero_threshold: float = 0.005,
    relative_threshold: float = 0.20,
) -> dict[str, float | int]:
    """Compute separate continuous MAE/accuracy and binary MAE/class accuracy."""
    if pred.shape != true.shape or pred.shape != valid_mask.shape:
        raise ValueError("pred, true, and valid_mask must have the same shape")

    continuous_indices, binary_indices = sensor_index_sets(
        binary_sensor_indices_1based or DEFAULT_BINARY_SENSOR_INDICES_1BASED,
        sensor_dim=pred.shape[-1],
    )
    valid = valid_mask.to(dtype=torch.bool)

    out: dict[str, float | int] = {}
    combined_correct = 0
    combined_total = 0

    if continuous_indices:
        continuous_idx = torch.as_tensor(continuous_indices, device=pred.device, dtype=torch.long)
        cont_pred = pred.index_select(-1, continuous_idx)
        cont_true = true.index_select(-1, continuous_idx)
        cont_valid = valid.index_select(-1, continuous_idx)
        cont_abs_error = (cont_pred - cont_true).abs()
        cont_near_zero = cont_true.abs() <= near_zero_threshold
        cont_rel_error = cont_abs_error / cont_true.abs().clamp_min(near_zero_threshold)
        cont_correct = torch.where(
            cont_near_zero,
            cont_pred.abs() < near_zero_threshold,
            cont_rel_error < relative_threshold,
        )
        cont_correct = cont_correct & cont_valid
        cont_total = int(cont_valid.sum().item())
        cont_count = int(cont_correct.sum().item())
        cont_mae_sum = float((cont_abs_error * cont_valid.to(dtype=cont_abs_error.dtype)).sum().item())
        out.update(
            {
                "continuous_accuracy": float(cont_count / cont_total) if cont_total else 0.0,
                "continuous_correct": cont_count,
                "continuous_total": cont_total,
                "continuous_mae": float(cont_mae_sum / cont_total) if cont_total else 0.0,
                "continuous_mae_sum": cont_mae_sum,
            }
        )
        combined_correct += cont_count
        combined_total += cont_total
    else:
        out.update(
            {
                "continuous_accuracy": 0.0,
                "continuous_correct": 0,
                "continuous_total": 0,
                "continuous_mae": 0.0,
                "continuous_mae_sum": 0.0,
            }
        )

    if binary_indices:
        binary_idx = torch.as_tensor(binary_indices, device=pred.device, dtype=torch.long)
        bin_pred = pred.index_select(-1, binary_idx)
        bin_true = true.index_select(-1, binary_idx)
        bin_valid = valid.index_select(-1, binary_idx)
        bin_abs_error = (bin_pred - bin_true).abs()
        bin_correct = ((bin_pred >= 0.5) == (bin_true >= 0.5)) & bin_valid
        bin_total = int(bin_valid.sum().item())
        bin_count = int(bin_correct.sum().item())
        bin_mae_sum = float((bin_abs_error * bin_valid.to(dtype=bin_abs_error.dtype)).sum().item())
        out.update(
            {
                "binary_accuracy": float(bin_count / bin_total) if bin_total else 0.0,
                "binary_correct": bin_count,
                "binary_total": bin_total,
                "binary_mae": float(bin_mae_sum / bin_total) if bin_total else 0.0,
                "binary_mae_sum": bin_mae_sum,
            }
        )
        combined_correct += bin_count
        combined_total += bin_total
    else:
        out.update(
            {
                "binary_accuracy": 0.0,
                "binary_correct": 0,
                "binary_total": 0,
                "binary_mae": 0.0,
                "binary_mae_sum": 0.0,
            }
        )

    out.update(
        {
            "split_accuracy": float(combined_correct / combined_total) if combined_total else 0.0,
            "split_correct": combined_correct,
            "split_total": combined_total,
        }
    )
    return out
