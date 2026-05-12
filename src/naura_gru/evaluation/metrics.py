"""Project accuracy metrics for normalized sensor predictions."""

from __future__ import annotations

import numpy as np
import torch


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
