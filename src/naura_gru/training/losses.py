"""Masked loss and sample-weight utilities."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


DEFAULT_BINARY_SENSOR_INDICES_1BASED = [
    22,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    146,
    147,
    148,
]


def to_zero_based_indices(indices: Iterable[int], sensor_dim: int = 150) -> list[int]:
    """Convert 1-based sensor ids to sorted 0-based indices."""
    out = sorted({int(index) - 1 for index in indices})
    invalid = [index + 1 for index in out if index < 0 or index >= sensor_dim]
    if invalid:
        raise ValueError(f"Invalid 1-based sensor indices: {invalid}")
    return out


def sensor_index_sets(
    binary_sensor_indices_1based: Iterable[int] | None = None, sensor_dim: int = 150
) -> tuple[list[int], list[int]]:
    """Return continuous and binary 0-based sensor indices."""
    binary = to_zero_based_indices(
        binary_sensor_indices_1based or DEFAULT_BINARY_SENSOR_INDICES_1BASED, sensor_dim
    )
    binary_set = set(binary)
    continuous = [index for index in range(sensor_dim) if index not in binary_set]
    return continuous, binary


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_sensor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    continuous_sample_weight: torch.Tensor | None = None,
    continuous_action_mask: torch.Tensor | None = None,
    continuous_change_mask: torch.Tensor | None = None,
    *,
    binary_sensor_indices_1based: Iterable[int] | None = None,
    continuous_weight: float = 1.0,
    binary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute full-sensor loss with MAE for continuous sensors and BCE for binary sensors."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    if pred.shape != loss_mask.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != loss_mask shape {tuple(loss_mask.shape)}")

    mask = loss_mask.to(dtype=pred.dtype)
    if sample_weight is not None:
        weight = sample_weight.to(dtype=pred.dtype).unsqueeze(-1)
        mask = mask * weight
    continuous_weight_tensor = None
    if continuous_sample_weight is not None:
        continuous_weight_tensor = continuous_sample_weight.to(dtype=pred.dtype)
        if continuous_weight_tensor.ndim == pred.ndim - 1:
            continuous_weight_tensor = continuous_weight_tensor.unsqueeze(-1)
        elif continuous_weight_tensor.ndim != pred.ndim:
            raise ValueError(
                "continuous_sample_weight must have shape [batch, time] or "
                "[batch, time, sensor_dim]"
            )
    action_mask_tensor = None
    if continuous_action_mask is not None:
        action_mask_tensor = continuous_action_mask.to(dtype=pred.dtype).unsqueeze(-1)
    change_mask_tensor = None
    if continuous_change_mask is not None:
        change_mask_tensor = continuous_change_mask.to(dtype=pred.dtype)
        if change_mask_tensor.ndim == pred.ndim - 1:
            change_mask_tensor = change_mask_tensor.unsqueeze(-1)
        elif change_mask_tensor.ndim != pred.ndim:
            raise ValueError(
                "continuous_change_mask must have shape [batch, time] or "
                "[batch, time, sensor_dim]"
            )

    continuous_indices, binary_indices = sensor_index_sets(
        binary_sensor_indices_1based, sensor_dim=pred.shape[-1]
    )
    continuous_idx = torch.as_tensor(continuous_indices, device=pred.device, dtype=torch.long)
    binary_idx = torch.as_tensor(binary_indices, device=pred.device, dtype=torch.long)

    continuous_loss = pred.new_tensor(0.0)
    binary_loss = pred.new_tensor(0.0)
    continuous_action_loss = pred.new_tensor(0.0)
    continuous_change_loss = pred.new_tensor(0.0)
    continuous_points = pred.new_tensor(0.0)
    binary_points = pred.new_tensor(0.0)
    continuous_action_points = pred.new_tensor(0.0)
    continuous_change_points = pred.new_tensor(0.0)

    if continuous_idx.numel() > 0:
        cont_pred = pred.index_select(-1, continuous_idx)
        cont_target = target.index_select(-1, continuous_idx)
        cont_mask = mask.index_select(-1, continuous_idx)
        cont_loss_mask = cont_mask
        if continuous_weight_tensor is not None:
            if continuous_weight_tensor.shape[-1] == 1:
                cont_loss_mask = cont_loss_mask * continuous_weight_tensor
            elif continuous_weight_tensor.shape[-1] == pred.shape[-1]:
                cont_loss_mask = cont_loss_mask * continuous_weight_tensor.index_select(
                    -1, continuous_idx
                )
            elif continuous_weight_tensor.shape[-1] == continuous_idx.numel():
                cont_loss_mask = cont_loss_mask * continuous_weight_tensor
            else:
                raise ValueError("continuous_sample_weight has incompatible sensor dimension")
        abs_error = torch.abs(cont_pred - cont_target)
        continuous_loss = _masked_mean(abs_error, cont_loss_mask)
        continuous_points = cont_loss_mask.sum()
        if action_mask_tensor is not None:
            cont_action_mask = cont_mask * action_mask_tensor
            continuous_action_loss = _masked_mean(abs_error, cont_action_mask)
            continuous_action_points = cont_action_mask.sum()
        if change_mask_tensor is not None:
            if change_mask_tensor.shape[-1] == 1:
                cont_change_mask = cont_mask * change_mask_tensor
            elif change_mask_tensor.shape[-1] == pred.shape[-1]:
                cont_change_mask = cont_mask * change_mask_tensor.index_select(-1, continuous_idx)
            elif change_mask_tensor.shape[-1] == continuous_idx.numel():
                cont_change_mask = cont_mask * change_mask_tensor
            else:
                raise ValueError("continuous_change_mask has incompatible sensor dimension")
            continuous_change_loss = _masked_mean(abs_error, cont_change_mask)
            continuous_change_points = cont_change_mask.sum()

    if binary_idx.numel() > 0:
        bin_pred = pred.index_select(-1, binary_idx)
        bin_target = target.index_select(-1, binary_idx).clamp(0.0, 1.0)
        bin_mask = mask.index_select(-1, binary_idx)
        bce = F.binary_cross_entropy_with_logits(bin_pred, bin_target, reduction="none")
        binary_loss = _masked_mean(bce, bin_mask)
        binary_points = bin_mask.sum()

    total = float(continuous_weight) * continuous_loss + float(binary_weight) * binary_loss
    return {
        "loss": total,
        "continuous_loss": continuous_loss.detach(),
        "binary_loss": binary_loss.detach(),
        "continuous_action_loss": continuous_action_loss.detach(),
        "continuous_change_loss": continuous_change_loss.detach(),
        "continuous_points": continuous_points.detach(),
        "binary_points": binary_points.detach(),
        "continuous_action_points": continuous_action_points.detach(),
        "continuous_change_points": continuous_change_points.detach(),
    }
