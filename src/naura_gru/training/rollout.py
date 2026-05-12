"""Helpers for closed-loop sensor feedback in the fixed 551-dim feature layout."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from naura_gru.training.dataset import SENSOR_DIM
from naura_gru.training.losses import sensor_index_sets

ACTION_DIM = 122
MASK_START = SENSOR_DIM
MASK_END = SENSOR_DIM * 2
SOURCE_START = SENSOR_DIM * 2 + ACTION_DIM * 2
SOURCE_END = SOURCE_START + 4
TIME_START = SOURCE_END
SINCE_ACTION_OFFSET = TIME_START + 1
SINCE_REAL_OFFSET = TIME_START + 2


def sensor_feedback_from_prediction(
    pred: torch.Tensor,
    binary_sensor_indices_1based: Iterable[int] | None = None,
) -> torch.Tensor:
    """Convert raw model output to sensor values that can be fed back as input."""
    feedback = pred.clone()
    continuous_indices, binary_indices = sensor_index_sets(
        binary_sensor_indices_1based, sensor_dim=pred.shape[-1]
    )
    if continuous_indices:
        continuous_idx = torch.as_tensor(continuous_indices, device=pred.device, dtype=torch.long)
        feedback.index_copy_(-1, continuous_idx, feedback.index_select(-1, continuous_idx).clamp(0, 1))
    if binary_indices:
        binary_idx = torch.as_tensor(binary_indices, device=pred.device, dtype=torch.long)
        feedback.index_copy_(-1, binary_idx, torch.sigmoid(pred.index_select(-1, binary_idx)))
    return feedback


def make_rollout_input(
    base_features: torch.Tensor,
    sensor_feedback: torch.Tensor,
    previous_since_real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace real sensor fields in one future feature step with closed-loop feedback."""
    if base_features.ndim != 3 or base_features.shape[1] != 1:
        raise ValueError("base_features must have shape [batch, 1, feature_dim]")
    if sensor_feedback.ndim != 2:
        raise ValueError("sensor_feedback must have shape [batch, sensor_dim]")

    out = base_features.clone()
    out[:, 0, :SENSOR_DIM] = sensor_feedback
    out[:, 0, MASK_START:MASK_END] = 0.0
    out[:, 0, SOURCE_START:SOURCE_END] = 0.0
    out[:, 0, SOURCE_START + 2] = 1.0

    delta_t = out[:, 0, TIME_START].clamp_min(0.0)
    since_real = (previous_since_real + delta_t).clamp(0.0, 1.0)
    out[:, 0, SINCE_REAL_OFFSET] = since_real
    return out, since_real
