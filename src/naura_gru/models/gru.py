"""GRU model definition for full sensor prediction."""

from __future__ import annotations

import torch
from torch import nn


class SensorGRU(nn.Module):
    """GRU that predicts all normalized sensor values for each input step."""

    def __init__(
        self,
        input_size: int = 551,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 150,
        head_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        if head_layer_norm:
            self.head = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
            )
        else:
            self.head = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw full-sensor predictions and the final hidden state.

        Continuous sensor dimensions use the raw output directly for MAE.
        Binary sensor dimensions are treated as logits by the loss and rollout code.
        """
        output, hidden = self.gru(x, hidden)
        return self.head(output), hidden
