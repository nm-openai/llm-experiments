"""Fully connected language model.

Moved from ``01_pretraining/main.py`` to keep the main script concise.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FCModel"]


class FCModel(nn.Module):
    """A deeper yet lightweight language model with multiple fully connected layers.

    Workflow:
        1. Embed token IDs into a ``hidden_units``-dimensional space via
           :class:`torch.nn.Embedding`.
        2. Optionally pass through ``num_layers − 1`` hidden ``Linear → ReLU``
           blocks to add non-linear depth.
        3. Project back to ``vocab_size`` for standard cross-entropy training.

    The additional depth typically improves modeling capacity while keeping the
    parameter count modest compared to Transformer models.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_units: int = 256,
        num_layers: int = 3,
    ) -> None:
        """Construct the model.

        Args:
            vocab_size: Size of the tokenizer vocabulary.
            hidden_units: Dimensionality of the embeddings and all hidden FC layers.
            num_layers: Total number of ``Linear → ReLU`` layers *after* the
                embedding. Must be ≥ 1. The final projection to ``vocab_size`` is
                added automatically.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Store attributes needed for __str__
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        self.embed = nn.Embedding(vocab_size, hidden_units)
        self.relu = nn.ReLU()

        # Build hidden stack (Linear → ReLU repeated)
        hidden_blocks: list[nn.Module] = []
        for _ in range(num_layers - 1):  # last layer handled by fc_out
            hidden_blocks.append(nn.Linear(hidden_units, hidden_units))
            hidden_blocks.append(nn.ReLU())
        self.hidden_stack: nn.Module = (
            nn.Sequential(*hidden_blocks) if hidden_blocks else nn.Identity()
        )

        # Final projection back to vocab size
        self.fc_out = nn.Linear(hidden_units, vocab_size)

    # ---------------------------------------------------------------------
    # Forward / helpers
    # ---------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        x = self.embed(input_ids)  # (B, T, hidden)
        x = self.relu(x)  # initial non-linearity
        x = self.hidden_stack(x)  # pass through hidden FC layers
        logits = self.fc_out(x)  # (B, T, vocab)
        return logits

    def __str__(self) -> str:
        """Human-friendly identifier, e.g. ``fc_512hu_3layers``."""
        return f"fc_{self.hidden_units}hu_{self.num_layers}layers"
