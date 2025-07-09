"""Lightweight Transformer encoder language model.

Moved from ``01_pretraining/main.py``.
"""

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """A lightweight Transformer encoder language model suitable for small-scale experiments.

    Workflow:
        1. Token + positional embeddings
        2. Stacked ``nn.TransformerEncoder`` layers
        3. Projection to vocabulary logits
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_units: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be divisible by num_heads")

        # Store for string representation
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Token and positional embeddings
        self.embed = nn.Embedding(vocab_size, hidden_units)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_units))
        self.dropout = nn.Dropout(dropout)

        # Stacked self-attention blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final LM head
        self.fc_out = nn.Linear(hidden_units, vocab_size)

        # Human-friendly model name
        self.friendly_name = f"Transformer_hiddenunits={hidden_units}_layers={num_layers}_heads={num_heads}"

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        seq_len = input_ids.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError("Sequence length exceeds maximum positional embeddings")

        x = self.embed(input_ids) + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits
