"""RNN, GRU, or LSTM language model.

Moved from ``01_pretraining/main.py``.
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """A simple RNN/GRU/LSTM language model.

    Workflow:
        1. Embed token IDs.
        2. Pass through an RNN variant (LSTM/GRU/vanilla RNN).
        3. Project hidden states to vocabulary logits.

    Args:
        vocab_size: Size of tokenizer vocabulary.
        hidden_units: Embedding and hidden dimension.
        num_layers: Number of RNN layers.
        rnn_type: 'lstm', 'gru', or 'rnn'.
        dropout: Dropout probability applied between RNN layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_units: int = 256,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.embed = nn.Embedding(vocab_size, hidden_units)
        rnn_cls = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }.get(self.rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'.")
        self.rnn: nn.Module = rnn_cls(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_units, vocab_size)

        # Human-friendly model name
        self.friendly_name = f"RNN_type={self.rnn_type.upper()}_hiddenunits={hidden_units}_layers={num_layers}"

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        x = self.embed(input_ids)  # (B, T, hidden)
        rnn_out, _ = self.rnn(x)  # ignore hidden/cell states
        logits = self.fc_out(rnn_out)
        return logits
