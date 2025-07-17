# %%
# Load data
from datasets import load_dataset
from transformers import AutoTokenizer

# Load raw WikiText (recommended for GPT-2)
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token


# %%
def tokenize(batch):
    return tok(batch["text"], truncation=True)


dst = ds.map(tokenize, num_proc=8, batched=True, remove_columns=["text"])

# %%
block_size = 128  # sequence length for training


def group_texts(examples):
    # Concatenate all texts
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # Drop the remainder
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = dst.map(group_texts, batched=True)

# %%
from transformers.data.data_collator import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# %%

from torch.utils.data import DataLoader

lm_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

batch_size = 16

train_loader = DataLoader(
    lm_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator  # type: ignore
)

valid_loader = DataLoader(
    lm_datasets["validation"], batch_size=batch_size, shuffle=True, collate_fn=data_collator  # type: ignore
)

# %%
import torch
import torch.nn as nn


# Training loop
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print("Using device:", device)


class TinyMLPLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, d_hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.GELU(),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.mlp(x)
        return self.lm_head(x)


class TinyRNNLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_hidden: int = 256,
        num_layers: int = 1,
        rnn_type: str = "gru",  # "rnn", "gru", or "lstm"
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                d_model, d_hidden, num_layers=num_layers, batch_first=True
            )
            self.is_lstm = False
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                d_model, d_hidden, num_layers=num_layers, batch_first=True
            )
            self.is_lstm = True
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(
                d_model, d_hidden, num_layers=num_layers, batch_first=True
            )
            self.is_lstm = False
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        self.lm_head = nn.Linear(d_hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embed(input_ids)
        out, _ = self.rnn(x)
        return self.lm_head(out)


class TinyTransformerLM(nn.Module):
    """A minimal Transformer encoder language model for small-scale experiments."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        # 1) Token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        # 2) Stacked self-attention blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, C) order to match the rest of the script
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) Final language-model head (weights tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

    def forward(
        self, input_ids: torch.LongTensor
    ) -> torch.Tensor:  # (B, T) → (B, T, V)
        seq_len = input_ids.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.pos_embed.size(1)}"
            )

        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.lm_head(x)


batch = next(iter(train_loader))
input_ids = batch["input_ids"].to(device)


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    encoded = torch.tensor(tokenizer.encode(start_context)).to(device).unsqueeze(0)
    with torch.no_grad():
        out_ids = generate(model, encoded, 50)
    decoded_text = tokenizer.decode(out_ids)
    print(decoded_text)


def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)

    return idx[0]


import torch.optim as optim

# model = TinyMLPLM(len(tok), d_model=512, d_hidden=256).to(device)
# model = TinyRNNLM(len(tok), d_model=128, d_hidden=256, num_layers=1).to(device)
model = TinyTransformerLM(
    vocab_size=len(tok),
    d_model=256,
    n_heads=8,
    num_layers=4,
    max_seq_len=block_size,
).to(device)

models = [
    ("basic mlp", TinyMLPLM(len(tok), d_model=512, d_hidden=256).to(device)),
    (
        "basic rnn",
        TinyRNNLM(len(tok), d_model=128, d_hidden=256, num_layers=1).to(device),
    ),
    (
        "basic transformer",
        TinyTransformerLM(
            vocab_size=len(tok),
            d_model=256,
            n_heads=8,
            num_layers=4,
            max_seq_len=block_size,
        ).to(device),
    ),
]

num_epochs = 2
learning_rate = 5e-4

for label, model in models:
    print(f"---------\nStarting training run for: {label}\n-----------")
    generate_and_print_sample(model, tok, device, "the quick brown fox")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Shift input_ids and labels for next-token prediction
            input_ids_shifted = input_ids[:, :-1]
            labels_shifted = labels[:, 1:]

            optimizer.zero_grad()
            logits = model(input_ids_shifted)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels_shifted.reshape(-1))
            loss.backward()
            optimizer.step()

            if (i % 50 == 0) & (i > 0):
                print(f"batch {i} of {len(train_loader)}. loss = {loss.item()}")

            if i == 100:
                break

        model.eval()
        with torch.no_grad():
            total_valid_loss = 0.0
            total_valid_tokens = 0

            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward on full sequence for proper context
                logits = model(input_ids)  # -> [B, T, V]

                # Shift to create next-token targets
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                # Compute mean loss over *non-ignored* tokens
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                # Count valid tokens (non-pad)
                valid_tokens = (shift_labels != tok.pad_token_id).sum().item()

                # Weight by valid token count
                total_valid_loss += loss.item() * valid_tokens
                total_valid_tokens += valid_tokens

            avg_valid_loss = total_valid_loss / total_valid_tokens
            print(f"Epoch {epoch+1} Validation loss: {avg_valid_loss:.4f}")
        generate_and_print_sample(model, tok, device, "the quick brown fox")
        model.train()
