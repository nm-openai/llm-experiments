# %%
import torch
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# 1.  Load the corpus  -------------------------------------------------------
dataset = load_dataset("karpathy/tiny_shakespeare")

with open(os.path.join("outputs", "train_text.txt"), "w", encoding="utf-8") as f:
    for line in dataset["train"]["text"]:
        f.write(line + "\n")


# %%
# Train some models!~
def save_model(model, filepath):
    """
    Save a PyTorch model's state_dict to the specified filepath, creating directories if needed.
    Args:
        model: The PyTorch model (nn.Module)
        filepath: Full path to save the checkpoint (including filename)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


# 2.  Pick / build a tokenizer  ---------------------------------------------
# Any HF tokenizer works.  Using a GPT-style tokenizer for demonstration.
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or a custom one
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token


# 3.  Tokenize line by line (fast batched map)  ------------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        return_attention_mask=False,  # masks are re-built later
        truncation=False,
    )  # keep full lines for now


tokenized = dataset.map(
    tokenize,
    batched=True,  # process ~1k rows per call (fast)
    remove_columns=["text"],  # drop the raw string to save RAM
    num_proc=16,
    desc="Tokenizing",
)

# 4.  Chunk into fixed-length blocks  ----------------------------------------
block_size = 128


def group_texts(batch):
    # Concatenate then split into blocks of block_size
    joined = sum(batch["input_ids"], [])  # 1-D list
    # Drop remainder to keep blocks equal size
    n_full = len(joined) // block_size
    joined = joined[: n_full * block_size]
    # Reshape
    input_ids = [joined[i : i + block_size] for i in range(0, len(joined), block_size)]
    return {"input_ids": input_ids}


lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized["train"].column_names,
    desc=f"Grouping into {block_size}-token blocks",
)

# 5.  (Optional) prepare a data collator for MLM or causal LM ---------------
causal_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # True for BERT-style masked LM
)

# Now `lm_dataset["train"]` (and "validation"/"test") contain contiguous
# 1024-token blocks of integer IDs, ready for use with a Trainer, Lightning
# module, Accelerator, etc.

# 6.  Simple PyTorch model ----------------------------------------------------
import torch.nn as nn
from torch.utils.data import DataLoader


# Select the best available device: prefer Apple Silicon MPS, then CUDA, else CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Free cached memory on the chosen accelerator (helps when re-running cells)
if device == "mps":
    torch.mps.empty_cache()
elif device == "cuda":
    torch.cuda.empty_cache()

# 7.1 Training helper ---------------------------------------------------------


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    device,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 40,
    temperature: float = 1.0,
) -> str:
    """Quick greedy-ish generation helper tied to the current model."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def train_and_evaluate(
    model: nn.Module,
    epochs: int = 5,
    n_train_samples: int = None,
    n_val_samples: int = None,
    learning_rate: float = 1e-3,
) -> None:
    """Train and evaluate an arbitrary ``torch.nn.Module``.

    Args:
        model: Instantiated model to train.
        model_name: Used for logging and checkpoint filename.
        epochs: Number of epochs.
    """
    # Optionally sample down the train and validation datasets
    train_data = lm_dataset["train"].shuffle(seed=42)

    val_data = lm_dataset["validation"].shuffle(seed=42)
    if n_train_samples is not None:
        train_data = train_data.select(range(n_train_samples))
    if n_val_samples is not None:
        val_data = val_data.select(range(n_val_samples))

    # Print number of rows and tokens per epoch
    num_rows = len(train_data)
    # Assumes each sample has an 'input_ids' field that is a list of tokens
    num_tokens = sum(len(sample["input_ids"]) for sample in train_data)
    print(f"Rows per epoch: {num_rows}")
    print(f"Tokens per epoch: {num_tokens}")

    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        collate_fn=causal_collator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        collate_fn=causal_collator,
    )

    # Move model and clear caches before starting
    model = model.to(device)
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print(f"Training {model.friendly_name} for {epochs} epoch(s)")
    print(f"{'=' * 80}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Memory footprint (parameters + activations for one batch)
    hidden_units = model.embed.embedding_dim
    param_mem_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    print(f"Parameter memory usage: {param_mem_mb:.2f} MB")

    # Attempt activation-memory estimate if model has an ``embed`` attr
    if hasattr(model, "embed") and hasattr(model.embed, "embedding_dim"):
        hidden_units = model.embed.embedding_dim
        act_mem_mb = train_loader.batch_size * block_size * hidden_units * 4 / (1024**2)
        print(f"1-batch activations: {act_mem_mb:.2f} MB")

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # ---------------- Training ----------------
        model.train()
        total_loss = 0.0
        print_every = 100  # adjust n as desired
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------------- Validation ----------------
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids[:, :-1])
                val_loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
                val_total_loss += val_loss.item()
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # ---------------- Logging ----------------
        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss = {avg_train_loss:.4f}, "
            f"val_loss = {avg_val_loss:.4f}"
        )

        # ---------------- Sample Generation ----------------
        print("\n--- Sample generation ---")
        print(generate(model, tokenizer, device, "What authority surfeits", 40))
        print("-" * 60)

    # Quick final sample
    print("\n=== Final sample after training ===")
    final_sample = generate(model, tokenizer, device, "Very well; and could", 100)
    print(final_sample)

    # Print summary of losses
    print("\n=== Losses ===")
    print("\nTraining loss per epoch:", [f"{l:.4f}" for l in train_losses])
    print("Validation loss per epoch:", [f"{l:.4f}" for l in val_losses])

    # Save checkpoint **before** freeing resources
    ckpt_path = os.path.join(
        "outputs", "model_checkpoints", f"{model.friendly_name}.pt"
    )
    save_model(model, ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")

    print(f"\nTraining complete for {model.friendly_name}")

    res = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_sample": final_sample,
        "model_name": model.friendly_name,
    }

    # Explicitly free GPU/Apple-Silicon memory before the next run
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return res


from models import FCModel, RNNModel, TransformerModel

models = [
    # ----------------- Fully Connected Models -----------------
    FCModel(vocab_size=len(tokenizer), hidden_units=128, num_layers=2),  # Modest FC
    FCModel(vocab_size=len(tokenizer), hidden_units=512, num_layers=6),  # Large FC
    # ----------------- Transformer Models ---------------------
    TransformerModel(
        vocab_size=len(tokenizer),
        hidden_units=128,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
    ),  # Modest Transformer
    TransformerModel(
        vocab_size=len(tokenizer),
        hidden_units=512,
        num_layers=8,
        num_heads=8,
        max_seq_len=block_size,
        dropout=0.1,
    ),  # Large Transformer
    # ----------------- RNN Models -----------------------------
    RNNModel(
        vocab_size=len(tokenizer),
        hidden_units=128,
        num_layers=2,
        rnn_type="rnn",
    ),  # Modest Simple RNN
    RNNModel(
        vocab_size=len(tokenizer),
        hidden_units=512,
        num_layers=4,
        rnn_type="lstm",
        dropout=0.2,
    ),  # Large LSTM
    RNNModel(
        vocab_size=len(tokenizer),
        hidden_units=512,
        num_layers=4,
        rnn_type="gru",
        dropout=0.2,
    ),  # Large GRU
]

res = []
for m in models:
    compiled_m = torch.compile(m)
    res_i = train_and_evaluate(compiled_m, epochs=5, learning_rate=0.001)
    res.append(res_i)

filename = os.path.join("outputs", "run_res.csv")
pd.DataFrame(res).to_csv(filename, index=False)
