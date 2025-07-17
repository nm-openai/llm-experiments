# %%
# import dataset and create dataloader
from datasets import load_dataset

ds = load_dataset(path="Salesforce/wikitext", name="wikitext-2-raw-v1")

# %%
ds["train"].tokens
# len(ds['test'][1]['text'])

# %%

# tokenize
from transformers import AutoTokenizer

# Load GPT‑2 tokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token


def tokenize(sample):
    out = tok(
        sample["text"],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_overflowing_tokens=True,
    )
    labels = out["input_ids"].copy()
    out["labels"] = labels
    return out


dst = ds.map(
    tokenize,
    num_proc=16,
    load_from_cache_file=True,
    remove_columns=["text"],
)

print(f"number of train samples {len(dst['train'])}")

# %%
dst["train"][3]["input_ids"]

# %%
for i, sample in enumerate(dst["test"]):
    print(sample)
    if i > 20:
        break


# %%
# print(dst["train"][4])

# Set format for PyTorch
dst.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from torch.utils.data import DataLoader

batch_size = 16
train_loader = DataLoader(dst["train"], batch_size=batch_size, shuffle=True)  # type: ignore
val_loader = DataLoader(dst["validation"], batch_size=batch_size)  # type: ignore

# Example: iterate through a batch
for batch in train_loader:
    print(batch)
    break  # Remove this break to iterate through all batches

# Define model
from models import FCModel
import torch
import torch.nn as nn

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Instantiate a simple FC model
vocab_size = len(tok)
model = FCModel(vocab_size=vocab_size, hidden_units=512, num_layers=5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 3
print_every = 10  # Print every n batches
max_batches = 100  # <-- Set your max number of batches per epoch


def sample_from_model(model, tokenizer, device, max_length=50, prompt=None):
    model.eval()
    if prompt is None:
        # Start with a random token
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids
    for _ in range(max_length):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        # Sample from the distribution or use argmax for greedy
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        # Optionally stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated[0].tolist())
    print(f"Sampled text: {text}")


# Sample from the model before training starts
print("Sample from model before training:")
sample_from_model(model, tok, device, max_length=50, prompt="The meaning of life is")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # Predict next token: input[:-1], target: input[1:]
        logits = model(input_ids[:, :-1])
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == max_batches:
            print(
                f"Epoch {epoch+1} | Batch {batch_idx+1}/{max_batches} | Loss: {loss.item():.4f}"
            )
    avg_loss = total_loss / min(len(train_loader), max_batches)
    print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Val loss: {avg_val_loss:.4f}")

    # Sample from the model after each epoch
    print(f"Sample from model after epoch {epoch+1}:")
    sample_from_model(
        model, tok, device, max_length=50, prompt="The meaning of life is"
    )


# %%

# %%
# Define loss function, optimizer, training loop

# %%
# Train transformer on next token prediction
