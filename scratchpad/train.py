"""
Training Script for Character-Level GPT on Tiny Shakespeare.

Pipeline:
1. Load dataset, derive character vocabulary and train/val splits
2. Initialize GPT model on device (MPS / CUDA / CPU)
3. Train with AdamW, cosine decay learning rate schedule with warmup, and gradient clipping
4. Periodic validation loss calculation and sample text generation
5. Checkpointing and loss history logging

For full architectural notes and training mechanics, see design-document.md Sections 8-10.
"""

import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

from model import GPT, GPTConfig
from generate import generate


def load_data(filepath: str | Path, block_size: int, batch_size: int, device: torch.device):
    """
    Loads text data, creates character mappings, and returns batch generator functions.
    
    See design-document.md Section 8 for tensor stacking and boundary arithmetic.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")

    def get_batch(split_tokens: torch.Tensor):
        # Randomly choose starting indices for each sequence in the batch
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        # Stack sequences into (batch_size, block_size) and move to target device
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        # Target tokens are shifted forward by 1
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    # 90% train, 10% validation split
    n = int(0.9 * len(tokens))
    train_tokens = tokens[:n]
    val_tokens = tokens[n:]

    get_train = lambda: get_batch(train_tokens)
    get_val = lambda: get_batch(val_tokens)

    return get_train, get_val, vocab_size, stoi, itos


def get_device() -> torch.device:
    """Detects available hardware accelerator (Apple Silicon MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Computes learning rate using linear warmup followed by cosine decay.
    
    See design-document.md Section 9 for schedule derivation.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(
    data_path: str = "./data/shakespeare.txt",
    max_steps: int = 5000,
    batch_size: int = 64,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    block_size: int = 256,
):
    """
    Main training execution loop.
    
    See design-document.md Section 10 for step-by-step training lifecycle.
    """
    device = get_device()
    print(f"Using Device: {device}")

    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model = GPT(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_layer}L / {n_head}H / {n_embd}D, {param_count / 1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log = {"steps": [], "train": [], "val": []}
    val_loss = float("nan")

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        # --- Validation Loss ---
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x_val, y_val = get_val_batch()
                    _, val_step_loss = model(x_val, y_val)
                    if val_step_loss is not None:
                        val_losses.append(val_step_loss.item())
                val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
                tqdm.write(f"Step {step:5d} | Val loss: {val_loss:.4f}")
            model.train()

        # --- Update Learning Rate ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Training Step ---
        x, y = get_train_batch()
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.4e}")

        # --- Logging ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)

        # --- Sample Generation ---
        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate(
                model,
                prompt="To be or not",
                stoi=stoi,
                itos=itos,
                max_new_tokens=100,
                temperature=0.8,
                device=device,
            )
            tqdm.write(f"\n--- Step {step} Sample ---\n{sample}\n")
            model.train()

        # --- Checkpoint Saving ---
        if step > 0 and step % 1000 == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stoi": stoi,
                    "itos": itos,
                },
                f"checkpoint_{step}.pt",
            )

    # --- Save Final Checkpoint and Loss Log ---
    torch.save(
        {
            "step": max_steps,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        "checkpoint_final.pt",
    )

    with open("loss_log.json", "w", encoding="utf-8") as f:
        json.dump(loss_log, f, indent=2)

    return model, stoi, itos


if __name__ == "__main__":
    train()
