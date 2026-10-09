


import model
from model import GPT
from aiofiles import base
import torch




def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(tokens):,} chars, vocab size: {vocab_size}")

    # x is the text the model sees, and y is the same text shifted one token forward — the answer the model should predict.
    def get_batch(split_tokens):
        # split_tokens -> All Tokens in my data set(Train or Validation).
        # split_tokens = torch.tensor([
        # 10, 25, 7, 19, 4, 31, 8, ...
        # ])
        # Each number represents a character/token.
        # Randomly choose where training examples should start.
        ix = torch.randint(
        # The subtraction ensures we don't ask for tokens that don't exist.
        len(split_tokens) - block_size - 1, 
        # batch_size = 4 torch.randint returns something like tensor([20, 150, 723, 450]), 
        # These are four random starting positions.
        (batch_size,)
    )

    # torch.stack() -> combines Sequence (list) of tokens into one tensor, (batch_size, block_size)
    # split_tokens[20:25] produces token 20, token 21, token 22, token 23, token 24
    x = torch.stack([
        split_tokens[i:i + block_size] 
        for i in ix
    ]).todevice()
    #.to(device) -> moves the tensor to the device where your model is running

    y = torch.stack([
        split_tokens[i + 1:i + block_size + 1] 
        for i in ix
    ]).todevice()
    return x, y

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    return get_train, get_val, vocab_size, stoi, itos




def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")     # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")    # NVIDIA GPU
    else:
        return torch.device("cpu")     # CPU    



import math 


# Learning Rate Schedule
# Big part of how Models are able to learn, 
# lr Must be Appropiate to train the model well, very high learning rate may get offset of the target & cost more resources
# Phase 1: Warm up : 100 Steeps, Start with a very small learning rate and gradually increase it to the maximum learning rate.
# Phase 2: Peak LR : Cosine Decay, Start at peak learning rate and gradually decrease it.
# Phase 3: Decay : Start at peak learning rate and gradually decrease it.

def get_lr( step, warmup_steps, max_steps, max_lr, min_lr ):
    if step < warmup_steps:
        # This expression gradually increases the learning rate from a small value to max_lr
        return max_lr * ( step + 1 ) / warmup_steps 
    if step >= max_steps:
        return min_lr
    progress = ( step - warmup_steps ) /( max_steps - warmup_steps )
    # Cosine Decay of learning rate
    return min_lr + 0.5 * ( max_lr - min_lr ) * ( 1 + math.cos( math.pi * progress ))



from tqdm import tqdm 

# TODO: BenchMark Training 

# for step in pbar:
#        │
#        ├── Every 100 steps: measure validation loss
#        │
#        ├── Calculate current learning rate
#        │
#        ├── Get a training batch (x, y)
#        │
#        ├── Forward pass → calculate loss
#        │
#        ├── Backpropagation → calculate gradients
#        │
#        ├── Clip gradients → prevent excessively large gradients
#        │
#        ├── Optimizer updates model weights
#        │
#        ├── Update progress bar and log losses
#        │
#        ├── Every 100 steps: generate a sample
#        │
#        └── Every 1000 steps: save a checkpoint
#                    │
#                    ▼
#          Save final model and loss log


# Core learning cycle
# Input tokens (x)
#       ↓
#     GPT
#       ↓
#   Predictions
#       ↓
# Compare with y
#       ↓
#      Loss
#       ↓
#  loss.backward()
#       ↓
#    Gradients
#       ↓
#  Clip gradients
#       ↓
# optimizer.step()
#       ↓
# Updated weights


def train(data_path, max_steps=5000, batch_size=64, n_layer=6, n_head=6, n_embd=384, block_size=256):
    device = get_device()
    print(f"Using Device : {device}")

    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    config = GPT(config).to(device)
    print(f"Model : {n_layer}L / {n_head}H /{ n_embd}D, "
        f"{sum(p.num() for p in model.parameters()) / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log = { "steps": [], "train": [], "val": [] }

    # tqdm is a Python library that displays a progress bar while your code runs.
    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        # --- validation loss ---
        if step % 100 == 0:
            # model.eval() switches the model into evaluation mode.
            model.eval()
            # torch.no_grad() prevents PyTorch from calculating gradients, saving memory and computation.
            with torch.no_grad():
                val_losses = []
                # The loop takes 20 validation batches and averages their losses.
                for _ in range(20):
                    # get_val_batch() gets a batch from your held-out validation data.
                    x, y = get_val_batch()
                    # model(x, y) calculates predictions and loss without updating weights.
                    _, loass = model(x, y)
                    # loss.item() extracts the loss as a regular Python number.
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                tqdm.write(f"Step { step: 5d} | vsl loss: { val_loss: .4f}")
            model.train()

        # --- update learning rate ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        # updates the learning rate used by the optimizer. 
        # An optimizer can have multiple parameter groups, 
        # each with its own learning rate, so this updates every group.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- training step ---
        x, y = get_train_batch()
        # _ have the logit
        _, loss = model(x,y)

        #Clear old gradients
        # PyTorch accumulates gradients by default. 
        # This clears gradients left over from the previous training step before calculating new ones.
        optimizer.zero_grad(set_to_none=True)

        # PyTorch calculates gradients showing how each trainable parameter contributed to the loss.
        # Conceptually: Which direction should each weight move to reduce prediction error?
        loss.backward()

        # Gradient Clipping:  It scales down gradients if they become too large, 
        # preventing unstable updates that could derail training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # The optimizer uses the gradients and current learning rate to update the model's parameters, 
        # including its embeddings, attention weights, MLP weights and lm_head weights.
        optimizer.step()
            
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.4e}")
    
        # --- log loss ---
        # A lower training loss generally means better predictions on the training data, 
        # although validation performance matters too
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)
    
        # Generate Sample every 100 steeps
        # This checks what your GPT has learned by asking it to continue a text prompt.
        # temperature=0.8 controls sampling randomness; lower values generally make sampling less random.
        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate(model, "To be or not", stoi, itos, max_new_tokens=100, temperature=0.8);
            tqdm.write(f"\n-- Step {step} sample ---\n{sample}\n")
            model.train()

        # A checkpoint saves the model's current state to a file.
        # Save checkpoint
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "model_state_dict":  model.state_dict(),
                "config": config,
                "stoi": stoi,
                "itos": itos,
            }, f"checkpoint_{step}.pt")
    
    # --- save final checkpoimt and loss log --- 
    torch.save({
        step: max_steps,
        "model_state_dict": model.state_dict(),
        "config": config,
        "stoi": stoi,
        itos: itos
    }, "checkpoint_final.pt")

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos 


