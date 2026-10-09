"""
Text Generation Utilities for GPT.

Supports:
- Greedy search generation
- Temperature-scaled probabilistic sampling with Top-K filtering

For conceptual notes on autoregressive sampling, see design-document.md Section 11.
"""

import torch
import torch.nn.functional as F
from model import GPT, GPTConfig


def generate_greedy(model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """
    Generate tokens using deterministic greedy search (argmax).
    
    Args:
        model: GPT model.
        idx: Starting token indices tensor of shape (B, T).
        max_new_tokens: Number of tokens to append.
        
    Returns:
        idx: Tensor of shape (B, T + max_new_tokens).
    """
    for _ in range(max_new_tokens):
        # Crop context to block_size if needed
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        # Focus only on the last time step
        logits = logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    device: torch.device | None = None,
) -> str:
    """
    Generate text from a prompt using temperature and top-k sampling.
    
    Args:
        model: Trained GPT model.
        prompt: Initial string prompt.
        stoi: Character-to-integer mapping dictionary.
        itos: Integer-to-character mapping dictionary.
        max_new_tokens: Number of characters to generate.
        temperature: Sampling temperature (lower = more deterministic, higher = more creative).
        top_k: If set, retain only top-k most likely tokens for sampling.
        device: Device to place prompt tensor on (defaults to model parameter device).
        
    Returns:
        Generated text string including prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt characters to token IDs
    encoded_prompt = [stoi[c] for c in prompt if c in stoi]
    if not encoded_prompt:
        encoded_prompt = [0]
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop context window to block_size
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    generated_ids = idx[0].tolist()
    return "".join(itos.get(i, "") for i in generated_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint file (e.g. checkpoint_final.pt)")
    parser.add_argument("--prompt", default="To be or not", help="Starting text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Only sample from top-k most likely tokens")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output = generate(
        model,
        args.prompt,
        stoi,
        itos,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(output)