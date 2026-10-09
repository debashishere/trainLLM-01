# GPT Architecture & Training Design Document

This document serves as the comprehensive conceptual, mathematical, and design reference for the character-level Generative Pre-trained Transformer (GPT) implemented in this repository.

All deep-dive educational notes, ASCII architectural diagrams, tensor shape derivations, and step-by-step training mechanisms migrated from the codebase are organized here by section for continuous learning and revision.

---

## Table of Contents
1. [High-Level Architecture & Forward Pipeline](#1-high-level-architecture--forward-pipeline)
2. [Hyperparameter Configuration (`GPTConfig`)](#2-hyperparameter-configuration-gptconfig)
3. [Token & Positional Embeddings + Weight Tying](#3-token--positional-embeddings--weight-tying)
4. [Causal Multi-Head Self-Attention (`CausalSelfAttention`)](#4-causal-multi-head-self-attention-causalselfattention)
5. [Feed-Forward Network (`MLP`)](#5-feed-forward-network-mlp)
6. [Transformer Block & Residual Streams (`Block`)](#6-transformer-block--residual-streams-block)
7. [Full GPT Model Architecture (`GPT`)](#7-full-gpt-model-architecture-gpt)
8. [Data Loading & Next-Token Batching](#8-data-loading--next-token-batching)
9. [Learning Rate Scheduling (Warmup + Cosine Decay)](#9-learning-rate-scheduling-warmup--cosine-decay)
10. [The Core Training Loop & Backpropagation Lifecycle](#10-the-core-training-loop--backpropagation-lifecycle)
11. [Autoregressive Generation (Greedy vs. Temperature & Top-K Sampling)](#11-autoregressive-generation-greedy-vs-temperature--top-k-sampling)
12. [Common Pitfalls & Bugs Resolved](#12-common-pitfalls--bugs-resolved)

---

## 1. High-Level Architecture & Forward Pipeline

```
INPUT (Token indices: shape (B, T))
 │
 ▼
Token Embedding (wte) + Positional Embedding (wpe)
 │
 ▼ (B, T, n_embd)
 ┌─────────────────────────────────────────┐
 │ Transformer Block 1                     │
 │  ├── LayerNorm 1                        │
 │  ├── Causal Multi-Head Attention + Add  │
 │  ├── LayerNorm 2                        │
 │  └── MLP (GELU) + Add                   │
 └─────────────────────────────────────────┘
 │
 ▼
 ... (repeated n_layer times)
 │
 ▼ (B, T, n_embd)
Final LayerNorm (ln_f)
 │
 ▼
Language Modeling Head (lm_head: n_embd -> vocab_size)
 │
 ▼ (B, T, vocab_size)
OUTPUT (Logits) ──► Cross-Entropy Loss (if targets provided)
```

### The Transformer Block Dataflow
```
INPUT
 │
 ▼
LayerNorm
 │
 ▼
Attention ─────┐
 │             │
 └── + INPUT ◄─┘ (Residual Connection)
 │
 ▼
LayerNorm
 │
 ▼
MLP ───────────┐
 │             │
 └── + INPUT ◄─┘ (Residual Connection)
 │
 ▼
OUTPUT
```

---

## 2. Hyperparameter Configuration (`GPTConfig`)

Reference Implementation: [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py)

```python
@dataclass
class GPTConfig:
    vocab_size: int = 65    # Character-level vocabulary: 65 unique characters in Tiny Shakespeare
    block_size: int = 256   # Max sequence length (context window) - number of tokens model sees at once
    n_embd: int = 384       # Embedding dimension - width of model; all hidden states are vectors of this size
    n_layer: int = 6        # Number of transformer blocks (depth)
    n_head: int = 6         # Number of attention heads (6 heads * 64 head_dim = 384 n_embd)
```

### Conceptual Notes
- **`vocab_size` (65):** In character-level modeling on Tiny Shakespeare, each character (uppercase, lowercase, punctuation, newline, spaces) is mapped to an integer in `[0, 64]`.
- **`block_size` ($T = 256$):** The maximum context window. The model can condition its predictions on up to 256 preceding characters.
- **`n_embd` ($C = 384$):** The channel/embedding dimension. Every character is represented as a 384-dimensional continuous vector.
- **`n_layer` (6):** Depth of the model. 6 identical Transformer blocks are stacked sequentially, refining contextual features step-by-step.
- **`n_head` (6):** Divides the 384 channels into 6 distinct subspaces of dimension $384 / 6 = 64$ (`head_dim`), enabling 6 parallel representation perspectives.

---

## 3. Token & Positional Embeddings + Weight Tying

Reference Implementation: [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py)

### Embeddings: WHAT + WHERE
1. **Token Embeddings (`wte` - Words/Tokens Embedding):**
   - Shape: `(vocab_size, n_embd)` = `(65, 384)`.
   - Answers: *"WHAT token is this?"*
   - For an input batch `idx` of shape `(B, T)`, `wte(idx)` yields a tensor of shape `(B, T, 384)`.

2. **Position Embeddings (`wpe` - Positional Embedding):**
   - Shape: `(block_size, n_embd)` = `(256, 384)`.
   - Answers: *"WHERE in the sequence is this token located?"*
   - Input indices: `pos = torch.arange(0, T, device=idx.device)` representing positions `0, 1, 2, ..., T-1`.
   - `wpe(pos)` yields `(T, 384)` which broadcasts across batch dimension `B`.

3. **Combined Representation:**
   $$\mathbf{x} = \text{tok\_emb} + \text{pos\_emb} \quad \in \mathbb{R}^{B \times T \times 384}$$
   "Respective Token" in position 17 now possesses identity and spatial coordinate information.

### Weight Tying (Weight Sharing)
```python
self.transformer.wte.weight = self.lm_head.weight
```
- **Concept:** The output projection layer `lm_head` (`Linear(n_embd, vocab_size, bias=False)`) maps 384-dimensional hidden representations back to 65 vocabulary logits. Its weight matrix has shape `(65, 384)`.
- **Duality:**
  - `wte` asks: *"What 384-dimensional vector represents character 'a'?"*
  - `lm_head` asks: *"How well does the current 384-dimensional vector match character 'a'?"*
- **Advantages:**
  - Drastically reduces parameter count (avoids storing $65 \times 384 = 24,960$ duplicate weights).
  - Regularizes training and ensures geometric consistency between the input token space and output prediction space.

### Output Logits Generation
```
384-dimensional x
       ↓
┌──────────────────┐
│   W (65 × 384)   │  (lm_head weight matrix)
└──────────────────┘
       ↓ matrix multiply
   65 numbers (logits)
       ↓
65 rows × 384 dimensions = 65 logits
Softmax converts those 65 logits into 65 probabilities over next possible tokens.
```

---

## 4. Causal Multi-Head Self-Attention (`CausalSelfAttention`)

Reference Implementation: [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py)

### The Role of Attention
> **Attention = Tokens talk to other tokens.**  
> It allows each token to gather information from previous tokens in the sequence.

### Linear Projections (Q, K, V)
Input token vector ($C = 384$ numbers):
```
ONE TOKEN
   │
   ▼
384 numbers
   │
   ├────────────── c_attn (Linear: 384 -> 3*384) ──────────────┐
   │                                                           ▼
   │                                                     1152 numbers
   │                                                     /    |    \
   │                                                    Q     K     V
   │                                                  (384) (384) (384)
   │
   │                    Attention Mechanism
   │                           ↓
   │
   └─────────────────────────► 384 numbers
                               │
                               ▼ c_proj (Linear: 384 -> 384)
                               384 numbers
```

- **Query ($Q$):** *"What information am I looking for?"*
- **Key ($K$):** *"What information do I contain?"*
- **Value ($V$):** *"What information do I offer?"*

### Multi-Head Reshaping & Transposition
Condition: `n_embd % n_head == 0` ($384 / 6 = 64 = \text{head\_dim}$).

The dimensions undergo the following transformation:
```
(B, T, 384) ──► view(B, T, n_head, head_dim) ──► (B, T, 6, 64)
            ──► transpose(1, 2)               ──► (B, 6, T, 64)
```

**Why transpose dimensions 1 and 2?**
- Initially: `(B, T, 6, 64)` $\to$ `[Batch, Token, Head, Channel]`.
- Transposed: `(B, 6, T, 64)` $\to$ `[Batch, Head, Token, Channel]`.
- Batch and Head dimensions now act as leading batch dimensions. Matrix multiplication $Q \times K^T$ automatically computes attention maps across the sequence tokens $(T \times T)$ for all 6 heads in parallel without loops.

### Causal Masking & Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

1. **Similarity Matrix ($Q \times K^T$):** Shape `(B, 6, T, T)`. Compares each token's query with every token's key.
2. **Scale Factor ($\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{64}} = \frac{1}{8}$):** Prevents dot products from growing excessively large, avoiding vanishing gradients in softmax.
3. **Causal Mask ($M$):** Masks out all tokens following the current token ($j > i$). In PyTorch, `is_causal=True` in `F.scaled_dot_product_attention` applies this upper-triangular mask automatically using FlashAttention / hardware-fused kernels.
4. **Softmax:** Converts scores to probabilities along the key sequence dimension.
5. **Weighted Value Sum ($A \times V$):** Output shape `(B, 6, T, 64)`.
6. **Recombination:**
   ```python
   y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, 384)
   return self.c_proj(y)
   ```
   `c_proj` mixes the 6 specialized heads back into a coherent 384-dimensional representation.

---

## 5. Feed-Forward Network (`MLP`)

Reference Implementation: [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py)

### The Role of MLP
> **MLP = Each token processes and transforms its own information.**  
> Attention is token communication; MLP is individual token computation and memory retrieval.

### The Complete MLP Transformation
```
Input: (B, T, 384)
  │
  ▼
c_fc: Linear(384 -> 4 * 384 = 1536)  [EXPAND]
  │
  ▼
GELU(approximate='tanh')              [THINK / NONLINEAR GATE]
  │
  ▼
c_proj: Linear(1536 -> 384)          [COMPRESS]
  │
  ▼
Output: (B, T, 384)
```

```
        EXPAND                 THINK (GELU)               COMPRESS
384 ───────────────► 1536 ──────────────────────► 1536 ──────────────► 384
         c_fc                                               c_proj
```

### Why $4\times$ Expansion?
Expanding the channel dimension from 384 to 1536 gives the neural network higher-dimensional space to perform nonlinear feature separation and store factual patterns, before projecting back down to maintain uniform tensor dimensions across blocks.

### Why GELU (Gaussian Error Linear Unit)?
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
- Approximation: $\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi} (x + 0.044715 x^3)\right)\right)$.
- Unlike ReLU which hard-zeros negative values ($x < 0$), GELU provides a smooth, probabilistic gating curve that preserves small negative gradients and improves training stability.

---

## 6. Transformer Block & Residual Streams (`Block`)

Reference Implementation: [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py)

### Pre-LayerNorm Design with Residual Additions
```python
x = x + self.attn(self.ln_1(x))  # Stage 1: Communicate with prior context
x = x + self.mlp(self.ln_2(x))   # Stage 2: Process gathered information
```

### Why Residual Connections (`x + ...`)?
- Residual connections provide an unobstructed gradient highway (identity mapping):
  $$\frac{\partial \mathcal{L}}{\partial x_{\text{in}}} = \frac{\partial \mathcal{L}}{\partial x_{\text{out}}} \cdot \left(1 + \frac{\partial f(x)}{\partial x}\right)$$
- The "+1" term prevents gradients from vanishing during backpropagation through all 6 transformer blocks.
- Conceptually: *"Keep what I already know, and add what attention/MLP just learned."*

### Why Pre-LayerNorm?
In original Attention Is All You Need (Post-LN), normalization occurred after the residual add (`LN(x + Sublayer(x))`). Modern GPT models place LayerNorm *before* the sublayer (`x + Sublayer(LN(x))`), which stabilizes training dynamics and removes the need for warm-up workarounds in deep networks.

---

## 7. Full GPT Model Architecture (`GPT`)

Reference Implementation: [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py)

### Model Structure
```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = nn.LayerNorm(config.n_embd),
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

### Forward Pass Pipeline
1. Check context length: $T \le \text{block\_size}$.
2. Embed tokens: `tok_emb = self.transformer.wte(idx)`.
3. Embed positions: `pos_emb = self.transformer.wpe(torch.arange(0, T, device=idx.device))`.
4. Combine: `x = tok_emb + pos_emb`.
5. Iterate through all $N$ blocks: `for block in self.transformer.h: x = block(x)`.
6. Final normalization: `x = self.transformer.ln_f(x)`.
7. Logits projection: `logits = self.lm_head(x)`.
8. Loss calculation (if `targets` provided):
   ```python
   loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
   ```

---

## 8. Data Loading & Next-Token Batching

Reference Implementation: [`scratchpad/train.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/train.py)

### The Next-Token Prediction Formulation
Language modeling frames text generation as predicting token $t+1$ given tokens $1 \dots t$:
```
Given Context (x):   [ "T", "o", " ", "b", "e" ]
Target Token (y):    [ "o", " ", "b", "e", " " ]
```
$y$ is identical to $x$, shifted forward by exactly 1 character position.

### Batch Generation Logic
```python
def get_batch(split_tokens):
    ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
    x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y
```
- `len(split_tokens) - block_size - 1` prevents indexing beyond array boundaries when extracting the target sequence $y$.
- `torch.stack` combines list of 1D tensors into a 2D batch tensor of shape `(batch_size, block_size)`.
- Data split: 90% training (`tokens[:n]`), 10% validation (`tokens[n:]`).

---

## 9. Learning Rate Scheduling (Warmup + Cosine Decay)

Reference Implementation: [`scratchpad/train.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/train.py)

```
Learning Rate
  ▲
  │           Peak LR (max_lr)
  │             ╭─────────╮
  │            ╱           ╲
  │           ╱             ╲  Cosine Decay
  │          ╱               ╲
  │  Warmup ╱                 ╲
  │        ╱                   ╰────── min_lr
  └───────┴─────────────────────────────► Steps
        step=100                     step=5000
```

### The 3 Phases
1. **Warmup Phase (`step < warmup_steps`):**
   $$\text{lr} = \text{max\_lr} \times \frac{\text{step} + 1}{\text{warmup\_steps}}$$
   Prevents large gradient updates from corrupting randomized initial weights before statistics stabilize.
2. **Decay Phase (`warmup_steps <= step < max_steps`):**
   $$\text{progress} = \frac{\text{step} - \text{warmup\_steps}}{\text{max\_steps} - \text{warmup\_steps}}$$
   $$\text{lr} = \text{min\_lr} + 0.5 \times (\text{max\_lr} - \text{min\_lr}) \times (1 + \cos(\pi \times \text{progress}))$$
   Gradually decreases learning rate following a cosine curve towards `min_lr` ($0.1 \times \text{max\_lr}$).
3. **Post-Schedule (`step >= max_steps`):**
   Maintains `min_lr`.

---

## 10. The Core Training Loop & Backpropagation Lifecycle

Reference Implementation: [`scratchpad/train.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/train.py)

```
                  ┌──────────────────────┐
                  │ Input Batch (x, y)   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Model Forward Pass   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Cross-Entropy Loss   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ optimizer.zero_grad()│ (Clears residual gradients)
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ loss.backward()      │ (Computes dLoss/dWeight)
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ clip_grad_norm_      │ (Prevents gradient explosion)
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ optimizer.step()     │ (Updates model parameters)
                  └──────────────────────┘
```

### Critical Operations Explained
1. `optimizer.zero_grad(set_to_none=True)`: PyTorch accumulates gradients in `.grad` by default. Clearing gradients with `set_to_none=True` frees memory rather than writing zeros.
2. `loss.backward()`: Reverse-mode autodiff through the computation graph.
3. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`: Caps total gradient norm to 1.0. Essential for transformer stability to prevent sudden loss spikes caused by rare extreme batches.
4. Periodic Validation (`model.eval()` + `torch.no_grad()`): Evaluates loss on 20 validation batches to monitor overfitting without storing activation graphs.

---

## 11. Autoregressive Generation (Greedy vs. Temperature & Top-K Sampling)

Reference Implementation: [`scratchpad/generate.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/generate.py)

### Autoregressive Loop
```
Context: [ "T", "o", " ", "b", "e" ]
   │
   ▼
Forward pass through model -> logits for final position
   │
   ▼
Apply Temperature & Top-K filtering
   │
   ▼
Sample next token: " "
   │
   ▼
Append to sequence: [ "T", "o", " ", "b", "e", " " ]
   │
   ▼
Repeat for max_new_tokens
```

### Generation Techniques
1. **Greedy Search (`generate_greedy`):**
   $$t_{\text{next}} = \arg\max(\text{logits})$$
   Deterministic; always chooses highest probability token. Can cause repetitive or degenerate text loops.
2. **Temperature Scaling:**
   $$\text{logits}_{\text{scaled}} = \frac{\text{logits}}{T}$$
   - $T < 1.0$: Sharper probability distribution $\to$ more deterministic, focused, conservative text.
   - $T > 1.0$: Flatter probability distribution $\to$ more diverse, creative, exploratory text.
3. **Top-K Filtering:**
   Only the top $K$ most probable tokens are retained. All other logits are masked to $-\infty$ before softmax, preventing unlikely tail tokens from corrupting text coherence.
4. **Context Window Cropping:**
   If generated sequence length exceeds `block_size` (256), only the most recent 256 tokens are fed to the model: `idx_cond = idx[:, -model.config.block_size:]`.

---

## 12. Common Pitfalls & Bugs Resolved

| Error / Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| `ImportError: cannot import name 'config'` | In `model.py`, invalid import `from torch._dynamo.config import config`. | Removed unnecessary imports (`sympy`, `hashlib`, invalid torch internals); cleaned up standard library imports. |
| `AttributeError: ... has no attribute 'todevice'` | Typo in tensor device transfer method `x.todevice()`. | Corrected to standard PyTorch method `x.to(device)`. |
| Premature `return` before data split | In `load_data()`, `return x, y` was placed at line 52 inside outer function, making `get_train`, `get_val` unreachable. | Indented `return x, y` under inner `get_batch` helper; allowed outer function to return `(get_train, get_val, vocab_size, stoi, itos)`. |
| Model instantiating `config = GPT(config)` | Circular variable shadow bug in `train()` (`config = GPT(config)`). | Correctly instantiated `cfg = GPTConfig(...)` and `gpt_model = GPT(cfg).to(device)`. |
| Parameter count `p.num()` | Non-existent method `p.num()`. | Changed to `p.numel()`. |
| Attribute typo `self.transformer.lm_f` | Final LayerNorm was named `ln_f` during init, but called as `lm_f` in `forward()`. | Fixed reference to `self.transformer.ln_f(x)`. |
| Unbound variable `targets` vs `target` | Parameter in `GPT.forward(self, idx, target=None)` was named `target`, but loss computed using `targets`. | Standardized argument name to `targets`. |
| Generation function missing in `train.py` | `train.py` invoked `generate(...)` but never defined or imported it. | Implemented standalone, robust `generate(model, prompt, stoi, itos, ...)` supporting temperature and top-k sampling. |
| Checkpoint dictionary key syntax error | In final save: `{ step: max_steps, ..., itos: itos }` treated variables as keys. | Quoted dictionary keys: `{"step": max_steps, ..., "itos": itos}`. |
| Missing `json` import | `json.dump(...)` called in `train.py` without `import json`. | Added `import json` at module level. |
