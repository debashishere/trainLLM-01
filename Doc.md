# LLM Training Workshop: Building GPT from Scratch

> **Reference Document:** [LLM Workshop Google Doc](https://docs.google.com/document/d/1YptmAx7JVCcsHyW1GKVmd-xrbTPrYp9gObQ_bJ2GqtA/edit?tab=t.0)  
> **Project Goal:** Build, understand, and train a Generative Pre-trained Transformer (GPT) language model from scratch on Apple Silicon (MacBook) using PyTorch.

---

## 1. Project Overview

This repository contains an end-to-end implementation of an autoregressive GPT-style decoder-only transformer. It is designed to run efficiently on Apple Silicon (M-series MacBooks) using PyTorch's MPS (Metal Performance Shaders) acceleration.

The model is trained on the character-level **Tiny Shakespeare** dataset (~1.1 MB text), allowing rapid iteration and complete transparency into every layer of the architecture—from raw character tokenization to causal multi-head self-attention and cross-entropy loss computation.

---

## 2. Directory Structure

```text
trainLLM-01/
├── Doc.md                     # Comprehensive project documentation (this file)
├── .gitignore                 # Git ignore rules for venv, cache, and checkpoints
└── scratchpad/
    ├── pyproject.toml         # Project dependencies and configuration
    ├── uv.lock                # Deterministic dependency lockfile
    ├── model.py               # GPT model architecture & attention implementation
    ├── train.py               # Tokenization and training script
    └── data/
        └── shakespeare.txt    # Training corpus (Tiny Shakespeare)
```

---

## 3. Environment & Prerequisites

### Tooling
- **Python:** `>= 3.12`
- **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (or standard Python `venv` + `pip`)
- **Hardware Acceleration:** Apple Silicon MPS backend (`torch.device("mps")`) or CPU/CUDA

### Core Dependencies
Defined in [`pyproject.toml`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/pyproject.toml):
- `torch >= 2.8.0` — Deep learning primitives and automatic differentiation.
- `tiktoken >= 0.12.0` — Fast BPE tokenizer (for future subword tokenization).
- `numpy >= 2.0.2` — Numerical arrays and batch indexing.
- `tqdm >= 4.67.3` — Training progress indicators.
- `datasets >= 4.5.0` & `huggingface-hub >= 1.8.0` — Dataset fetching and management.

---

## 4. Setup and Quickstart

### Step 1: Navigate to the Project Workspace
```bash
cd scratchpad
```

### Step 2: Set up Virtual Environment
Using `uv`:
```bash
uv sync
```
Or using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pyproject.toml
```

### Step 3: Run the Training / Tokenization Script
```bash
# With active venv
python train.py

# Or via uv directly
uv run python train.py
```

---

## 5. Architecture & Model Specification

The model implementation in [`scratchpad/model.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/model.py) follows the decoder-only transformer architecture (similar to GPT-2 / NanoGPT).

### Hyperparameter Configuration (`GPTConfig`)

```python
@dataclass
class GPTConfig:
    vocab_size : int = 65      # 65 unique characters in Tiny Shakespeare
    block_size : int = 256     # Maximum context window (sequence length T)
    n_layer    : int = 6       # Number of stacked transformer blocks
    n_head     : int = 6       # Number of causal self-attention heads
    n_embd     : int = 384     # Embedding & hidden state dimension
```

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | `65` | Character-level vocabulary derived from Shakespeare. |
| `block_size` | `256` | Maximum token sequence length the model attends to at once ($T$). |
| `n_layer` | `6` | Depth of the network (transformer blocks $h_1 \dots h_6$). |
| `n_head` | `6` | Attention heads; head dimension $d_{\text{head}} = 384 / 6 = 64$. |
| `n_embd` | `384` | Channel dimension ($C$) for hidden activations and embeddings. |

---

## 6. Deep Dive: Model Components

### 6.1 Token & Position Embeddings
The model maps discrete token IDs into dense continuous representations:
- **`wte` (`nn.Embedding(vocab_size, n_embd)`)**: Token identity embedding. Maps token index to a 384-dimensional vector.
- **`wpe` (`nn.Embedding(block_size, n_embd)`)**: Learned positional embedding. Injects spatial sequence order $pos \in [0, T-1]$.
- **Embedding Fusion:**
  $$\mathbf{x} = \text{wte}(\text{idx}) + \text{wpe}(\text{pos})$$

### 6.2 Weight Tying
To save parameters and improve training efficiency, the output linear projection layer shares weights with the token embedding table:
```python
self.transformer.wte.weight = self.lm_head.weight
```

### 6.3 Causal Multi-Head Self-Attention (`CausalSelfAttention`)
Attention enables tokens to gather context from preceding tokens in the sequence:

1. **Q, K, V Linear Projection:**
   Project input $\mathbf{x} \in \mathbb{R}^{B \times T \times C}$ into Query, Key, and Value tensors simultaneously:
   $$\text{qkv} = \text{Linear}_{384 \to 1152}(\mathbf{x}) \implies \mathbf{q}, \mathbf{k}, \mathbf{v} \in \mathbb{R}^{B \times T \times 384}$$

2. **Multi-Head Tensor Reshape & Transpose:**
   Split the 384 dimensions across 6 heads ($6 \times 64 = 384$):
   $$(B, T, 384) \xrightarrow{\text{view}} (B, T, 6, 64) \xrightarrow{\text{transpose(1, 2)}} (B, 6, T, 64)$$

3. **Scaled Dot-Product Attention with Causal Mask:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   Where $M$ is the upper-triangular causal mask ($M_{ij} = -\infty$ for $j > i$) to prevent tokens from peeking into the future.
   Executed efficiently using PyTorch's native scaled dot-product attention:
   ```python
   y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
   ```

4. **Output Projection:**
   Reassemble heads:
   $$(B, 6, T, 64) \xrightarrow{\text{transpose}} (B, T, 6, 64) \xrightarrow{\text{view}} (B, T, 384)$$
   Pass through `c_proj = nn.Linear(384, 384)`.

---

## 7. Data Pipeline & Tokenization

Implemented in [`scratchpad/train.py`](file:///Users/debashisroy/Documents/trainLLM-01/scratchpad/train.py):

- **Corpus:** 1,115,394 characters from Shakespeare's works.
- **Vocabulary:** 65 distinct characters:
  ```text
  \n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
  ```
- **Encoder/Decoder:**
  - `stoi`: Character $\to$ Integer ID (`dict[str, int]`)
  - `itos`: Integer ID $\to$ Character (`dict[int, str]`)
  - `encode(str) -> list[int]`
  - `decode(list[int]) -> str`

---

## 8. Next Milestones & Roadmap

- [x] Character-level tokenizer & vocab mapping
- [x] GPT configuration dataclass (`GPTConfig`)
- [x] Embeddings (`wte`, `wpe`) and weight tying
- [x] Causal multi-head self-attention module
- [ ] Transformer `Block` class (LayerNorm $\to$ Attention $\to$ LayerNorm $\to$ MLP with residual connections)
- [ ] Training batch generator ($B=32, T=256$) with train/validation split (90/10)
- [ ] Optimizer configuration (AdamW with decoupled weight decay)
- [ ] Training loop with loss tracking and MPS GPU backend
- [ ] Autoregressive sampling / text generation loop
