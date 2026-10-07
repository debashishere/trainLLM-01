
# Import Data Class
from torch.nn import quantized
from dataclasses import dataclass

# Declear Data Class 
@dataclass
class GPTConfig:
    vocab_size : int = 65                                   # character-levl: 65 unique chars in Shakespare
    block_size : int = 256                                  # max sequency length (context window) , Number of Token Model Can see at Once
    n_layer: int = 6                                        # number of transformer blocks
    n_head: int = 6                                         # number of attention heads
    n_embd: int = 384                                       # embbeding dimention, width of the model — every hidden state is a vector of this size.


    

# ---------------------------------Embeddings----------------------
import torch 
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer  = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),            # token embdeddings
                wpe = nn.Embedding(config.block_size, config.n_embd),            # position embeddings
                # Cofigure each layer
                h = nn.ModuleList(
                    [
                        Block(config) for _ in range(config.n_layer)
                        ]
                    ),
                # Normalization the embbedings
                ln_f = nn.LayerNorm(config.n_embd),

        )
        )
        # To find the head we need to know the Embbeding vector's size (n_embd) and Vocab Size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight typing: the output projection shares weights with the token embeddings
        self.transformer.wte.weight = self.lm_head.weight


# -----------------------Forword Pass---------------- 
#WHAT + WHERE → send that into the Transformer.
def forward(self, idx, target=None):
    # B = 32    → number of sequences
    # T = 256   → number of tokens in each sequence
    # B, T = idx.shape 
    B, T = idx.shape 

    # Create a number for every token position in my current sequence ( data in the context ).
    # device=idx.device = "Put these position numbers on the same device as my input."
    pos = torch.arange(0, T, device=idx.device)

    # All Token Embbedings -> 384 numbers
    tok_emb = self.transformer.wte(idx)        # (B, T, n_emdb)
    # Total Position Embbedings -> 384 numbers
    pos_emb = self.transformer.wpe(pos)        # (T, n_embd)

    # Final Representation ( "RespectiveToken" in position 17)
    x = tok_emb + pos_emb                     
    
    #"Take the token representations, pass them through each of the 6 Transformer blocks sequentially, 
    # allowing each block to refine the contextual understanding, and carry the result into the next block."
    for block in self.transformer.h:
        x = block(x)
    
    #“Clean up the understanding → convert understanding into scores for every possible token.”
    #ln_f means final LayerNorm.
    x = self.transformer.lm_f(x)
    #Translate the 384-dimensional internal representation into 65 prediction scores (logits).
    logits = self.lm_head(x)


    loss = None 
    if target is not None:
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
    return logits, loss


#-----------------------------------------Self-Attention---------------------------------
# This is the mechanism that lets each token attend to (look at) every previous token in the sequence.




class CausalSelfAttention(nn.Module):
    # This creates a PyTorch neural-network component called Causal Self-Attention.
    # Self-Attention : A token looks at other tokens in the same sequence.
    def __init__(self, config):
        super().__init__()
        
        #“Can I divide the embedding dimensions equally among all attention heads?”
        # assert means “This condition MUST be true. If it isn't, stop the program.”
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V projections, Q(Query : “What information am I looking for?”),
        # K(Key: “What information do I contain?”),
        # V(Value: “What information do I offer?”).    
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       
        
        #“Each token has 384 numbers, and I want to use 6 attention heads.”
        self.n_head = config.n_head
        self.n_embd = config.n_embd




    def forward(self, x):
        #“Take every token's 384-dimensional representation, 
        # transform it into 1152 numbers, 
        # then separate those numbers into Query, Key, and Value — each 384-dimensional.”


        # B = Batch
        #    How many sequences?
        # T = Tokens
        #    How many token positions per sequence?
        # C = Channels / embedding dimensions
        #    How many numbers represent each token? ->384
        B, T, C = x.shape


        # Your x coming into attention looks like:
        # c_attn -> Input x (32, 256, 384)
        # c_atten -> Prduces qkv (32, 256, 1152)
        qkv = self.c_attn(x)


        # .split() mean “Cut this tensor into pieces of size 384 along dimension 2.”
        q, k, v = qkv.split(self.n_embd, dim=2)
        # our tensor qkv.shape = (32, 256, 1152)
        # dim 0 → 32
        # dim 1 → 256
        # dim 2 → 1152
        #qkv.split(384, dim=2) Means “Keep dimensions 0 and 1 as they are, but cut the last dimension into chunks of 384.”
        # 1152
        #  ↓
        # 384 + 384 + 384

        # q.shape = (32, 256, 384)
        # k.shape = (32, 256, 384)
        # v.shape = (32, 256, 384)

        # Imagine the last dimension of qkv as one long box:
        #   qkv:
        #   |------------- 1152 -------------|
        #   |------384------|------384------|------384------|
        #         Q                 K                V

        # After:
        # q → (B,T,384)
        # k → (B,T,384)
        # v → (B,T,384)

        # q → (32,256,384)
        # k → (32,256,384)
        # v → (32,256,384)

        
        head_dim = C // self.n_head
        # C = 384
        # n_head = 6 
        # 384 // 6 = 64  -> Each attention head gets 64 dimensions.
        # 64 is called head_dim
        # 384 dimensions
        # Head 1 → 64
        # Head 2 → 64
        # Head 3 → 64
        # Head 4 → 64
        # Head 5 → 64
        # Head 6 → 64


        # The 384 dimensions aren't one big group anymore. Treat them as 6 groups of 64.
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        #(B, T, 384)

            # ↓ view

        # (B, T, 6, 64)
        # Numerically (32, 256, 384) becomes (32, 256, 6, 64)

        # ImportantS
        # view() isn't changing the actual values.
        # It's essentially changing how we organize/interpret the dimensions.
        #Think of 384 books:
        # 384 books
        # becoming:
        # 6 shelves × 64 books
        
        # Why do we want (B, T, 6, 64)?
        # Because we want to perform attention independently for each head.
        # Because now we can run 6 attention calculations in parallel. 
        # 6 heads × 64 dimensions = 384

        # Currently

        # Means
        # Batch
        #  → Token positions
        #  → Head 1 → 64 dimensions
        #  → Head 2 → 64 dimensions
        #  → Head 3 → 64 dimensions
        #  → Head 4 → 64 dimensions
        #  → Head 5 → 64 dimensions
        #  → Head 6 → 64 dimensions

        #But there's one more problem.
        #For attention calculations, we want the head dimension before the token dimension.
        #That's why we do:
        #.transpose(1, 2)

        # What does .transpose(1, 2) do?
        # Currently (B, T, n_head, head_dim) or (32, 256, 6, 64)
        # Dimension positions:
        #   dim 0 → B = 32
        #   dim 1 → T = 256
        #   dim 2 → heads = 6
        #   dim 3 → head_dim = 64
        # Then .transpose(1, 2) swaps dimensions 1 and 2 -> (B, heads, T, head_dim) = (32,6,256,64)

        # Why swap them?
        # Because now we can easily perform matrix multiplications:
        # (Batch, num_heads, seq_len, head_dim)
        # This layout is standard for parallel attention calculations.
        # Because attention wants to work like this:
        # Batch
        #  ↓
        # Heads
        #  ↓
        # Token positions
        #  ↓
        # Head dimensions


        # Same thing happens to K and V
        # "I want to compare every token's Query with every other token's Key."
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)


        # Before:
        # Q → (B,T,384)
        # K → (B,T,384)
        # V → (B,T,384)
        # After:
        # Q → (B,6,T,64)
        # K → (B,6,T,64)
        # V → (B,6,T,64)

        # Q → (32,6,256,64)
        # K → (32,6,256,64)
        # V → (32,6,256,64)

        # The whole transformation
        # This is the important part to memorize:
        #      (B, T, 384)
        #           │
        #           │ view
        #           ↓
        #      (B, T, 6, 64)
        #           │
        #           │ transpose(1,2)
        #           ↓
        #      (B, 6, T, 64)

        # Same for K and V.3


        # This allows the model to have 6 different attention perspectives.
        #Imagine:
        # Head 1 → relationships between nearby words
        # Head 2 → grammatical relationships
        # Head 3 → subject/object relationships
        # Head 4 → longer-range relationships
        # Head 5 → semantic relationships
        # Head 6 → other learned patterns
        # ⚠️ These are not hard-coded roles. The model learns what each head should focus on during training.


        # attention with causal mask (each token can only attend to previous tokens)
        # is_causal = Mask all other charaters then the current One and all previuous one's, It prevents GPT from cheating during training.
        # For every token, which previous tokens should I pay attention to, and how much information should I take from them?
        # 1. Q × Kᵀ
        # 2.↓
        #    attention scores
        # 3.↓
        #    causal mask
        # 4.↓
        # softmax
        # ↓
        # attention weights
        # ↓
        # weights × V
        # ↓
        # y
        #Attention does not change the overall dimensions.
        #It changes the information inside the vectors.
        #Each token still has 64 dimensions per head
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # current shape of y (B, heads, T, head_dim)
        # We required : (B, T, heads, head_dim) , so we are applying transpose() function to swap the dimensions 1 and 2.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


# Attention = tokens talk to other tokens.
# MLP = each token processes and transforms its own information.
#                  x
#                  │
#        ┌────────▼────────┐
#        │    Attention    │
#        │ "Who should I   │
#        │    listen to?"  │
#        └────────┬────────┘
#                 │
#          Residual Add
#                 │
#        ┌────────▼────────┐
#        │      MLP        │
#        │ "What should I  │
#        │    understand?" │
#        └────────┬────────┘
#                 │
#          Residual Add
#                 │
#                 ▼
#              output

# So, Attention → communication
# MLP → processing

# The MLP temporarily gives the token more room to think
# The complete MLP journey
# Input
# (B, T, 384)
#      │
#      ▼
#┌──────────────┐
#│ Linear       │
#│ 384 → 1536   │
#└──────┬───────┘
#       │
#       ▼
#(B,T,1536)
#       │
#       ▼
#┌──────────────┐
#│ GELU         │
#│ nonlinear    │
#└──────┬───────┘
#       │
#       ▼
#(B,T,1536)
#       │
#       ▼
#┌──────────────┐
#│ Linear       │
#│ 1536 → 384   │
#└──────┬───────┘
#       │
#       ▼
#(B,T,384)

# x
# │
# ├── Linear(384 → 1536)
# │
# ├── GELU
# │
# └── Linear(1536 → 384)
# │
# ▼
# output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

       # Linear transformation
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
       # The 4× expansion is a common Transformer design choice.
       # It's not because the token suddenly has 4 times more information in some literal sense. 
       # It gives the neural network more dimensions in which to perform nonlinear computation.
       # creates learnable parameters roughly equivalent to 4 * 384 = 1536 nurons
       # Every one of the 1536 output neurons looks at the 384 input values.
       # output = input * weights + bias
       # c_fc means roughly: Take the token's 384-dimensional representation and 
       # transform it into a richer 1536-dimensional representation.

        # Use the GELU activation, with a faster tanh-based approximation.
        # GELU(x) = x · Φ(x), Φ(x) is the standard normal cumulative distribution function.
        self.gelu = nn.GELU(approximate='tanh')
        # GELU allows the network to learn much more complicated patterns.
        # GELU This is the nonlinear part of the MLP.
        # Without GELU: the two Linear operations could effectively collapse into another linear transformation.
        # GELU breaks that simple linear relationship.
        # Think of it as a smart gate:
        # 1536 signals
        #   │
        #   ▼
        #  GELU
        #   │
        #   ├── strongly useful → pass strongly
        #   ├── somewhat useful → pass partially
        #   └── less useful → suppress
        
        
        
        # Transformation go back to 384
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Rest of the Transformer expects the model representation to remain: (B, T, 384)
        # So the MLP temporarily expands the representation:
        # 384
        # ↓
        # 1536
        # ↓
        # 384
        # This allows the MLP to do complex processing while maintaining a consistent interface between Transformer blocks.
        # The 384 → 1536 → 384 transformation happens independently for every token.

        # c_fc = expand the brain
        # GELU = nonlinear thinking/gating
        # c_proj = compress back to model size

        # So your MLP is essentially
        #         EXPAND              THINK          COMPRESS
        # 384 ──────────────► 1536 ──────────► 1536 ──────────► 384
        # c_fc              GELU              c_proj

        # And that's why the MLP is often described as the Transformer's position-wise feed-forward network.

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


