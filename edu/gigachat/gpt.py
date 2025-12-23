# Implement a modern GPT model
# B (Batch Size): The number of independent sequences you are processing at once.
# T (Time / Sequence Length): The number of tokens in your sequence (e.g., the length of the sentence).
# C (Channels / Embedding Size): The total width of the model's hidden state (e.g., 4096 in Llama 2 7B).
# H (Heads): The number of attention heads.
# D (Dimension of Head): The size of a single attention head.

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def apply_rotary_emb(x, cos, sin):
    """Applies rotary embeddings to the input tensor."""
    def rotate_half(x):
        """Rotates half the hidden dims of the input tensor."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotate_half(x) * sin)

def rms_norm(x, epsilon=1e-6):
    """Applies native RMSNorm to the input tensor (requires PyTorch 2.4+)."""
    return F.rms_norm(x, (x.size(-1),), eps=epsilon)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK rotary embedding (RoPE)
        # apply_rotary_emb expects (B, T, H, D)
        q = apply_rotary_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)

        # QK norm
        # F.rms_norm expects the last dim to be the norm dim
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # Native GQA support
        enable_gqa = self.n_head != self.n_kv_head
        Tq = q.size(2)  # Number of new queries
        Tk = k.size(2)  # Total keys (cached + new)
        if kv_cache is None or Tq == Tk:
            # Training or prefill: standard causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # Single token decode: attend to all cached keys (no masking needed)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x)**2
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(rms_norm(x), cos, sin, kv_cache=kv_cache)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between embedding and output projection
        self.lm_head.weight = self.transformer["wte"].weight
        cos, sin = self._precompute_freqs(head_dim, config.sequence_len, device=device)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_freqs(self, head_dim, max_seq_len, theta=10000, device=None):
        inv_freq = 1. / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos().view(1, max_seq_len, 1, -1)
        sin = freqs.sin().view(1, max_seq_len, 1, -1)
        return cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Standard initialization for Linear layers
            # We use 0.02 as a standard deviation (approx 1 / sqrt(768))
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            # Embeddings use the same standard deviation
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_initialization(self):
        """
        Apply special scaling to residual projections (c_proj) 
        to account for the accumulation of variance in deep networks.
        """
        self.apply(self._init_weights)
        # Scale residual projections
        # c_proj is the output of Attention and MLP that gets added back to the residual stream
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                # Scale by 1/sqrt(2 * n_layer)
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.config.n_layer)**0.5)

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, x, targets=None, kv_cache=None):
        B, T = x.shape
        x = self.transformer["wte"](x)
        x = rms_norm(x)

        # Offset RoPE based on cache position
        # cos/sin are (1, T, 1, D), need to handle broadcasting or slicing
        # Here we slice to T in case we have a shorter sequence during inference
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos = self.cos[:, T0:T0+T, :, :]
        sin = self.sin[:, T0:T0+T, :, :]

        for block in self.transformer["h"]:
            x = block(x, cos, sin, kv_cache=kv_cache)
        x = rms_norm(x)
        x = self.lm_head(x) # Logits
        
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, self.config.vocab_size), targets.view(-1))
            return x, loss
        return x

    @torch.inference_mode()
    def generate(self, x, max_new_tokens=1024, temperature=1.0, top_k=0, seed=42):
        torch.manual_seed(seed)
        device = self.get_device()
        x = torch.tensor([x], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_new_tokens):
            # Handle context overflow: only use last sequence_len tokens
            x_cond = x if x.size(1) <= self.config.sequence_len else x[:, -self.config.sequence_len:]
            logits = self.forward(x_cond) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat((x, next_ids), dim=1)
            token = next_ids.item()
            yield token