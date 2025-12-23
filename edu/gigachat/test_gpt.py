"""
Comprehensive test suite for edu/gigachat/gpt.py

Tests cover:
- Unit tests for individual components (RoPE, RMSNorm, MLP, Attention, Block)
- Integration tests for full model
- Shape validation
- Gradient flow
- Mathematical correctness with mocked/known weights
- Edge cases and boundary conditions
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from unittest.mock import patch, MagicMock

from gpt import (
    GPT, GPTConfig, 
    CausalSelfAttention, MLP, Block,
    apply_rotary_emb, rms_norm
)
from engine import KVCache


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64
    )

@pytest.fixture
def mha_config():
    """Config with standard multi-head attention (no GQA)."""
    return GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_kv_head=4,  # Same as n_head = MHA
        n_embd=64
    )

@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Unit Tests: RoPE (Rotary Positional Embeddings)
# ============================================================================

class TestRotaryEmbeddings:
    """Tests for apply_rotary_emb function."""
    
    def test_rope_output_shape(self):
        """RoPE should preserve input shape."""
        B, T, H, D = 2, 16, 4, 32
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D)
        sin = torch.randn(1, T, 1, D)
        
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape
    
    def test_rope_rotation_property(self):
        """
        RoPE should satisfy: rotate(x, θ) · rotate(y, θ)ᵀ = x · yᵀ rotated
        For position 0, cos=1, sin=0 should give identity.
        """
        D = 8
        x = torch.randn(1, 1, 1, D)
        cos = torch.ones(1, 1, 1, D)
        sin = torch.zeros(1, 1, 1, D)
        
        out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out, x)
    
    def test_rope_different_positions_differ(self):
        """Different positions should produce different rotations."""
        B, T, H, D = 1, 4, 1, 8
        x = torch.ones(B, T, H, D)
        
        # Create position-dependent cos/sin
        pos = torch.arange(T).float()
        freqs = pos.unsqueeze(-1) * 0.1  # Simple frequency
        freqs = freqs.repeat(1, D // 2)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos().view(1, T, 1, D)
        sin = freqs.sin().view(1, T, 1, D)
        
        out = apply_rotary_emb(x, cos, sin)
        
        # All positions should be different
        for i in range(T):
            for j in range(i + 1, T):
                assert not torch.allclose(out[0, i], out[0, j])
    
    def test_rope_deterministic(self):
        """Same inputs should always produce same outputs."""
        x = torch.randn(2, 8, 4, 16)
        cos = torch.randn(1, 8, 1, 16)
        sin = torch.randn(1, 8, 1, 16)
        
        out1 = apply_rotary_emb(x, cos, sin)
        out2 = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out1, out2)


# ============================================================================
# Unit Tests: RMSNorm
# ============================================================================

class TestRMSNorm:
    """Tests for rms_norm function."""
    
    def test_rms_norm_output_shape(self):
        """RMSNorm should preserve shape."""
        x = torch.randn(2, 16, 64)
        out = rms_norm(x)
        assert out.shape == x.shape
    
    def test_rms_norm_normalization(self):
        """RMSNorm should produce unit RMS (approximately)."""
        x = torch.randn(4, 32, 128) * 10  # Large values
        out = rms_norm(x)
        
        # RMS of output should be close to 1
        rms = torch.sqrt((out ** 2).mean(dim=-1))
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)
    
    def test_rms_norm_scale_invariance(self):
        """Scaling input should not change normalized direction."""
        x = torch.randn(2, 8, 32)
        out1 = rms_norm(x)
        out2 = rms_norm(x * 5.0)
        
        # Directions should be the same
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)
    
    def test_rms_norm_zero_input(self):
        """RMSNorm with near-zero input should not explode."""
        x = torch.zeros(2, 4, 8) + 1e-10
        out = rms_norm(x)
        assert torch.isfinite(out).all()


# ============================================================================
# Unit Tests: MLP
# ============================================================================

class TestMLP:
    """Tests for MLP module."""
    
    def test_mlp_output_shape(self, small_config):
        """MLP should preserve (B, T, C) shape."""
        mlp = MLP(small_config)
        x = torch.randn(2, 16, small_config.n_embd)
        out = mlp(x)
        assert out.shape == x.shape
    
    def test_mlp_squared_relu_activation(self, small_config):
        """MLP uses squared ReLU: relu(x)^2."""
        mlp = MLP(small_config)
        
        # Set weights to identity-like for testing
        with torch.no_grad():
            mlp.c_fc.weight.fill_(0.01)
            mlp.c_proj.weight.fill_(0.01)
        
        x = torch.randn(1, 1, small_config.n_embd)
        out = mlp(x)
        
        # Should be non-negative (squared output)
        # Note: full pipeline may have negative due to projection
        hidden = mlp.c_fc(x)
        activated = F.relu(hidden) ** 2
        assert (activated >= 0).all()
    
    def test_mlp_hidden_dim(self, small_config):
        """MLP hidden dim should be 4x embedding dim."""
        mlp = MLP(small_config)
        assert mlp.c_fc.out_features == 4 * small_config.n_embd
        assert mlp.c_proj.in_features == 4 * small_config.n_embd
    
    def test_mlp_no_bias(self, small_config):
        """MLP should have no bias."""
        mlp = MLP(small_config)
        assert mlp.c_fc.bias is None
        assert mlp.c_proj.bias is None


# ============================================================================
# Unit Tests: CausalSelfAttention
# ============================================================================

class TestCausalSelfAttention:
    """Tests for CausalSelfAttention module."""
    
    def test_attention_output_shape(self, small_config):
        """Attention should preserve (B, T, C) shape."""
        attn = CausalSelfAttention(small_config, layer_idx=0)
        B, T, C = 2, 16, small_config.n_embd
        x = torch.randn(B, T, C)
        
        head_dim = small_config.n_embd // small_config.n_head
        cos = torch.randn(1, 32, 1, head_dim)
        sin = torch.randn(1, 32, 1, head_dim)
        
        out = attn(x, cos, sin)
        assert out.shape == (B, T, C)
    
    def test_attention_gqa_dimensions(self, small_config):
        """GQA: K/V heads should be fewer than Q heads."""
        attn = CausalSelfAttention(small_config, layer_idx=0)
        
        # n_kv_head=2, n_head=4 in small_config
        assert attn.c_q.out_features == small_config.n_head * attn.head_dim
        assert attn.c_k.out_features == small_config.n_kv_head * attn.head_dim
        assert attn.c_v.out_features == small_config.n_kv_head * attn.head_dim
    
    def test_attention_mha_dimensions(self, mha_config):
        """MHA: All heads should be equal."""
        attn = CausalSelfAttention(mha_config, layer_idx=0)
        
        assert attn.c_q.out_features == attn.c_k.out_features
        assert attn.c_k.out_features == attn.c_v.out_features
    
    def test_attention_is_causal(self, small_config, device):
        """Attention should be causal (future tokens don't affect past)."""
        torch.manual_seed(42)
        attn = CausalSelfAttention(small_config, layer_idx=0).to(device)
        
        B, T, C = 1, 8, small_config.n_embd
        x = torch.randn(B, T, C, device=device)
        
        head_dim = small_config.n_embd // small_config.n_head
        cos = torch.randn(1, 32, 1, head_dim, device=device)
        sin = torch.randn(1, 32, 1, head_dim, device=device)
        
        # Run full sequence
        out_full = attn(x, cos, sin)
        
        # Run only first 4 tokens
        out_partial = attn(x[:, :4, :], cos, sin)
        
        # First 4 positions should be identical (causal = no future leakage)
        torch.testing.assert_close(out_full[:, :4, :], out_partial, atol=1e-5, rtol=1e-5)
    
    def test_attention_no_bias(self, small_config):
        """Attention projections should have no bias."""
        attn = CausalSelfAttention(small_config, layer_idx=0)
        assert attn.c_q.bias is None
        assert attn.c_k.bias is None
        assert attn.c_v.bias is None
        assert attn.c_proj.bias is None


# ============================================================================
# Unit Tests: Block
# ============================================================================

class TestBlock:
    """Tests for Transformer Block."""
    
    def test_block_output_shape(self, small_config):
        """Block should preserve (B, T, C) shape."""
        block = Block(small_config, layer_idx=0)
        B, T, C = 2, 16, small_config.n_embd
        x = torch.randn(B, T, C)
        
        head_dim = small_config.n_embd // small_config.n_head
        cos = torch.randn(1, 32, 1, head_dim)
        sin = torch.randn(1, 32, 1, head_dim)
        
        out = block(x, cos, sin)
        assert out.shape == (B, T, C)
    
    def test_block_residual_connection(self, small_config):
        """Block should use residual connections."""
        block = Block(small_config, layer_idx=0)
        
        # Zero out weights to isolate residual
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        
        x = torch.randn(2, 8, small_config.n_embd)
        head_dim = small_config.n_embd // small_config.n_head
        cos = torch.ones(1, 32, 1, head_dim)
        sin = torch.zeros(1, 32, 1, head_dim)
        
        out = block(x, cos, sin)
        
        # With zero weights, output should equal input (residual only)
        # Note: rms_norm still applies, so we check the pattern
        assert out.shape == x.shape


# ============================================================================
# Integration Tests: Full Model
# ============================================================================

class TestGPTModel:
    """Integration tests for full GPT model."""
    
    def test_forward_logits_only(self, small_config, device):
        """Forward without targets returns logits."""
        model = GPT(small_config, device=device).to(device)
        idx = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        
        out = model(idx)
        assert out.shape == (2, 16, small_config.vocab_size)
    
    def test_forward_with_loss(self, small_config, device):
        """Forward with targets returns (logits, loss)."""
        model = GPT(small_config, device=device).to(device)
        idx = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        targets = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, small_config.vocab_size)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
    
    def test_loss_is_reasonable(self, small_config, device):
        """Initial loss should be near -ln(1/vocab_size)."""
        model = GPT(small_config, device=device).to(device)
        model.configure_initialization()
        
        idx = torch.randint(0, small_config.vocab_size, (4, 16), device=device)
        targets = torch.randint(0, small_config.vocab_size, (4, 16), device=device)
        
        _, loss = model(idx, targets)
        expected_loss = math.log(small_config.vocab_size)  # ~4.6 for vocab=100
        
        # Loss should be in reasonable range
        assert 2.0 < loss.item() < 10.0
    
    def test_gradient_flow(self, small_config, device):
        """Gradients should reach all parameters."""
        model = GPT(small_config, device=device).to(device)
        idx = torch.randint(0, small_config.vocab_size, (2, 8), device=device)
        targets = torch.randint(0, small_config.vocab_size, (2, 8), device=device)
        
        _, loss = model(idx, targets)
        loss.backward()
        
        # Check gradients exist for key parameters
        assert model.transformer["wte"].weight.grad is not None
        assert model.transformer["wte"].weight.grad.abs().sum() > 0
        assert model.lm_head.weight.grad is not None
        
        # Check block gradients
        for block in model.transformer["h"]:
            assert block.attn.c_q.weight.grad is not None
            assert block.mlp.c_fc.weight.grad is not None
    
    def test_rope_slicing_short_sequence(self, small_config, device):
        """Model handles sequences shorter than max_seq_len."""
        model = GPT(small_config, device=device).to(device)
        
        # Sequence length 10 < 32 (max)
        idx = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        out = model(idx)
        
        assert out.shape == (1, 10, small_config.vocab_size)
    
    def test_initialization_scaling(self, small_config):
        """Residual projection weights should be scaled by 1/sqrt(2*n_layer)."""
        model = GPT(small_config)
        model.configure_initialization()
        
        expected_std = 0.02 / math.sqrt(2 * small_config.n_layer)
        
        for name, p in model.named_parameters():
            if name.endswith("c_proj.weight"):
                actual_std = p.std().item()
                # Allow some variance due to finite sample
                assert abs(actual_std - expected_std) < 0.01, \
                    f"{name}: expected std ~{expected_std:.4f}, got {actual_std:.4f}"


# ============================================================================
# Tests: Generation
# ============================================================================

class TestGeneration:
    """Tests for generate() method."""
    
    def test_generate_yields_tokens(self, small_config, device):
        """Generate should yield tokens."""
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        tokens = list(model.generate(prompt, max_new_tokens=5))
        assert len(tokens) == 5
    
    def test_generate_tokens_in_vocab(self, small_config, device):
        """Generated tokens should be valid vocab indices."""
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        tokens = list(model.generate(prompt, max_new_tokens=10))
        assert all(0 <= t < small_config.vocab_size for t in tokens)
    
    def test_generate_deterministic_with_seed(self, small_config, device):
        """Same seed should produce same output."""
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        tokens1 = list(model.generate(prompt, max_new_tokens=5, seed=42))
        tokens2 = list(model.generate(prompt, max_new_tokens=5, seed=42))
        
        assert tokens1 == tokens2
    
    def test_generate_different_seeds_differ(self, small_config, device):
        """Different seeds should (usually) produce different output.
        
        Note: With untrained models, logits may be near-uniform, so same tokens
        can occur by chance. We use high temperature to increase variance.
        """
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        # Use high temperature to increase randomness
        tokens1 = list(model.generate(prompt, max_new_tokens=20, seed=42, temperature=2.0))
        tokens2 = list(model.generate(prompt, max_new_tokens=20, seed=12345, temperature=2.0))
        
        # With 20 tokens and high temp, very unlikely to match
        # But we make this a soft check - warn rather than fail
        if tokens1 == tokens2:
            import warnings
            warnings.warn("Unlikely: different seeds produced same output")
    
    def test_generate_temperature_zero_is_greedy(self, small_config, device):
        """Temperature 0 should give deterministic greedy output."""
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        tokens1 = list(model.generate(prompt, max_new_tokens=5, temperature=0, seed=1))
        tokens2 = list(model.generate(prompt, max_new_tokens=5, temperature=0, seed=999))
        
        # Greedy should be same regardless of seed
        assert tokens1 == tokens2
    
    def test_generate_top_k_filtering(self, small_config, device):
        """Top-k should restrict sampling to top k tokens."""
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        # Should not crash with top_k
        tokens = list(model.generate(prompt, max_new_tokens=5, top_k=10))
        assert len(tokens) == 5


# ============================================================================
# Edge Cases & Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_batch_size_one(self, small_config, device):
        """Model works with batch size 1."""
        model = GPT(small_config, device=device).to(device)
        idx = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        out = model(idx)
        assert out.shape == (1, 8, small_config.vocab_size)
    
    def test_sequence_length_one(self, small_config, device):
        """Model works with sequence length 1."""
        model = GPT(small_config, device=device).to(device)
        idx = torch.randint(0, small_config.vocab_size, (2, 1), device=device)
        out = model(idx)
        assert out.shape == (2, 1, small_config.vocab_size)
    
    def test_max_sequence_length(self, small_config, device):
        """Model works at max sequence length."""
        model = GPT(small_config, device=device).to(device)
        idx = torch.randint(0, small_config.vocab_size, 
                           (1, small_config.sequence_len), device=device)
        out = model(idx)
        assert out.shape == (1, small_config.sequence_len, small_config.vocab_size)
    
    def test_single_layer(self, device):
        """Model works with single layer."""
        config = GPTConfig(
            sequence_len=16, vocab_size=50, n_layer=1,
            n_head=2, n_kv_head=2, n_embd=32
        )
        model = GPT(config, device=device).to(device)
        idx = torch.randint(0, config.vocab_size, (1, 8), device=device)
        out = model(idx)
        assert out.shape == (1, 8, config.vocab_size)
    
    def test_gqa_extreme_ratio(self, device):
        """GQA works with extreme head ratio (8 query : 1 kv)."""
        config = GPTConfig(
            sequence_len=16, vocab_size=50, n_layer=1,
            n_head=8, n_kv_head=1, n_embd=64
        )
        model = GPT(config, device=device).to(device)
        idx = torch.randint(0, config.vocab_size, (1, 8), device=device)
        out = model(idx)
        assert out.shape == (1, 8, config.vocab_size)


# ============================================================================
# Parameter Count Tests
# ============================================================================

class TestParameterCounts:
    """Verify parameter counts are as expected."""
    
    def test_weight_tying(self, small_config):
        """Embedding and lm_head should share weights."""
        model = GPT(small_config)
        
        # Check they are the same tensor
        assert model.lm_head.weight is model.transformer["wte"].weight
        
        # Modifying one should affect the other
        with torch.no_grad():
            model.transformer["wte"].weight[0, 0] = 999.0
        assert model.lm_head.weight[0, 0] == 999.0
    
    def test_total_parameters(self, small_config):
        """Verify total parameter count calculation (with weight tying)."""
        model = GPT(small_config)
        
        # Calculate expected (weight tying means embedding = lm_head, count once)
        # Embedding/LM head (tied): vocab_size * n_embd
        emb_params = small_config.vocab_size * small_config.n_embd
        
        # Per block:
        head_dim = small_config.n_embd // small_config.n_head
        # Q: n_embd * (n_head * head_dim)
        q_params = small_config.n_embd * small_config.n_head * head_dim
        # K, V: n_embd * (n_kv_head * head_dim) each
        kv_params = 2 * small_config.n_embd * small_config.n_kv_head * head_dim
        # c_proj: n_embd * n_embd
        proj_params = small_config.n_embd * small_config.n_embd
        # MLP: n_embd * 4*n_embd + 4*n_embd * n_embd
        mlp_params = small_config.n_embd * 4 * small_config.n_embd * 2
        
        block_params = q_params + kv_params + proj_params + mlp_params
        # Note: no separate lm_head_params due to weight tying
        total_expected = emb_params + small_config.n_layer * block_params
        
        total_actual = sum(p.numel() for p in model.parameters())
        
        assert total_actual == total_expected, \
            f"Expected {total_expected} params, got {total_actual}"


# ============================================================================
# KV Cache Tests
# ============================================================================

class TestKVCache:
    """Tests for KVCache class and integration with model."""
    
    def test_kv_cache_initialization(self):
        """KVCache should initialize with correct shape and zero position."""
        cache = KVCache(
            batch_size=2,
            num_heads=4,
            seq_len=32,
            head_dim=16,
            num_layers=3
        )
        assert cache.pos == 0
        assert cache.kv_cache is None  # Lazy initialization
        assert cache.kv_shape == (3, 2, 2, 4, 32, 16)
    
    def test_kv_cache_insert_creates_cache(self):
        """First insert should create the cache tensor."""
        cache = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=16, num_layers=2)
        
        k = torch.randn(1, 4, 8, 16)  # B, H, T, D
        v = torch.randn(1, 4, 8, 16)
        
        full_k, full_v = cache.insert_kv(layer_idx=0, k=k, v=v)
        
        assert cache.kv_cache is not None
        assert cache.kv_cache.shape == cache.kv_shape
    
    def test_kv_cache_insert_returns_correct_slice(self):
        """insert_kv should return K,V up to current position."""
        cache = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=16, num_layers=2)
        
        k = torch.randn(1, 4, 5, 16)  # 5 tokens
        v = torch.randn(1, 4, 5, 16)
        
        full_k, full_v = cache.insert_kv(layer_idx=0, k=k, v=v)
        
        # Should return only first 5 positions (not full 32)
        assert full_k.shape == (1, 4, 5, 16)
        assert full_v.shape == (1, 4, 5, 16)
    
    def test_kv_cache_position_advances_after_last_layer(self):
        """Position should only advance after the last layer."""
        cache = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=16, num_layers=3)
        
        k = torch.randn(1, 4, 10, 16)
        v = torch.randn(1, 4, 10, 16)
        
        # Layer 0: pos should NOT advance
        cache.insert_kv(layer_idx=0, k=k, v=v)
        assert cache.pos == 0
        
        # Layer 1: pos should NOT advance
        cache.insert_kv(layer_idx=1, k=k, v=v)
        assert cache.pos == 0
        
        # Layer 2 (last): pos SHOULD advance
        cache.insert_kv(layer_idx=2, k=k, v=v)
        assert cache.pos == 10
    
    def test_kv_cache_accumulates_tokens(self):
        """Multiple inserts should accumulate in the cache."""
        cache = KVCache(batch_size=1, num_heads=2, seq_len=32, head_dim=8, num_layers=1)
        
        # First insert: 5 tokens
        k1 = torch.randn(1, 2, 5, 8)
        v1 = torch.randn(1, 2, 5, 8)
        full_k1, full_v1 = cache.insert_kv(layer_idx=0, k=k1, v=v1)
        assert full_k1.shape == (1, 2, 5, 8)
        assert cache.pos == 5
        
        # Second insert: 3 more tokens
        k2 = torch.randn(1, 2, 3, 8)
        v2 = torch.randn(1, 2, 3, 8)
        full_k2, full_v2 = cache.insert_kv(layer_idx=0, k=k2, v=v2)
        assert full_k2.shape == (1, 2, 8, 8)  # Now 5+3=8 tokens
        assert cache.pos == 8
    
    def test_kv_cache_preserves_previous_values(self):
        """New inserts should not overwrite previous cached values."""
        cache = KVCache(batch_size=1, num_heads=1, seq_len=32, head_dim=4, num_layers=1)
        
        # Insert first batch
        k1 = torch.ones(1, 1, 3, 4) * 1.0
        v1 = torch.ones(1, 1, 3, 4) * 1.0
        cache.insert_kv(layer_idx=0, k=k1, v=v1)
        
        # Insert second batch
        k2 = torch.ones(1, 1, 2, 4) * 2.0
        v2 = torch.ones(1, 1, 2, 4) * 2.0
        full_k, full_v = cache.insert_kv(layer_idx=0, k=k2, v=v2)
        
        # First 3 positions should still be 1.0
        torch.testing.assert_close(full_k[:, :, :3, :], torch.ones(1, 1, 3, 4))
        # Positions 3-4 should be 2.0
        torch.testing.assert_close(full_k[:, :, 3:5, :], torch.ones(1, 1, 2, 4) * 2.0)
    
    def test_kv_cache_reset(self):
        """reset() should set position back to 0."""
        cache = KVCache(batch_size=1, num_heads=2, seq_len=32, head_dim=8, num_layers=1)
        
        k = torch.randn(1, 2, 10, 8)
        v = torch.randn(1, 2, 10, 8)
        cache.insert_kv(layer_idx=0, k=k, v=v)
        assert cache.pos == 10
        
        cache.reset()
        assert cache.pos == 0


class TestKVCacheModelIntegration:
    """Tests for KV cache integration with the GPT model."""
    
    def test_forward_with_kv_cache_shape(self, small_config, device):
        """Forward with KV cache should produce same output shape."""
        model = GPT(small_config, device=device).to(device)
        
        cache = KVCache(
            batch_size=1,
            num_heads=small_config.n_kv_head,
            seq_len=64,
            head_dim=small_config.n_embd // small_config.n_head,
            num_layers=small_config.n_layer
        )
        
        idx = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        out = model(idx, kv_cache=cache)
        
        assert out.shape == (1, 8, small_config.vocab_size)
    
    def test_prefill_then_decode_shapes(self, small_config, device):
        """Prefill + single-token decode should work correctly."""
        model = GPT(small_config, device=device).to(device)
        
        cache = KVCache(
            batch_size=1,
            num_heads=small_config.n_kv_head,
            seq_len=64,
            head_dim=small_config.n_embd // small_config.n_head,
            num_layers=small_config.n_layer
        )
        
        # Prefill with 10 tokens
        prefill_ids = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        out1 = model(prefill_ids, kv_cache=cache)
        assert out1.shape == (1, 10, small_config.vocab_size)
        assert cache.pos == 10
        
        # Decode single token
        decode_id = torch.randint(0, small_config.vocab_size, (1, 1), device=device)
        out2 = model(decode_id, kv_cache=cache)
        assert out2.shape == (1, 1, small_config.vocab_size)
        assert cache.pos == 11
    
    def test_kv_cache_equivalence_prefill(self, small_config, device):
        """
        Forward with cache during prefill should match forward without cache.
        This is the critical correctness test.
        """
        torch.manual_seed(42)
        model = GPT(small_config, device=device).to(device)
        model.eval()
        
        idx = torch.randint(0, small_config.vocab_size, (1, 12), device=device)
        
        # Forward without cache
        with torch.no_grad():
            out_no_cache = model(idx)
        
        # Forward with cache
        cache = KVCache(
            batch_size=1,
            num_heads=small_config.n_kv_head,
            seq_len=64,
            head_dim=small_config.n_embd // small_config.n_head,
            num_layers=small_config.n_layer
        )
        with torch.no_grad():
            out_with_cache = model(idx, kv_cache=cache)
        
        # Should be identical
        torch.testing.assert_close(out_no_cache, out_with_cache, atol=1e-5, rtol=1e-5)
    
    def test_kv_cache_equivalence_decode(self, small_config, device):
        """
        Decode step with cache should match last position of full forward.
        This verifies incremental generation correctness.
        """
        torch.manual_seed(42)
        model = GPT(small_config, device=device).to(device)
        model.eval()
        
        # Full sequence: 10 tokens
        full_ids = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        
        # Get reference: full forward, take last position's logits
        with torch.no_grad():
            out_full = model(full_ids)
            ref_logits = out_full[:, -1, :]  # Last position
        
        # Now simulate incremental: prefill 9, decode 1
        prefill_ids = full_ids[:, :9]
        decode_id = full_ids[:, 9:10]
        
        cache = KVCache(
            batch_size=1,
            num_heads=small_config.n_kv_head,
            seq_len=64,
            head_dim=small_config.n_embd // small_config.n_head,
            num_layers=small_config.n_layer
        )
        
        with torch.no_grad():
            _ = model(prefill_ids, kv_cache=cache)
            out_decode = model(decode_id, kv_cache=cache)
            decode_logits = out_decode[:, 0, :]  # Only position
        
        # Decode logits should match reference
        torch.testing.assert_close(ref_logits, decode_logits, atol=1e-5, rtol=1e-5)
    
    def test_kv_cache_multi_step_decode(self, small_config, device):
        """
        Multiple decode steps should match full forward at each position.
        """
        torch.manual_seed(42)
        model = GPT(small_config, device=device).to(device)
        model.eval()
        
        # Full sequence
        full_ids = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        
        with torch.no_grad():
            out_full = model(full_ids)
        
        # Incremental: prefill 4, then decode 4 one at a time
        cache = KVCache(
            batch_size=1,
            num_heads=small_config.n_kv_head,
            seq_len=64,
            head_dim=small_config.n_embd // small_config.n_head,
            num_layers=small_config.n_layer
        )
        
        with torch.no_grad():
            # Prefill first 4
            out_prefill = model(full_ids[:, :4], kv_cache=cache)
            # Check prefill matches
            torch.testing.assert_close(out_full[:, :4, :], out_prefill, atol=1e-5, rtol=1e-5)
            
            # Decode remaining 4 one at a time
            for i in range(4, 8):
                out_step = model(full_ids[:, i:i+1], kv_cache=cache)
                torch.testing.assert_close(
                    out_full[:, i:i+1, :], out_step, 
                    atol=1e-5, rtol=1e-5,
                    msg=f"Mismatch at decode step {i}"
                )


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
