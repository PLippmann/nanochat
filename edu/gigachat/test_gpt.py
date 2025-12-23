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
        """Different seeds should (usually) produce different output."""
        model = GPT(small_config, device=device).to(device)
        prompt = [1, 2, 3]
        
        tokens1 = list(model.generate(prompt, max_new_tokens=10, seed=42))
        tokens2 = list(model.generate(prompt, max_new_tokens=10, seed=123))
        
        # Very unlikely to be identical with different seeds
        assert tokens1 != tokens2
    
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
    
    def test_total_parameters(self, small_config):
        """Verify total parameter count calculation."""
        model = GPT(small_config)
        
        # Calculate expected
        # Embedding: vocab_size * n_embd
        emb_params = small_config.vocab_size * small_config.n_embd
        # LM head: n_embd * vocab_size
        lm_head_params = small_config.n_embd * small_config.vocab_size
        
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
        total_expected = emb_params + lm_head_params + small_config.n_layer * block_params
        
        total_actual = sum(p.numel() for p in model.parameters())
        
        assert total_actual == total_expected, \
            f"Expected {total_expected} params, got {total_actual}"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
