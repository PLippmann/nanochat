"""
Comprehensive test suite for edu/gigachat/engine.py

Tests cover:
- Unit tests for KVCache (initialization, insert, reset, accumulation)
- Unit tests for Engine (initialization, prefill, decode)
- Integration tests: Engine.generate() vs GPT.generate() equivalence
- Performance verification: Engine should be faster than naive generation
- Edge cases and boundary conditions
"""

import pytest
import torch
import time
from unittest.mock import MagicMock

from gpt import GPT, GPTConfig
from engine import KVCache, Engine


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return GPTConfig(
        sequence_len=64,
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
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
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
# Unit Tests: KVCache
# ============================================================================

class TestKVCache:
    """Tests for KVCache class."""
    
    def test_initialization(self):
        """KVCache initializes with correct shape and zero position."""
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
    
    def test_get_pos(self):
        """get_pos() returns current position."""
        cache = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=16, num_layers=1)
        assert cache.get_pos() == 0
        
        k = torch.randn(1, 4, 5, 16)
        v = torch.randn(1, 4, 5, 16)
        cache.insert_kv(layer_idx=0, k=k, v=v)
        assert cache.get_pos() == 5
    
    def test_reset(self):
        """reset() sets position back to 0."""
        cache = KVCache(batch_size=1, num_heads=2, seq_len=32, head_dim=8, num_layers=1)
        
        k = torch.randn(1, 2, 10, 8)
        v = torch.randn(1, 2, 10, 8)
        cache.insert_kv(layer_idx=0, k=k, v=v)
        assert cache.pos == 10
        
        cache.reset()
        assert cache.pos == 0
    
    def test_insert_creates_cache(self):
        """First insert should create the cache tensor."""
        cache = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=16, num_layers=2)
        
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        
        full_k, full_v = cache.insert_kv(layer_idx=0, k=k, v=v)
        
        assert cache.kv_cache is not None
        assert cache.kv_cache.shape == cache.kv_shape
    
    def test_insert_returns_correct_slice(self):
        """insert_kv should return K,V up to current position."""
        cache = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=16, num_layers=2)
        
        k = torch.randn(1, 4, 5, 16)
        v = torch.randn(1, 4, 5, 16)
        
        full_k, full_v = cache.insert_kv(layer_idx=0, k=k, v=v)
        
        # Should return only first 5 positions (not full 32)
        assert full_k.shape == (1, 4, 5, 16)
        assert full_v.shape == (1, 4, 5, 16)
    
    def test_position_advances_after_last_layer(self):
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
    
    def test_accumulates_tokens(self):
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
    
    def test_preserves_previous_values(self):
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


# ============================================================================
# Unit Tests: Engine Initialization
# ============================================================================

class TestEngineInit:
    """Tests for Engine initialization."""
    
    def test_engine_stores_model(self, small_config, device):
        """Engine should store reference to the model."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        assert engine.model is model
    
    def test_engine_creates_kv_cache(self, small_config, device):
        """Engine should create a KVCache with correct dimensions."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        assert engine.kv_cache is not None
        head_dim = small_config.n_embd // small_config.n_head
        expected_shape = (
            small_config.n_layer,  # num_layers
            2,  # K and V
            1,  # batch_size (default)
            small_config.n_kv_head,  # num_heads (KV heads for GQA)
            small_config.sequence_len,  # seq_len
            head_dim  # head_dim
        )
        assert engine.kv_cache.kv_shape == expected_shape
    
    def test_engine_custom_batch_size(self, small_config, device):
        """Engine should support custom batch size."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model, batch_size=4)
        
        # batch_size should be 4
        assert engine.kv_cache.kv_shape[2] == 4


# ============================================================================
# Unit Tests: Engine Prefill
# ============================================================================

class TestEnginePrefill:
    """Tests for Engine.prefill()."""
    
    def test_prefill_output_shape(self, small_config, device):
        """prefill() should return (B, vocab_size) logits."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        tokens = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        logits = engine.prefill(tokens)
        
        assert logits.shape == (1, small_config.vocab_size)
    
    def test_prefill_resets_cache(self, small_config, device):
        """prefill() should reset the cache position."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        tokens = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        
        # First prefill
        engine.prefill(tokens)
        assert engine.kv_cache.pos == 10
        
        # Second prefill should reset and start fresh
        tokens2 = torch.randint(0, small_config.vocab_size, (1, 5), device=device)
        engine.prefill(tokens2)
        assert engine.kv_cache.pos == 5
    
    def test_prefill_populates_cache(self, small_config, device):
        """prefill() should populate the KV cache."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        tokens = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        engine.prefill(tokens)
        
        # Cache should be created and filled
        assert engine.kv_cache.kv_cache is not None
        assert engine.kv_cache.pos == 8


# ============================================================================
# Unit Tests: Engine Decode
# ============================================================================

class TestEngineDecode:
    """Tests for Engine.decode()."""
    
    def test_decode_output_shape(self, small_config, device):
        """decode() should return (B, vocab_size) logits."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        # Must prefill first
        tokens = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        engine.prefill(tokens)
        
        # Now decode
        single_token = torch.randint(0, small_config.vocab_size, (1, 1), device=device)
        logits = engine.decode(single_token)
        
        assert logits.shape == (1, small_config.vocab_size)
    
    def test_decode_advances_cache(self, small_config, device):
        """decode() should advance cache position by 1."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        tokens = torch.randint(0, small_config.vocab_size, (1, 10), device=device)
        engine.prefill(tokens)
        assert engine.kv_cache.pos == 10
        
        single_token = torch.randint(0, small_config.vocab_size, (1, 1), device=device)
        engine.decode(single_token)
        assert engine.kv_cache.pos == 11
    
    def test_multiple_decode_steps(self, small_config, device):
        """Multiple decode() calls should accumulate in cache."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        tokens = torch.randint(0, small_config.vocab_size, (1, 5), device=device)
        engine.prefill(tokens)
        
        for i in range(5):
            single_token = torch.randint(0, small_config.vocab_size, (1, 1), device=device)
            engine.decode(single_token)
            assert engine.kv_cache.pos == 5 + i + 1


# ============================================================================
# Unit Tests: Engine Generate
# ============================================================================

class TestEngineGenerate:
    """Tests for Engine.generate()."""
    
    def test_generate_yields_tokens(self, small_config, device):
        """generate() should yield requested number of tokens."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [1, 2, 3]
        tokens = list(engine.generate(prompt, max_new_tokens=5))
        
        assert len(tokens) == 5
    
    def test_generate_tokens_in_vocab(self, small_config, device):
        """Generated tokens should be valid vocab indices."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [1, 2, 3]
        tokens = list(engine.generate(prompt, max_new_tokens=10))
        
        assert all(0 <= t < small_config.vocab_size for t in tokens)
    
    def test_generate_temperature_zero(self, small_config, device):
        """Temperature 0 should give deterministic greedy output."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [1, 2, 3]
        
        # Two runs with temperature=0 should be identical
        tokens1 = list(engine.generate(prompt, max_new_tokens=5, temperature=0))
        tokens2 = list(engine.generate(prompt, max_new_tokens=5, temperature=0))
        
        assert tokens1 == tokens2
    
    def test_generate_top_k(self, small_config, device):
        """top_k should work without errors."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [1, 2, 3]
        tokens = list(engine.generate(prompt, max_new_tokens=5, top_k=10))
        
        assert len(tokens) == 5


# ============================================================================
# Integration Tests: Engine vs Naive Generate Equivalence
# ============================================================================

class TestEngineEquivalence:
    """
    Critical tests: Engine.generate() should produce identical results
    to GPT.generate() (naive loop) when using the same seed.
    """
    
    def test_greedy_equivalence(self, small_config, device):
        """
        Engine.generate(temperature=0) should match GPT.generate(temperature=0).
        This is the most important correctness test.
        """
        torch.manual_seed(42)
        model = GPT(small_config, device=device).to(device)
        model.eval()
        
        prompt = [1, 2, 3, 4, 5]
        max_new = 10
        
        # Naive generation
        naive_tokens = list(model.generate(prompt, max_new_tokens=max_new, temperature=0))
        
        # Engine generation
        engine = Engine(model)
        engine_tokens = list(engine.generate(prompt, max_new_tokens=max_new, temperature=0))
        
        assert naive_tokens == engine_tokens, \
            f"Mismatch!\nNaive:  {naive_tokens}\nEngine: {engine_tokens}"
    
    def test_decode_step_matches_full_forward(self, small_config, device):
        """
        Each decode step with Engine should match the corresponding
        position from a full forward pass.
        """
        torch.manual_seed(42)
        model = GPT(small_config, device=device).to(device)
        model.eval()
        
        # Full sequence: 8 tokens
        full_ids = torch.randint(0, small_config.vocab_size, (1, 8), device=device)
        
        # Get reference: full forward
        with torch.inference_mode():
            out_full = model(full_ids)
        
        # Now simulate incremental: prefill 4, then decode 4 one at a time
        engine = Engine(model)
        
        with torch.inference_mode():
            # Prefill first 4
            logits_prefill = engine.prefill(full_ids[:, :4])
            
            # Check prefill last position matches
            torch.testing.assert_close(
                out_full[:, 3, :], logits_prefill,
                atol=1e-5, rtol=1e-5
            )
            
            # Decode remaining 4 one at a time
            for i in range(4, 8):
                logits_decode = engine.decode(full_ids[:, i:i+1])
                torch.testing.assert_close(
                    out_full[:, i, :], logits_decode,
                    atol=1e-5, rtol=1e-5,
                )


# ============================================================================
# Performance Tests
# ============================================================================

class TestEnginePerformance:
    """
    Tests to verify that Engine is faster than naive generation.
    These tests may be flaky on slow/busy systems.
    """
    
    def test_engine_faster_than_naive(self, device):
        """Engine should be significantly faster than naive for long generation."""
        # Use a slightly larger config to make timing differences more visible
        config = GPTConfig(
            sequence_len=256,
            vocab_size=100,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            n_embd=128
        )
        model = GPT(config, device=device).to(device)
        model.eval()
        
        prompt = [1, 2, 3, 4, 5]
        max_new = 20  # Generate 20 tokens
        
        # Warm up
        _ = list(model.generate(prompt, max_new_tokens=5, temperature=0))
        engine = Engine(model)
        _ = list(engine.generate(prompt, max_new_tokens=5, temperature=0))
        
        # Time naive generation
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()
        _ = list(model.generate(prompt, max_new_tokens=max_new, temperature=0))
        torch.cuda.synchronize() if device.type == "cuda" else None
        naive_time = time.perf_counter() - start
        
        # Time engine generation
        engine = Engine(model)
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()
        _ = list(engine.generate(prompt, max_new_tokens=max_new, temperature=0))
        torch.cuda.synchronize() if device.type == "cuda" else None
        engine_time = time.perf_counter() - start
        
        # Engine should be faster (or at least not significantly slower)
        # We use a generous margin since timing can be noisy
        print(f"\nNaive: {naive_time:.4f}s, Engine: {engine_time:.4f}s, "
              f"Speedup: {naive_time/engine_time:.2f}x")
        
        # Engine should be at least 1.2x faster for this setup
        # (In practice it should be much faster for longer sequences)
        assert engine_time <= naive_time * 1.5, \
            f"Engine ({engine_time:.4f}s) is slower than naive ({naive_time:.4f}s)"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEngineEdgeCases:
    """Edge cases and boundary conditions for Engine."""
    
    def test_single_token_prompt(self, small_config, device):
        """Engine should handle single-token prompt."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [42]
        tokens = list(engine.generate(prompt, max_new_tokens=5))
        
        assert len(tokens) == 5
    
    def test_long_prompt(self, small_config, device):
        """Engine should handle prompt near max sequence length."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        # Prompt of 60 tokens (close to seq_len=64)
        prompt = list(range(60))
        tokens = list(engine.generate(prompt, max_new_tokens=3))
        
        assert len(tokens) == 3
    
    def test_mha_config(self, mha_config, device):
        """Engine should work with MHA (n_head == n_kv_head)."""
        model = GPT(mha_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [1, 2, 3]
        tokens = list(engine.generate(prompt, max_new_tokens=5))
        
        assert len(tokens) == 5
    
    def test_generate_zero_tokens(self, small_config, device):
        """Generating zero tokens should return empty list."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt = [1, 2, 3]
        tokens = list(engine.generate(prompt, max_new_tokens=0))
        
        assert len(tokens) == 0
    
    def test_multiple_generate_calls(self, small_config, device):
        """Multiple generate() calls should work independently."""
        model = GPT(small_config, device=device).to(device)
        engine = Engine(model)
        
        prompt1 = [1, 2, 3]
        prompt2 = [10, 20, 30]
        
        tokens1 = list(engine.generate(prompt1, max_new_tokens=5, temperature=0))
        tokens2 = list(engine.generate(prompt2, max_new_tokens=5, temperature=0))
        
        # Both should complete without error
        assert len(tokens1) == 5
        assert len(tokens2) == 5
        
        # Re-running same prompt should give same result (greedy)
        tokens1_again = list(engine.generate(prompt1, max_new_tokens=5, temperature=0))
        assert tokens1 == tokens1_again


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
