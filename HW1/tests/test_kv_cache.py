"""
Unit tests for MultiLayerKVCache in src/kv_cache.py.

Run from HW1/:  pytest tests/test_kv_cache.py -v
Uses only CPU tensors — no model downloads required.
"""

import os, sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kv_cache import MultiLayerKVCache


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_cache():
    """4 layers, 40-token total budget, 2 heads, head_dim 8."""
    return MultiLayerKVCache(num_layers=4, total_budget=40, num_heads=2, d_head=8)


def _kv(num_heads, T, d_head):
    """Create random (1, num_heads, T, d_head) K/V pair."""
    k = torch.randn(1, num_heads, T, d_head)
    v = torch.randn(1, num_heads, T, d_head)
    return k, v


# ─────────────────────────────────────────────────────────────────────────────
# allocate_uniform
# ─────────────────────────────────────────────────────────────────────────────

class TestAllocateUniform:
    def test_equal_budgets(self, small_cache):
        small_cache.allocate_uniform()
        assert small_cache.layer_budgets == [10, 10, 10, 10]

    def test_floor_division(self):
        cache = MultiLayerKVCache(num_layers=3, total_budget=10, num_heads=1, d_head=4)
        cache.allocate_uniform()
        per = cache.layer_budgets[0]
        assert per == 3                        # floor(10/3) = 3
        assert all(b == per for b in cache.layer_budgets)

    def test_zero_budget(self):
        """With budget=0, all layers should get 0 tokens."""
        cache = MultiLayerKVCache(num_layers=3, total_budget=0, num_heads=1, d_head=4)
        cache.allocate_uniform()
        assert all(b == 0 for b in cache.layer_budgets)



# ─────────────────────────────────────────────────────────────────────────────
# allocate_priority
# ─────────────────────────────────────────────────────────────────────────────

class TestAllocatePriority:
    def test_higher_score_gets_more_budget(self, small_cache):
        scores = [1.0, 2.0, 3.0, 4.0]
        small_cache.allocate_priority(scores)
        b = small_cache.layer_budgets
        assert b[3] >= b[2] >= b[1] >= b[0], "Higher score → more budget"

    def test_total_within_budget(self, small_cache):
        small_cache.allocate_priority([1.0, 1.0, 2.0, 6.0])
        assert sum(small_cache.layer_budgets) <= small_cache.total_budget

    def test_uniform_scores_gives_roughly_equal_budget(self, small_cache):
        small_cache.allocate_priority([1.0, 1.0, 1.0, 1.0])
        b = small_cache.layer_budgets
        assert max(b) - min(b) <= 1          # floor rounding at most 1 token diff



# ─────────────────────────────────────────────────────────────────────────────
# update
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_first_update_initialises_cache(self, small_cache):
        small_cache.allocate_uniform()
        k, v = _kv(2, 5, 8)
        small_cache.update(0, k, v)
        assert small_cache.tokens_cached(0) == 5

    def test_second_update_appends(self, small_cache):
        small_cache.allocate_uniform()
        small_cache.update(0, *_kv(2, 3, 8))
        small_cache.update(0, *_kv(2, 4, 8))
        assert small_cache.tokens_cached(0) == 7

    def test_fifo_eviction_enforces_budget(self):
        cache = MultiLayerKVCache(num_layers=1, total_budget=5, num_heads=2, d_head=8)
        cache.allocate_uniform()             # budget = 5
        cache.update(0, *_kv(2, 4, 8))      # 4 tokens
        cache.update(0, *_kv(2, 3, 8))      # +3 → 7 > 5, evict 2
        assert cache.tokens_cached(0) == 5
        assert cache.eviction_counts[0] == 2

    def test_fifo_preserves_newest_tokens(self):
        cache = MultiLayerKVCache(num_layers=1, total_budget=3, num_heads=1, d_head=4)
        cache.allocate_uniform()

        k_old = torch.ones(1, 1, 2, 4) * 1.0
        k_new = torch.ones(1, 1, 2, 4) * 9.0
        cache.update(0, k_old, torch.zeros_like(k_old))
        cache.update(0, k_new, torch.zeros_like(k_new))

        k_out, _ = cache.get(0)
        assert k_out.shape[2] == 3
        # Last 2 positions must be from k_new
        assert torch.allclose(k_out[:, :, -2:, :], k_new)

    def test_layers_are_independent(self, small_cache):
        small_cache.allocate_uniform()
        small_cache.update(0, *_kv(2, 5, 8))
        small_cache.update(2, *_kv(2, 3, 8))
        assert small_cache.tokens_cached(0) == 5
        assert small_cache.tokens_cached(1) == 0
        assert small_cache.tokens_cached(2) == 3

    def test_single_token_update(self, small_cache):
        small_cache.allocate_uniform()
        small_cache.update(0, *_kv(2, 1, 8))
        assert small_cache.tokens_cached(0) == 1

    def test_update_larger_than_budget_evicts_all(self):
        """If new tokens exceed budget, evict everything and keep only newest."""
        cache = MultiLayerKVCache(num_layers=1, total_budget=3, num_heads=1, d_head=4)
        cache.allocate_uniform()
        cache.update(0, *_kv(1, 2, 4))  # Add 2 tokens
        cache.update(0, *_kv(1, 5, 4))  # Add 5 tokens (>3 budget)

        assert cache.tokens_cached(0) == 3, "Should keep only budget tokens"
        # Should keep last 3 of the 5 new tokens
        k_out, _ = cache.get(0)
        assert k_out.shape[2] == 3


# ─────────────────────────────────────────────────────────────────────────────
# get
# ─────────────────────────────────────────────────────────────────────────────

class TestGet:
    def test_empty_returns_none(self, small_cache):
        small_cache.allocate_uniform()
        k, v = small_cache.get(0)
        assert k is None and v is None

    def test_shape_after_update(self, small_cache):
        small_cache.allocate_uniform()
        small_cache.update(0, *_kv(2, 6, 8))
        k, v = small_cache.get(0)
        assert k.shape == (1, 2, 6, 8)
        assert v.shape == (1, 2, 6, 8)

    def test_values_preserved(self, small_cache):
        small_cache.allocate_uniform()
        k_in, v_in = _kv(2, 4, 8)
        small_cache.update(0, k_in, v_in)
        k_out, v_out = small_cache.get(0)
        assert torch.allclose(k_out, k_in)
        assert torch.allclose(v_out, v_in)


# ─────────────────────────────────────────────────────────────────────────────
# to_hf_format
# ─────────────────────────────────────────────────────────────────────────────

class TestToHFFormat:
    def test_empty_cache_returns_none(self, small_cache):
        small_cache.allocate_uniform()
        assert small_cache.to_hf_format() is None

    def test_partially_filled_returns_none(self, small_cache):
        small_cache.allocate_uniform()
        small_cache.update(0, *_kv(2, 3, 8))   # only layer 0 filled
        assert small_cache.to_hf_format() is None

    def test_full_cache_returns_correct_structure(self, small_cache):
        small_cache.allocate_uniform()
        for i in range(4):
            small_cache.update(i, *_kv(2, 5, 8))
        hf = small_cache.to_hf_format()
        assert hf is not None

        # Handle both transformers <5.0 (tuple) and 5.x (DynamicCache)
        try:
            from transformers.cache_utils import DynamicCache
            if isinstance(hf, DynamicCache):
                assert len(hf) == 4  # 4 layers
                # Access via iteration (DynamicCache is iterable)
                for layer_data in hf:
                    assert len(layer_data) >= 2  # (K, V) or (K, V, None)
                    assert layer_data[0].shape == (1, 2, 5, 8)
                return
        except ImportError:
            pass

        # Fallback for tuple format (transformers <5.0)
        assert len(hf) == 4
        assert len(hf[0]) == 2              # (K, V) pair
        assert hf[0][0].shape == (1, 2, 5, 8)
