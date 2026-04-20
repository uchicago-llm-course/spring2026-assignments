"""
Autograder for Question 2: Attention Residuals.
Do not modify this file.
"""

import torch
import torch.nn as nn
import pytest

from src.attn_residuals import prenorm_residual_layer, full_attn_res, block_attn_res


# ── Helpers ──────────────────────────────────────────────────────────────────

class IdentityMod(nn.Module):
    def forward(self, x):
        return x


class ScaleMod(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class RMSNorm(nn.Module):
    """Standard RMSNorm (no learned scale)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


# ── Tests for prenorm_residual_layer ─────────────────────────────────────────

class TestPreNormResidualLayer:
    def test_identity_sublayer(self):
        """h + identity(norm(h)) = h + norm(h)."""
        torch.manual_seed(0)
        h = torch.randn(2, 3, 8)
        norm = RMSNorm()
        expected = h + norm(h)
        result = prenorm_residual_layer(h, IdentityMod(), norm)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_scale_sublayer(self):
        torch.manual_seed(1)
        h = torch.randn(1, 4, 16)
        norm = RMSNorm()
        expected = h + 2.0 * norm(h)
        result = prenorm_residual_layer(h, ScaleMod(scale=2.0), norm)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_does_not_modify_input(self):
        h = torch.randn(1, 4, 8)
        original = h.clone()
        prenorm_residual_layer(h, IdentityMod(), RMSNorm())
        assert torch.allclose(h, original, atol=1e-8)

    def test_shape_preserved(self):
        h = torch.randn(3, 5, 12)
        result = prenorm_residual_layer(h, IdentityMod(), RMSNorm())
        assert result.shape == h.shape


# ── Tests for full_attn_res ──────────────────────────────────────────────────

class TestFullAttnRes:
    def test_single_layer_returns_input(self):
        """With l=1, softmax over a single key = 1, so output = v_0."""
        torch.manual_seed(0)
        v0 = torch.randn(2, 3, 8)
        query = torch.randn(8)
        result = full_attn_res([v0], query, RMSNorm())
        assert result.shape == (2, 3, 8)
        assert torch.allclose(result, v0, atol=1e-6)

    def test_shape_multi_layer(self):
        torch.manual_seed(1)
        prev = [torch.randn(2, 3, 8) for _ in range(5)]
        query = torch.randn(8)
        result = full_attn_res(prev, query, RMSNorm())
        assert result.shape == (2, 3, 8)

    def test_equal_values_collapse_to_that_value(self):
        """If all v_i are identical, any convex combination equals v."""
        torch.manual_seed(2)
        v = torch.randn(1, 3, 8)
        prev = [v.clone() for _ in range(4)]
        query = torch.randn(8)
        result = full_attn_res(prev, query, RMSNorm())
        assert torch.allclose(result, v, atol=1e-6)

    def test_output_norm_bounded_by_max_value_norm(self):
        """Convex combination ⇒ output norm ≤ max_i ||v_i|| per token."""
        torch.manual_seed(3)
        prev = [torch.randn(1, 2, 4) for _ in range(4)]
        query = torch.randn(4)
        result = full_attn_res(prev, query, RMSNorm())

        max_norms = torch.stack([v.norm(dim=-1) for v in prev]).max(dim=0).values
        result_norms = result.norm(dim=-1)
        assert torch.all(result_norms <= max_norms + 1e-5)

    def test_sharp_query_selects_matching_key(self):
        """A strongly-aligned query should pick the matching value."""
        B, T, d = 1, 2, 4
        v0 = torch.zeros(B, T, d)
        v1 = torch.ones(B, T, d) * 5.0
        v2 = -torch.ones(B, T, d) * 5.0
        prev = [v0, v1, v2]
        # Normalized v1 keys point in the all-ones direction; a large positive
        # query aligned with that should saturate softmax onto v1.
        query = torch.ones(d) * 100.0
        result = full_attn_res(prev, query, RMSNorm())
        assert torch.allclose(result, v1, atol=1e-3)

    def test_does_not_modify_inputs(self):
        torch.manual_seed(4)
        prev = [torch.randn(1, 2, 4) for _ in range(3)]
        originals = [v.clone() for v in prev]
        query = torch.randn(4)
        query_orig = query.clone()
        full_attn_res(prev, query, RMSNorm())
        for v, o in zip(prev, originals):
            assert torch.allclose(v, o, atol=1e-8)
        assert torch.allclose(query, query_orig, atol=1e-8)


# ── Tests for block_attn_res ─────────────────────────────────────────────────

class TestBlockAttnRes:
    def test_shape(self):
        torch.manual_seed(0)
        B, T, d = 2, 3, 8
        summaries = [torch.randn(B, T, d) for _ in range(3)]
        partial = torch.randn(B, T, d)
        query = torch.randn(d)
        proj = nn.Linear(d, d)
        result = block_attn_res(summaries, partial, query, proj, RMSNorm())
        assert result.shape == (B, T, d)

    def test_single_partial_only_returns_partial(self):
        """With zero completed blocks, the only key is partial_block."""
        B, T, d = 1, 2, 4
        partial = torch.randn(B, T, d)
        query = torch.randn(d)
        proj = nn.Linear(d, d)
        result = block_attn_res([], partial, query, proj, RMSNorm())
        assert result.shape == (B, T, d)
        assert torch.allclose(result, partial, atol=1e-6)

    def test_output_bounded_by_keys(self):
        """Output is a convex combination of block summaries + partial."""
        torch.manual_seed(1)
        B, T, d = 1, 2, 4
        summaries = [torch.randn(B, T, d) for _ in range(3)]
        partial = torch.randn(B, T, d)
        query = torch.randn(d)
        proj = nn.Linear(d, d)
        result = block_attn_res(summaries, partial, query, proj, RMSNorm())

        all_keys = summaries + [partial]
        max_norms = torch.stack([v.norm(dim=-1) for v in all_keys]).max(dim=0).values
        assert torch.all(result.norm(dim=-1) <= max_norms + 1e-5)

    def test_equal_keys_collapse(self):
        """If every key is the same tensor, output equals that tensor."""
        torch.manual_seed(2)
        B, T, d = 1, 2, 4
        v = torch.randn(B, T, d)
        summaries = [v.clone() for _ in range(3)]
        partial = v.clone()
        query = torch.randn(d)
        proj = nn.Linear(d, d)
        result = block_attn_res(summaries, partial, query, proj, RMSNorm())
        assert torch.allclose(result, v, atol=1e-6)
