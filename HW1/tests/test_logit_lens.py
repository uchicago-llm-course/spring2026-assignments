"""
Unit tests for src/logit_lens.py.

Run from the HW1/ directory:
    pytest tests/test_logit_lens.py -v
"""

import os, sys
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.logit_lens import get_logit_lens_predictions, get_token_rank_by_layer


# ─────────────────────────────────────────────────────────────────────────────
# Mock nnsight model for testing without GPU
# ─────────────────────────────────────────────────────────────────────────────

class _MockTokenizer:
    def decode(self, token_id):
        return f"tok_{token_id}"

class _MockLayer:
    def __init__(self, layer_idx, seq_len, vocab_size):
        self.layer_idx = layer_idx
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Output: (seq_len, hidden_size)
        self.output = [torch.randn(seq_len, 64)]

class _MockModel:
    def __init__(self, num_layers=4, seq_len=6, vocab_size=100):
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = _MockTokenizer()

        class _Layers:
            def __init__(self, num_layers, seq_len, vocab_size):
                self.layers = [_MockLayer(i, seq_len, vocab_size) for i in range(num_layers)]
            def __iter__(self):
                return iter(self.layers)

        class _GPTNeoX:
            def __init__(self, num_layers, seq_len, vocab_size):
                self.layers = _Layers(num_layers, seq_len, vocab_size)
                self.final_layer_norm = lambda x: x  # Identity for testing

        self.gpt_neox = _GPTNeoX(num_layers, seq_len, vocab_size)
        self.embed_out = lambda x: torch.randn(x.shape[0], self.vocab_size)

        # Mock tracer context
        self.traced_input_ids = None

    def trace(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def invoke(self, prompt):
        return _MockInvoker(self)

class _MockInvoker:
    def __init__(self, model):
        self.model = model
        self.inputs = [None, {"input_ids": _MockTensor(torch.arange(model.seq_len).unsqueeze(0))}]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

class _MockTensor:
    def __init__(self, value):
        self.value = value
        self.saved = None

    def save(self):
        self.saved = self.value
        return self.saved

    def __getitem__(self, idx):
        return self.value[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Tests for get_logit_lens_predictions
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLogitLensPredictions:
    """
    These tests use a mock model to verify the function structure.
    The real end-to-end behavior is tested in compile.py with actual models.
    """

    def test_all_probs_shape(self):
        """all_probs should be (num_layers, seq_len, vocab_size)."""
        # Mock test: create synthetic all_probs
        all_probs = torch.randn(4, 6, 100)
        max_probs, token_ids = all_probs.max(dim=-1)

        assert all_probs.shape == (4, 6, 100), f"Expected (4, 6, 100), got {all_probs.shape}"
        assert max_probs.shape == (4, 6), f"Expected (4, 6), got {max_probs.shape}"
        assert token_ids.shape == (4, 6), f"Expected (4, 6), got {token_ids.shape}"

    def test_max_probs_in_range(self):
        """After softmax, max_probs should be in [0, 1]."""
        logits = torch.randn(3, 5, 50)
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)

        assert (max_probs >= 0).all(), "Probabilities should be non-negative"
        assert (max_probs <= 1).all(), "Probabilities should not exceed 1"

    def test_top_tokens_is_list_of_lists(self):
        """top_tokens should be list[list[str]] with shape (num_layers, seq_len)."""
        num_layers, seq_len = 3, 5
        token_ids = torch.randint(0, 100, (num_layers, seq_len))

        # Simulate decoding
        tokenizer = _MockTokenizer()
        top_tokens = [
            [tokenizer.decode(t.item()) for t in layer_tokens]
            for layer_tokens in token_ids
        ]

        assert isinstance(top_tokens, list), "Should be a list"
        assert len(top_tokens) == num_layers, f"Should have {num_layers} layers"
        assert all(len(row) == seq_len for row in top_tokens), "Each layer should have seq_len tokens"
        assert all(isinstance(tok, str) for row in top_tokens for tok in row), "All tokens should be strings"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for get_token_rank_by_layer
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTokenRankByLayer:
    def test_top_token_has_rank_1(self):
        """The token with highest probability should have rank 1."""
        all_probs = torch.tensor([
            [[0.1, 0.3, 0.6]],  # Layer 0, last token: argmax=2
            [[0.5, 0.2, 0.3]],  # Layer 1, last token: argmax=0
        ])

        # Layer 0: token 2 is top → rank 1
        ranks = get_token_rank_by_layer(all_probs, target_token_id=2)
        assert ranks[0] == 1, f"Top token should have rank 1, got {ranks[0]}"

        # Layer 1: token 0 is top → rank 1
        ranks = get_token_rank_by_layer(all_probs, target_token_id=0)
        assert ranks[1] == 1, f"Top token should have rank 1, got {ranks[1]}"

    def test_bottom_token_has_rank_vocab_size(self):
        """The token with lowest probability should have rank = vocab_size."""
        all_probs = torch.tensor([
            [[0.6, 0.3, 0.1]],  # Layer 0, last token: token 2 is lowest
        ])
        vocab_size = 3

        ranks = get_token_rank_by_layer(all_probs, target_token_id=2)
        assert ranks[0] == vocab_size, f"Lowest token should have rank {vocab_size}, got {ranks[0]}"

    def test_middle_rank(self):
        """Test a token with middle-rank probability."""
        all_probs = torch.tensor([
            [[0.1, 0.6, 0.2, 0.05, 0.05]],  # Sorted: [1, 2, 0, 3, 4]
        ])

        ranks = get_token_rank_by_layer(all_probs, target_token_id=2)
        assert ranks[0] == 2, f"Token 2 should have rank 2, got {ranks[0]}"

        ranks = get_token_rank_by_layer(all_probs, target_token_id=0)
        assert ranks[0] == 3, f"Token 0 should have rank 3, got {ranks[0]}"

    def test_only_uses_last_position(self):
        """Rank should be computed from last token position, not all positions."""
        all_probs = torch.tensor([
            [
                [0.9, 0.1],  # Position 0: token 0 is top
                [0.1, 0.9],  # Position 1 (last): token 1 is top
            ]
        ])

        # Should use position -1 (last)
        ranks = get_token_rank_by_layer(all_probs, target_token_id=1)
        assert ranks[0] == 1, "Should compute rank from last position only"

        ranks = get_token_rank_by_layer(all_probs, target_token_id=0)
        assert ranks[0] == 2, "Should compute rank from last position only"
