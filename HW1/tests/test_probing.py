"""
Unit tests for src/probing.py.

Run from the HW1/ directory:
    pytest tests/test_probing.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Make sure the project root is on the path regardless of where pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probing import ProbingClassifier, build_classifier, get_sentence_repr


# ─────────────────────────────────────────────────────────────────────────────
# ProbingClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestProbingClassifier:
    def test_output_shape(self):
        """Forward pass should return (batch_size, num_classes)."""
        clf = ProbingClassifier(input_dim=64, num_classes=2)
        x = torch.randn(8, 64)
        out = clf(x)
        assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"

    def test_exactly_one_linear_layer(self):
        """A linear probe must have exactly one nn.Linear module."""
        clf = ProbingClassifier(input_dim=32, num_classes=3)
        linears = [m for m in clf.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 1, (
            f"Expected exactly 1 Linear layer, found {len(linears)}"
        )

    def test_gradients_flow(self):
        """Loss.backward() should produce non-None gradients for all params."""
        clf = ProbingClassifier(input_dim=16, num_classes=2)
        x = torch.randn(4, 16)
        out = clf(x)
        out.sum().backward()
        for name, param in clf.named_parameters():
            assert param.grad is not None, f"No gradient for parameter '{name}'"

    def test_output_is_raw_logits(self):
        """Output should NOT already be a probability distribution (no softmax)."""
        clf = ProbingClassifier(input_dim=32, num_classes=2)
        x = torch.randn(10, 32)
        out = clf(x)
        # If softmax were applied, rows would sum to 1.  Raw logits typically don't.
        row_sums = out.detach().softmax(dim=-1).sum(dim=-1)
        # After softmax the sum is 1; before it the row sum can be anything.
        # Here we check the raw output is not already summing to 1.
        raw_sums = out.detach().sum(dim=-1).abs()
        assert not torch.allclose(raw_sums, torch.ones(10)), (
            "Output looks like it already has softmax applied. "
            "Return raw logits instead."
        )

    def test_handles_single_sample(self):
        """Should work with batch_size=1."""
        clf = ProbingClassifier(input_dim=64, num_classes=2)
        x = torch.randn(1, 64)
        out = clf(x)
        assert out.shape == (1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# build_classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildClassifier:
    def test_return_types(self):
        clf, crit, opt = build_classifier(emb_dim=64, num_labels=2)
        assert isinstance(clf,  ProbingClassifier),   "First return value should be a ProbingClassifier"
        assert isinstance(crit, torch.nn.CrossEntropyLoss), "Second return value should be CrossEntropyLoss"
        assert isinstance(opt,  torch.optim.Adam),    "Third return value should be Adam"

    def test_classifier_on_device(self):
        device = "cpu"
        clf, _, _ = build_classifier(emb_dim=32, num_labels=2, device=device)
        for param in clf.parameters():
            assert param.device.type == device, (
                f"Parameter should be on '{device}', found '{param.device.type}'"
            )



# ─────────────────────────────────────────────────────────────────────────────
# get_sentence_repr
# ─────────────────────────────────────────────────────────────────────────────

# ---------- Minimal mocks so the test runs without GPU or model downloads ----

HIDDEN_SIZE = 32
NUM_LAYERS  = 4      # transformer layers (not counting embedding)
SEQ_LEN     = 5      # tokens the mock tokenizer returns


class _MockConfig:
    hidden_size = HIDDEN_SIZE


class _MockOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _MockModel:
    config = _MockConfig()

    def __call__(self, input_ids):
        bs, seq_len = input_ids.shape
        # Index 0 = embedding; indices 1..NUM_LAYERS = transformer layers
        hs = tuple(
            torch.zeros(bs, seq_len, HIDDEN_SIZE)
            for _ in range(NUM_LAYERS + 1)
        )
        return _MockOutput(hs)


class _MockTokenizer:
    def encode(self, text: str):
        return list(range(SEQ_LEN))   # always returns SEQ_LEN tokens


# -----------------------------------------------------------------------------

class TestGetSentenceRepr:
    def test_return_type(self):
        result = get_sentence_repr("hello", _MockModel(), _MockTokenizer(), torch.device("cpu"))
        assert isinstance(result, np.ndarray), "Should return a numpy array"

    def test_ndim(self):
        result = get_sentence_repr("hello", _MockModel(), _MockTokenizer(), torch.device("cpu"))
        assert result.ndim == 3, f"Expected 3-D array (layers, tokens, hidden), got {result.ndim}-D"

    def test_num_layers(self):
        result = get_sentence_repr("hello", _MockModel(), _MockTokenizer(), torch.device("cpu"))
        expected = NUM_LAYERS + 1   # embedding layer + N transformer layers
        assert result.shape[0] == expected, (
            f"Expected {expected} layer slices (embedding + {NUM_LAYERS} layers), "
            f"got {result.shape[0]}"
        )

    def test_seq_len(self):
        result = get_sentence_repr("hello", _MockModel(), _MockTokenizer(), torch.device("cpu"))
        assert result.shape[1] == SEQ_LEN, (
            f"Expected seq_len={SEQ_LEN}, got {result.shape[1]}"
        )

    def test_hidden_size(self):
        result = get_sentence_repr("hello", _MockModel(), _MockTokenizer(), torch.device("cpu"))
        assert result.shape[2] == HIDDEN_SIZE, (
            f"Expected hidden_size={HIDDEN_SIZE}, got {result.shape[2]}"
        )
