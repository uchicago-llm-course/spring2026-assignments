"""
Autograder for Section 1: Supervised Fine-Tuning (src/sft.py)

Run with:
    pytest tests/test_sft.py -v
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


# ── Mock tokenizer ────────────────────────────────────────────────────────────

class _MockTokenizer:
    """Minimal tokenizer that encodes text as character-level token ids."""

    pad_token    = "<pad>"
    eos_token    = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text):
        # Produce a deterministic integer list from the string
        return [ord(c) % 200 + 2 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((i - 2) % 200 + 32) for i in ids if i > 1)

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)
        if return_tensors == "pt":
            t = torch.tensor([ids], dtype=torch.long)
            return MagicMock(input_ids=t)
        return {"input_ids": ids}


# ── Mock model ────────────────────────────────────────────────────────────────

class _MockCausalLM(nn.Module):
    """Minimal CausalLM that returns a fixed loss and fake logits."""

    def __init__(self, vocab_size: int = 202):
        super().__init__()
        self.vocab_size = vocab_size
        # A learnable parameter so optimisers can step
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        logits = torch.randn(B, T, self.vocab_size, requires_grad=True)
        loss   = None
        if labels is not None:
            # Compute real cross-entropy so that the loss is a differentiable scalar
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
        return MagicMock(logits=logits, loss=loss)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=5, **kwargs):
        B, T = input_ids.shape
        new_tokens = torch.randint(2, 100, (B, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)


# ── Imports under test ────────────────────────────────────────────────────────

from src.sft import tokenize_for_sft, compute_sft_loss, SEPARATOR


# ── Tests for tokenize_for_sft ────────────────────────────────────────────────

class TestTokenizeForSFT:
    """Tests for Problem 1.1."""

    TOKENIZER  = _MockTokenizer()
    ARTICLE    = "The quick brown fox jumps."
    SUMMARY    = "Fox jumps."
    MAX_LENGTH = 64

    def _get_output(self):
        return tokenize_for_sft(
            self.ARTICLE, self.SUMMARY, self.TOKENIZER, self.MAX_LENGTH
        )

    def test_returns_dict_with_required_keys(self):
        out = self._get_output()
        assert isinstance(out, dict)
        assert "input_ids" in out
        assert "attention_mask" in out
        assert "labels" in out

    def test_output_shape(self):
        out = self._get_output()
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].shape == (self.MAX_LENGTH,), \
                f"{key} has wrong shape: {out[key].shape}"

    def test_output_dtype_is_long(self):
        out = self._get_output()
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].dtype == torch.long, \
                f"{key} has wrong dtype: {out[key].dtype}"

    def test_attention_mask_binary(self):
        out = self._get_output()
        unique = set(out["attention_mask"].tolist())
        assert unique.issubset({0, 1}), "attention_mask should only contain 0 and 1"

    def test_padding_positions_have_zero_attention(self):
        out = self._get_output()
        # Padding positions must have attention_mask == 0
        pad_positions = (out["input_ids"] == self.TOKENIZER.pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            assert (out["attention_mask"][pad_positions] == 0).all(), \
                "Padded tokens should have attention_mask == 0"

    def test_padding_positions_have_minus_100_labels(self):
        out = self._get_output()
        # All padding positions must have labels == -100
        pad_positions = (out["input_ids"] == self.TOKENIZER.pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            assert (out["labels"][pad_positions] == -100).all(), \
                "Padded positions should have labels == -100"

    def test_labels_equal_input_ids_for_real_tokens(self):
        """In the naive SFT baseline, labels should match input_ids for all non-padded positions."""
        out = self._get_output()
        real = out["attention_mask"].bool()
        assert (out["labels"][real] == out["input_ids"][real]).all(), \
            "labels should equal input_ids at non-padded positions"

    def test_separator_present_in_sequence(self):
        """The SEPARATOR string should appear somewhere in the encoded sequence."""
        sep_ids = self.TOKENIZER.encode(SEPARATOR)
        ids_list = out = self._get_output()["input_ids"].tolist()
        # Check that sep_ids appear as a contiguous sub-sequence
        found = any(
            ids_list[i:i + len(sep_ids)] == sep_ids
            for i in range(len(ids_list) - len(sep_ids) + 1)
        )
        assert found, "SEPARATOR token ids not found in input_ids"

    def test_truncation_respects_max_length(self):
        """A very long article should be truncated, not expanded beyond max_length."""
        long_article = "word " * 300
        out = tokenize_for_sft(long_article, self.SUMMARY, self.TOKENIZER, 64)
        assert out["input_ids"].shape == (64,)

    def test_summary_tokens_present_after_truncation(self):
        """Summary tokens must still appear even when the article is truncated."""
        long_article = "word " * 300
        summary      = "short"
        out = tokenize_for_sft(long_article, summary, self.TOKENIZER, 64)
        summary_ids = self.TOKENIZER.encode(summary)
        ids_list = out["input_ids"].tolist()
        found = any(
            ids_list[i:i + len(summary_ids)] == summary_ids
            for i in range(len(ids_list) - len(summary_ids) + 1)
        )
        assert found, "Summary tokens not found after truncation"


# ── Tests for compute_sft_loss ────────────────────────────────────────────────

class TestComputeSFTLoss:
    """Tests for Problem 1.2."""

    def test_returns_scalar_tensor(self):
        model = _MockCausalLM()
        B, T  = 2, 16
        input_ids      = torch.randint(2, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels         = input_ids.clone()

        loss = compute_sft_loss(model, input_ids, attention_mask, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_positive(self):
        model = _MockCausalLM()
        B, T  = 2, 16
        input_ids      = torch.randint(2, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels         = input_ids.clone()

        loss = compute_sft_loss(model, input_ids, attention_mask, labels)
        assert loss.item() > 0, "Loss should be positive for random model"

    def test_loss_zero_when_all_masked(self):
        """If all label positions are -100, the loss should be 0 (or very small)."""
        model = _MockCausalLM()
        B, T  = 2, 16
        input_ids      = torch.randint(2, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels         = torch.full((B, T), -100, dtype=torch.long)

        loss = compute_sft_loss(model, input_ids, attention_mask, labels)
        # When all labels are masked, HuggingFace returns 0 loss
        assert loss.item() == 0.0 or torch.isnan(loss), \
            "All-masked labels should give 0 (or NaN) loss"

    def test_loss_has_gradient(self):
        """Loss must be differentiable so backward() can update model weights."""
        model = _MockCausalLM()
        B, T  = 2, 16
        input_ids      = torch.randint(2, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels         = input_ids.clone()

        loss = compute_sft_loss(model, input_ids, attention_mask, labels)
        assert loss.requires_grad, "Loss should require grad for backprop"
