"""
Autograder for Section 2: Instruction Tuning (src/instruction_tuning.py)

Run with:
    pytest tests/test_instruction_tuning.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock


# ── Mock tokenizer (same as in test_sft.py) ───────────────────────────────────

class _MockTokenizer:
    pad_token    = "<pad>"
    eos_token    = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text):
        return [ord(c) % 200 + 2 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((i - 2) % 200 + 32) for i in ids if i > 1)

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)
        if return_tensors == "pt":
            t = torch.tensor([ids], dtype=torch.long)
            return MagicMock(input_ids=t)
        return {"input_ids": ids}


# ── Imports under test ────────────────────────────────────────────────────────

from src.instruction_tuning import (
    format_instruction,
    tokenize_for_instruction_tuning,
    INSTRUCTION_TEMPLATE,
)


# ── Tests for format_instruction ──────────────────────────────────────────────

class TestFormatInstruction:
    """Tests for Problem 2.1."""

    ARTICLE = "Scientists discover water on Mars."
    SUMMARY = "Water found on Mars."

    def test_without_summary_returns_template(self):
        result = format_instruction(self.ARTICLE)
        assert isinstance(result, str)
        expected = INSTRUCTION_TEMPLATE.format(article=self.ARTICLE)
        assert result == expected, \
            f"Expected:\n{expected!r}\n\nGot:\n{result!r}"

    def test_with_summary_appends_summary(self):
        result = format_instruction(self.ARTICLE, self.SUMMARY)
        assert isinstance(result, str)
        expected_prefix = INSTRUCTION_TEMPLATE.format(article=self.ARTICLE)
        assert result.startswith(expected_prefix), \
            "Result should start with the instruction template"
        assert self.SUMMARY in result, \
            "Summary should appear in the result when provided"

    def test_with_summary_has_space_before_summary(self):
        result = format_instruction(self.ARTICLE, self.SUMMARY)
        prefix = INSTRUCTION_TEMPLATE.format(article=self.ARTICLE)
        suffix = result[len(prefix):]
        assert suffix.startswith(" "), \
            "There should be a space between the template and the summary"

    def test_article_embedded_in_template(self):
        result = format_instruction(self.ARTICLE)
        assert self.ARTICLE in result, "Article should appear in the formatted string"

    def test_summary_key_present(self):
        result = format_instruction(self.ARTICLE)
        assert "Summary:" in result, "'Summary:' should appear in the template"


# ── Tests for tokenize_for_instruction_tuning ─────────────────────────────────

class TestTokenizeForInstructionTuning:
    """Tests for Problem 2.2."""

    TOKENIZER  = _MockTokenizer()
    ARTICLE    = "The quick brown fox jumps."
    SUMMARY    = "Fox jumps."
    MAX_LENGTH = 128

    def _get_output(self, article=None, summary=None, max_length=None):
        return tokenize_for_instruction_tuning(
            article    or self.ARTICLE,
            summary    or self.SUMMARY,
            self.TOKENIZER,
            max_length or self.MAX_LENGTH,
        )

    def test_returns_dict_with_required_keys(self):
        out = self._get_output()
        assert isinstance(out, dict)
        for key in ("input_ids", "attention_mask", "labels"):
            assert key in out, f"Missing key: {key}"

    def test_output_shape(self):
        out = self._get_output()
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].shape == (self.MAX_LENGTH,), \
                f"{key} has shape {out[key].shape}, expected ({self.MAX_LENGTH},)"

    def test_output_dtype_is_long(self):
        out = self._get_output()
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].dtype == torch.long

    def test_instruction_tokens_are_masked(self):
        """All instruction/article tokens should have labels == -100."""
        out    = self._get_output()
        # The prompt ends with "Summary:" — everything before the summary in
        # the labels must be -100.  We verify that at least SOME positions at
        # the start are masked (the instruction template itself).
        prompt = INSTRUCTION_TEMPLATE.format(article=self.ARTICLE)
        num_prompt_tokens = len(self.TOKENIZER.encode(prompt))
        # Allow for slight tokenization boundary differences of ≤1 token
        masked_count = (out["labels"][:num_prompt_tokens] == -100).sum().item()
        assert masked_count >= num_prompt_tokens - 1, \
            f"Expected first {num_prompt_tokens} labels to be -100, " \
            f"but only {masked_count} are"

    def test_summary_tokens_are_not_masked(self):
        """At least one position after the prompt prefix should have a non-(-100) label."""
        out    = self._get_output()
        prompt = INSTRUCTION_TEMPLATE.format(article=self.ARTICLE)
        num_prompt_tokens = len(self.TOKENIZER.encode(prompt))
        # Check that there are unmasked positions after the prompt
        real_label_positions = (out["labels"][num_prompt_tokens:] != -100).sum().item()
        assert real_label_positions > 0, \
            "Summary tokens should NOT be masked in labels"

    def test_padding_positions_are_masked(self):
        out = self._get_output()
        pad_positions = (out["input_ids"] == self.TOKENIZER.pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            assert (out["labels"][pad_positions] == -100).all(), \
                "Padding positions should have labels == -100"

    def test_padding_has_zero_attention(self):
        out = self._get_output()
        pad_positions = (out["input_ids"] == self.TOKENIZER.pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            assert (out["attention_mask"][pad_positions] == 0).all(), \
                "Padded tokens should have attention_mask == 0"

    def test_more_positions_masked_than_sft(self):
        """IT should mask strictly more positions than SFT (instruction tokens extra)."""
        from src.sft import tokenize_for_sft
        sft_out = tokenize_for_sft(
            self.ARTICLE, self.SUMMARY, self.TOKENIZER, self.MAX_LENGTH
        )
        it_out  = self._get_output()

        sft_masked = (sft_out["labels"] == -100).sum().item()
        it_masked  = (it_out["labels"]  == -100).sum().item()
        assert it_masked > sft_masked, \
            f"IT masking ({it_masked}) should be stricter than SFT masking ({sft_masked})"

    def test_truncation_keeps_summary(self):
        """Even with a very long article, summary tokens must still appear."""
        long_article = "word " * 300
        summary      = "short"
        out = tokenize_for_instruction_tuning(
            long_article, summary, self.TOKENIZER, 128
        )
        summary_ids = self.TOKENIZER.encode(summary)
        ids_list    = out["input_ids"].tolist()
        found = any(
            ids_list[i:i + len(summary_ids)] == summary_ids
            for i in range(len(ids_list) - len(summary_ids) + 1)
        )
        assert found, "Summary tokens not found after truncation"

    def test_format_and_tokenize_consistent(self):
        """The full sequence should start with the formatted instruction prompt."""
        out    = self._get_output()
        prompt = INSTRUCTION_TEMPLATE.format(article=self.ARTICLE)
        # The first token of input_ids should match the first token of the prompt
        first_prompt_tok = self.TOKENIZER.encode(prompt)[0]
        assert out["input_ids"][0].item() == first_prompt_tok, \
            "First token should be the start of the instruction template"
