"""
Section 2: Instruction Tuning

Instruction tuning extends SFT in two key ways:
  1. Inputs are wrapped in a natural-language instruction template, teaching
     the model to follow instructions rather than just continue text.
  2. The cross-entropy loss is computed ONLY on the response (summary) tokens.
     All instruction and article tokens are masked out (set to -100 in labels).
     This response masking focuses the gradient signal entirely on generating
     good summaries, and is the standard approach in modern alignment pipelines.

You will implement the instruction template formatter and the masked tokenizer,
then compare the resulting model against the Section-1 SFT baseline.

Dataset : CNN/DailyMail   (same subset as Section 1)
Model   : gpt2            (same base model as Section 1)
"""

import torch
from torch.utils.data import Dataset
from typing import Optional

# Instruction template

# This template wraps the article in a human-readable instruction.
# The model is trained to produce the text that comes AFTER "Summary:".
INSTRUCTION_TEMPLATE = (
    "Summarize the following article:\n\n"
    "{article}"
    "\n\nSummary:"
)


# Helper functions

class ITDataset(Dataset):
    """Dataset for instruction-tuning on summarization.

    Each item is produced by calling your tokenize_for_instruction_tuning
    implementation.
    """

    def __init__(self, data, tokenizer, max_length: int = 512):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        summary = self.data[idx]["highlights"]
        return tokenize_for_instruction_tuning(
            article, summary, self.tokenizer, self.max_length
        )


# Your implementations

def format_instruction(article: str, summary: Optional[str] = None) -> str:
    """Format an article (and optionally its summary) with the instruction template.
    (Problem 2.1 — 5 pts)

    If summary is None, return:
        INSTRUCTION_TEMPLATE.format(article=article)

    If summary is provided, append a single space followed by the summary:
        INSTRUCTION_TEMPLATE.format(article=article) + " " + summary

    The resulting string is the complete training sequence when summary is given,
    and the generation prompt when summary is None.

    Args:
        article: The source article string.
        summary: Optional summary string to append.

    Returns:
        The formatted string.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def tokenize_for_instruction_tuning(
    article: str,
    summary: str,
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokenize an (article, summary) pair for instruction tuning.  (Problem 2.2 — 8 pts)

    The full input sequence is:

        INSTRUCTION_TEMPLATE.format(article=article) + " " + summary + EOS

    The labels tensor must be -100 for EVERY token that belongs to the
    instruction or article (i.e., everything up to and including "Summary:"),
    and equal to the input_ids value for every summary and EOS token.
    This is the response-masking strategy that focuses the loss on generation.

    Implementation hint:
        1. Tokenize the prompt-only string (no summary) to find the boundary:
               prompt = INSTRUCTION_TEMPLATE.format(article=article)
               prompt_ids = tokenizer.encode(prompt)
           The number of prompt tokens, len(prompt_ids), is the mask boundary.
        2. Tokenize the full sequence (prompt + " " + summary + EOS).
        3. Truncate the full sequence to max_length, trimming the article if
           necessary (i.e., trim from the beginning of the article section, not
           from the summary).  Always keep the entire summary.
        4. Pad to exactly max_length on the right.
        5. Set labels = -100 for positions 0 … (num_prompt_tokens - 1) and for
           all padding positions; leave the remaining positions equal to input_ids.

    Args:
        article:    Source news article string.
        summary:    Human-written summary string.
        tokenizer:  A HuggingFace tokenizer with pad_token set.
        max_length: Desired total length after padding.

    Returns:
        A dict with keys "input_ids", "attention_mask", "labels",
        each a LongTensor of shape (max_length,).
    """
    # YOUR CODE HERE
    raise NotImplementedError
