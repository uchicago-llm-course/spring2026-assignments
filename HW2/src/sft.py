"""
Section 1: Supervised Fine-Tuning (SFT)

In SFT, we fine-tune a pre-trained language model on (input, output) pairs.
The key hyperparameter is which token positions the loss is computed on.
In this section you will implement the naive version: loss on ALL tokens,
both article and summary.  In Section 2 you will switch to response-only masking
and observe the difference.

Dataset : CNN/DailyMail   (abisee/cnn_dailymail)
Model   : gpt2            (117M parameters)
Task    : News summarization  (article → headline-style summary)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import numpy as np
import evaluate  # HuggingFace evaluate library; provides ROUGE metric

# Separator placed between the article and the summary in the concatenated sequence
SEPARATOR = " TL;DR: "


# Helper Functions

def get_data(num_train: int = 5000, num_val: int = 500, seed: int = 42):
    """Load CNN/DailyMail and return (train_split, val_split).

    Uses a fixed random seed for reproducibility.  Set num_train / num_val
    to smaller values for fast local debugging.
    """
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
    train = ds["train"].shuffle(seed=seed).select(range(num_train))
    val   = ds["validation"].shuffle(seed=seed).select(range(num_val))
    return train, val


def get_model_and_tokenizer(model_name: str = "gpt2", device: str = "cpu"):
    """Load a CausalLM and its tokenizer, ensuring a pad token is set.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token by default; reuse EOS token for padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    return model, tokenizer


class SFTDataset(Dataset):
    """PyTorch Dataset that tokenizes (article, summary) pairs for SFT.

    Each item is produced by calling your tokenize_for_sft implementation.
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
        return tokenize_for_sft(article, summary, self.tokenizer, self.max_length)


def train_epoch(model, dataloader, optimizer, device, max_grad_norm: float = 1.0):
    """Run one epoch of SFT training.

    Returns:
        mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        loss = compute_sft_loss(model, input_ids, attention_mask, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Your implementations

def tokenize_for_sft(
    article: str,
    summary: str,
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokenize an (article, summary) pair for SFT.  (Problem 1.1 — 6 pts)

    Concatenate the article and the summary with SEPARATOR between them,
    appending the tokenizer's EOS token at the very end:

        article  +  SEPARATOR  +  summary  +  EOS

    The labels tensor must equal the input_ids tensor (loss is computed on
    ALL positions — this is the naive SFT baseline you will analyse in 1.4(c)).

    Truncation strategy: if the combined token count exceeds max_length,
    trim the article from the right while always keeping the full summary.
    Pad shorter sequences on the right to exactly max_length tokens using
    the tokenizer's pad token id.  Set attention_mask to 0 for padded positions
    and labels to -100 for padded positions (padding tokens are never supervised).

    Args:
        article:    Source news article string.
        summary:    Human-written summary string.
        tokenizer:  A HuggingFace tokenizer with pad_token set.
        max_length: Maximum (and minimum, after padding) token length.

    Returns:
        A dict with three keys, each mapping to a LongTensor of shape (max_length,):
          "input_ids"      — token ids
          "attention_mask" — 1 for real tokens, 0 for padding
          "labels"         — same as input_ids for real tokens, -100 for padding
    """
    # YOUR CODE HERE
    raise NotImplementedError


def compute_sft_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-entropy loss for a SFT batch.  (Problem 1.2 — 6 pts)

    Pass input_ids, attention_mask, and labels through the model.  The model
    automatically ignores positions where labels == -100.

    Args:
        model:          A HuggingFace CausalLM (e.g., GPT-2).
        input_ids:      LongTensor of shape (B, T).
        attention_mask: LongTensor of shape (B, T).
        labels:         LongTensor of shape (B, T); -100 positions are ignored.

    Returns:
        Scalar loss tensor (the model's built-in cross-entropy over valid positions).
    """
    # YOUR CODE HERE
    raise NotImplementedError


def evaluate_rouge(
    model: nn.Module,
    tokenizer,
    articles: list,
    ground_truth_summaries: list,
    device,
    max_new_tokens: int = 80,
    num_examples: int = 50,
) -> dict:
    """Generate summaries and compute ROUGE-1 F1.  (Problem 1.3 — 6 pts)

    For each of the first num_examples articles, feed the article followed by
    SEPARATOR to the model (do NOT include the summary in the prompt) and
    generate up to max_new_tokens new tokens.  Decode the newly generated
    tokens to obtain the predicted summary.

    Use greedy decoding (do_sample=False).  Set the model to eval mode and
    use torch.no_grad().

    Args:
        model:                  A fine-tuned (or base) CausalLM.
        tokenizer:              Corresponding tokenizer.
        articles:               List of article strings.
        ground_truth_summaries: Corresponding reference summaries.
        device:                 torch.device for inference.
        max_new_tokens:         Max tokens to generate per example.
        num_examples:           Number of examples to evaluate (use the first N).

    Returns:
        A dict with:
          "rouge1"       — mean ROUGE-1 F1 score (float, range 0–1)
          "predictions"  — list of generated summary strings
          "references"   — list of ground-truth summary strings
    """
    # YOUR CODE HERE
    raise NotImplementedError
