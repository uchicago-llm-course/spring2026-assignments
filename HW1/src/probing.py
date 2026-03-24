"""
CMSC 25750 HW1 – Section 1: Probing LLMs for Grammatical Knowledge

Implement every block marked  # YOUR CODE HERE.
Run the autograder with:  pytest tests/test_probing.py  (from the HW1/ directory)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _melt_good(example, idx):
    return {"id": example["pair_id"], "label": "1",
            "sentence": example["sentence_good"], "idx": idx}


def _melt_bad(example, idx):
    return {"id": example["pair_id"], "label": "0",
            "sentence": example["sentence_bad"], "idx": idx}


def get_data(seed: int = 42):
    """Load the BLiMP subject-verb agreement dataset and split it.

    Returns
    -------
    (train_data, test_data) : HuggingFace Dataset objects
    """
    dataset = load_dataset("nyu-mll/blimp", "regular_plural_subject_verb_agreement_1")
    raw = dataset["train"]
    good = raw.map(_melt_good, with_indices=True)
    bad  = raw.map(_melt_bad,  with_indices=True)
    data = concatenate_datasets([good, bad])

    # Split data 80 % train / 20 % test.
    # Use .train_test_split(test_size=0.2, seed=seed)
    train_test_data = # YOUR CODE HERE

    print(train_test_data)
    return train_test_data["train"], train_test_data["test"]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def get_model_and_tokenizer(model_name: str, device: torch.device):
    """Load a HuggingFace causal LM with hidden-state output enabled."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emb_dim = model.config.hidden_size
    return model, tokenizer, emb_dim


# ─────────────────────────────────────────────────────────────────────────────
# Probing classifier
# ─────────────────────────────────────────────────────────────────────────────

class ProbingClassifier(nn.Module):
    """A linear probing classifier: hidden-state vector → class logits."""

    def __init__(self, input_dim: int, num_classes: int):
        """
        PROBLEM 1.1 (a)
        Initialize the classifier.

        Store a single Linear layer as  self.linear.
        It should map vectors of size `input_dim` to vectors of size
        `num_classes`.  No activation or bias change is needed.
        """
        super().__init__()
        # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PROBLEM 1.1 (b) 
        Implement the forward pass.

        Pass `x` through self.linear and return the result.
        Do NOT apply softmax yet
        """
        # YOUR CODE HERE
        pass


def build_classifier(
    emb_dim: int,
    num_labels: int,
    device: str = "cpu",
) -> Tuple[ProbingClassifier, nn.Module, torch.optim.Optimizer]:
    """
    PROBLEM 1.2
    Create and return (classifier, criterion, optimizer).

      classifier : ProbingClassifier(emb_dim, num_labels) moved to `device`
      criterion  : nn.CrossEntropyLoss()
      optimizer  : torch.optim.Adam(classifier.parameters(), lr=1e-3)
    """
    # YOUR CODE HERE
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_sentence_repr(
    sentence: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> np.ndarray:
    """Extract hidden states from every transformer layer for one sentence.

    PROBLEM 1.3
    Steps:
      1. Tokenize `sentence` using tokenizer.encode(sentence).
      2. Convert to a [1, seq_len] LongTensor and move to `device`.
      3. Run model(input_ids) under torch.no_grad().
      4. The model output has a `hidden_states` attribute: a tuple of
         (num_layers + 1) tensors of shape (batch_size, seq_len, hidden_size).
         (Index 0 is the embedding layer; indices 1..L are the transformer
         layers.  The model was loaded with output_hidden_states=True.)
      5. For each tensor, squeeze away the batch dimension (dim 0) and
         convert to a CPU numpy array.
      6. Stack all arrays into a single numpy array of shape
         (num_layers + 1, seq_len, hidden_size) and return it.

    Returns
    -------
    np.ndarray of shape (num_layers + 1, seq_len, hidden_size)
    """
    with torch.no_grad():
        # YOUR CODE HERE
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation loops  (do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def train_probe(
    train_representations: list,
    train_labels: list,
    classifier: ProbingClassifier,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """Train the probing classifier. Returns (final_loss, final_accuracy)."""
    n = len(train_representations)
    for epoch in range(num_epochs):
        total_loss, num_correct = 0.0, 0.0
        for start in range(0, n, batch_size):
            batch_repr   = torch.stack(train_representations[start: start + batch_size])
            batch_labels = torch.stack(train_labels[start: start + batch_size])
            optimizer.zero_grad()
            out  = classifier(batch_repr)
            pred = out.argmax(dim=1)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            num_correct += pred.long().eq(batch_labels.long()).cpu().sum().item()
            total_loss  += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={total_loss/n:.4f}  acc={num_correct/n:.4f}")
    return total_loss / n, num_correct / n


def evaluate_probe(
    test_representations: list,
    test_labels: list,
    classifier: ProbingClassifier,
    criterion: nn.Module,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """Evaluate the probing classifier. Returns (loss, accuracy)."""
    n = len(test_representations)
    total_loss, num_correct = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_repr   = torch.stack(test_representations[start: start + batch_size])
            batch_labels = torch.stack(test_labels[start: start + batch_size])
            out  = classifier(batch_repr)
            pred = out.argmax(dim=1)
            num_correct += pred.long().eq(batch_labels.long()).cpu().sum().item()
            total_loss  += criterion(out, batch_labels).item()
    loss = total_loss / n
    acc  = num_correct / n
    print(f"  Test  loss={loss:.4f}  acc={acc:.4f}")
    return loss, acc
