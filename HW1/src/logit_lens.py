"""
CMSC 25750 HW1 – Section 2: Logit Lens

Implement the two functions marked  # YOUR CODE HERE.
The notebook (hw1.ipynb) imports these and renders the visualizations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def get_logit_lens_predictions(prompt: str, model) -> Tuple:
    """
    PROBLEM 2.1
    Run the logit lens for `prompt` using a nnsight LanguageModel.

    At every transformer layer, project the residual stream through the
    model's final layer norm and unembedding matrix to obtain a full
    vocabulary probability distribution.

    Steps:
      1. Open a model.trace() context and invoke the prompt.
      2. Save input_ids via  invoker.inputs[1]["input_ids"].save().
      3. Loop over model.gpt_neox.layers.  For each layer:
           a. normed = model.gpt_neox.final_layer_norm(layer.output[0])
           b. logits = model.embed_out(normed)
           c. probs  = F.softmax(logits, dim=-1).save()
           d. Append probs to probs_layers.
      4. Stack all saved tensors:
           all_probs = torch.stack(probs_layers)   # (num_layers, seq_len, vocab)
      5. max_probs, token_ids = all_probs.max(dim=-1)
      6. Decode top_tokens (list[list[str]]) and input_words (list[str]).

    nnsight tracing pattern:
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            input_ids = invoker.inputs[1]["input_ids"].save()
            for layer in model.gpt_neox.layers:
                ...

    Returns
    -------
    max_probs  : Tensor (num_layers, seq_len)             max prob per cell
    top_tokens : list[list[str]] (num_layers, seq_len)    decoded top-1 token
    input_words: list[str]                                decoded input tokens
    all_probs  : Tensor (num_layers, seq_len, vocab_size) full distributions
    """
    probs_layers = []

    # YOUR CODE HERE

    all_probs            = torch.stack(probs_layers)              # (L, T, V)
    max_probs, token_ids = all_probs.max(dim=-1)                  # (L, T)

    top_tokens = [
        [model.tokenizer.decode(t.item()).encode("unicode_escape").decode()
         for t in layer_tokens]
        for layer_tokens in token_ids
    ]
    input_words = [model.tokenizer.decode(t.item()) for t in input_ids[0]]

    return max_probs, top_tokens, input_words, all_probs


def get_token_rank_by_layer(all_probs: torch.Tensor, target_token_id: int) -> list:
    """
    PROBLEM 2.2 
    For each layer, return the 1-indexed rank of `target_token_id` in the
    vocabulary distribution at the LAST token position.
    Rank 1 = highest probability; rank |V| = lowest.

    Parameters
    ----------
    all_probs       : Tensor (num_layers, seq_len, vocab_size)
    target_token_id : integer token ID to track

    Steps (for each layer):
      1. dist = all_probs[layer_idx, -1, :]          last-position distribution
      2. sorted_ids = torch.argsort(dist, descending=True)
      3. rank = position of target_token_id in sorted_ids + 1  (1-indexed)
         Hint: (sorted_ids == target_token_id).nonzero(as_tuple=True)[0].item()

    Returns
    -------
    list of ints, length num_layers
    """
    ranks = []
    for layer_idx in range(all_probs.shape[0]):
        dist = all_probs[layer_idx, -1, :]
        # YOUR CODE HERE
    return ranks
