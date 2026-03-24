"""
CMSC 25750 HW1 – Section 4: Budget-Constrained Multi-Layer KV Cache

Implement the four methods marked  # YOUR CODE HERE  inside MultiLayerKVCache.
The generation functions (generate_reference, generate_with_budget, benchmark)
are provided — do not modify them.
"""

from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class MultiLayerKVCache:
    """
    A budget-constrained KV cache shared across all transformer layers.

    Internal storage per layer: (1, num_heads, T_cached, d_head)
    — identical to HuggingFace's past_key_values tensor format so that
    to_hf_format() requires no reshaping.

    Workflow:
        cache = MultiLayerKVCache(num_layers, total_budget, num_heads, d_head)
        cache.allocate_uniform()          # or allocate_priority(scores)
        out = generate_with_budget(model, tokenizer, prompt, cache, N)
    """

    def __init__(
        self,
        num_layers: int,
        total_budget: int,
        num_heads: int,
        d_head: int,
        device: str = "cpu",
    ):
        self.num_layers   = num_layers
        self.total_budget = total_budget
        self.num_heads    = num_heads
        self.d_head       = d_head
        self.device       = device

        # Per-layer token budget  (set by allocate_*)
        self.layer_budgets: List[int] = [0] * num_layers

        # Per-layer K/V storage: list of (k_tensor, v_tensor) or None
        # Each tensor shape: (1, num_heads, T_cached, d_head)
        self._cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = \
            [None] * num_layers

        # Stats
        self.eviction_counts: List[int] = [0] * num_layers

    # PROBLEM 4.1
    def allocate_uniform(self) -> None:
        """
        Assign each layer an equal token budget.

        Set self.layer_budgets[i] = total_budget // num_layers  for all i.
        (Use floor division; the remainder is intentionally unused.)
        """
        # YOUR CODE HERE

    # PROBLEM 4.2 
    def allocate_priority(self, layer_scores: List[float]) -> None:
        """
        Assign each layer a budget proportional to its score.

        A higher score means the layer is more "important" and gets more tokens.
        The total allocated tokens must not exceed self.total_budget.

        Steps:
          1. Normalise scores so they sum to 1.
          2. layer_budgets[i] = int(normalised_score[i] * total_budget).
          3. Because int() floors, the total may fall slightly short — that is
             acceptable; do NOT round up beyond total_budget.

        Parameters
        ----------
        layer_scores : list of floats, length num_layers
            One importance score per layer (e.g. the rank of the target token
            at that layer from your Section 2 logit-lens results).
        """
        # YOUR CODE HERE

    # PROBLEM 4.3 
    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """
        Append new key/value tensors for `layer_idx` and evict if over budget.

        Parameters
        ----------
        new_k, new_v : (1, num_heads, T_new, d_head)
            New K/V tensors for one or more tokens (T_new ≥ 1).

        Steps:
          1. If self._cache[layer_idx] is None, initialise it with
             (new_k, new_v) directly (move to self.device).
          2. Otherwise concatenate:
               k_combined = torch.cat([existing_k, new_k], dim=2)
             and similarly for v.
          3. If the sequence dimension (dim 2) of k_combined now exceeds
             self.layer_budgets[layer_idx], evict the oldest tokens from
             the front:
               n_evict = k_combined.shape[2] - self.layer_budgets[layer_idx]
               k_combined = k_combined[:, :, n_evict:, :]
             Increment self.eviction_counts[layer_idx] by n_evict.
          4. Store the result back into self._cache[layer_idx].
        """
        # YOUR CODE HERE

    # PROBLEM 4.4 
    def get(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (K, V) for `layer_idx`, or (None, None) if the layer is empty.

        Shape of returned tensors: (1, num_heads, T_cached, d_head).
        """
        # YOUR CODE HERE

    # ── Provided helpers (do not modify) ─────────────────────────────────────

    def tokens_cached(self, layer_idx: int) -> int:
        """Number of tokens currently cached at `layer_idx`."""
        if self._cache[layer_idx] is None:
            return 0
        return self._cache[layer_idx][0].shape[2]

    def to_hf_format(self):
        """
        Convert to HuggingFace past_key_values format.
        For transformers 5.x: returns DynamicCache object.
        For transformers <5.0: returns tuple of (K,V) pairs.
        Returns None if any layer is still empty (prefill not done yet).
        """
        # Check if any layer is empty
        for i in range(self.num_layers):
            if self._cache[i] is None:
                return None

        # Try to use DynamicCache for transformers 5.x compatibility
        try:
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
            for i in range(self.num_layers):
                k, v = self.get(i)
                cache.update(k, v, layer_idx=i)
            return cache
        except ImportError:
            # Fallback for transformers <5.0
            result = []
            for i in range(self.num_layers):
                k, v = self.get(i)
                result.append((k, v))
            return tuple(result)

    def memory_stats(self) -> dict:
        """Report cache utilisation statistics."""
        total_cached = sum(self.tokens_cached(i) for i in range(self.num_layers))
        return {
            "total_budget":       self.total_budget,
            "total_tokens_cached": total_cached,
            "utilization":        total_cached / max(self.total_budget, 1),
            "per_layer_cached":   [self.tokens_cached(i) for i in range(self.num_layers)],
            "per_layer_budget":   list(self.layer_budgets),
            "eviction_counts":    list(self.eviction_counts),
        }

    def reset(self) -> None:
        """Clear all cached tensors (keeps budgets intact)."""
        self._cache = [None] * self.num_layers
        self.eviction_counts = [0] * self.num_layers


# ─────────────────────────────────────────────────────────────────────────────
# Generation functions  (provided — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_reference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """
    Reference generation using HuggingFace's built-in unlimited KV cache.
    Use this as the quality baseline for comparison.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    T_prompt  = input_ids.shape[1]

    outputs   = model(input_ids, use_cache=True)
    past_kv   = outputs.past_key_values
    next_tok  = outputs.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_tok.item()]

    for _ in range(max_new_tokens - 1):
        outputs  = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        past_kv  = outputs.past_key_values
        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_with_budget(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    cache: MultiLayerKVCache,
    max_new_tokens: int = 50,
) -> str:
    """
    Generate using a pre-configured MultiLayerKVCache.

    Call cache.allocate_uniform() or cache.allocate_priority() before this.
    The cache is mutated in-place; call cache.reset() to reuse it.

    Algorithm:
      Prefill  — run the full prompt through the model once.  Pass all
                 per-layer K/V tensors to cache.update().
      Decode   — at each step, pass only the new single token together with
                 cache.to_hf_format() as past_key_values.  The model returns
                 updated past_key_values containing the full history; extract
                 only the last token's K/V and call cache.update() with it.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Prefill
    outputs = model(input_ids, use_cache=True)
    for layer_idx, layer_cache in enumerate(outputs.past_key_values):
        k, v = layer_cache[0], layer_cache[1]  # Handle both (k,v) tuple and (k,v,_) from transformers 5.x
        cache.update(layer_idx, k, v)

    next_tok  = outputs.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_tok.item()]

    # Decode loop
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_tok,
            past_key_values=cache.to_hf_format(),
            use_cache=True,
        )
        # The returned past_key_values include ALL tokens (cache + new).
        # Extract only the last position's K/V (the newly generated token).
        for layer_idx, layer_cache in enumerate(outputs.past_key_values):
            full_k, full_v = layer_cache[0], layer_cache[1]  # Handle both (k,v) tuple and (k,v,_) from transformers 5.x
            cache.update(layer_idx, full_k[:, :, -1:, :], full_v[:, :, -1:, :])

        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    return tokenizer.decode(generated, skip_special_tokens=True)


def benchmark(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    total_budgets: List[int],
    layer_scores: Optional[List[float]] = None,
    max_new_tokens: int = 50,
    num_runs: int = 2,
) -> dict:
    """
    Run generate_with_budget at multiple budget levels under both policies.

    Returns a dict mapping budget → {"uniform": text, "priority": text,
    "uniform_stats": {...}, "priority_stats": {...}}.
    If layer_scores is None, only uniform allocation is benchmarked.
    """
    L          = model.config.num_hidden_layers
    num_heads  = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    d_head     = model.config.hidden_size // model.config.num_attention_heads
    device_str = str(next(model.parameters()).device)

    results = {}
    for budget in total_budgets:
        entry = {}

        for policy in (["uniform"] if layer_scores is None else ["uniform", "priority"]):
            cache = MultiLayerKVCache(L, budget, num_heads, d_head, device=device_str)
            if policy == "uniform":
                cache.allocate_uniform()
            else:
                cache.allocate_priority(layer_scores)

            with torch.no_grad():
                text = generate_with_budget(model, tokenizer, prompt, cache, max_new_tokens)

            entry[policy]            = text
            entry[f"{policy}_stats"] = cache.memory_stats()

        results[budget] = entry

    return results
