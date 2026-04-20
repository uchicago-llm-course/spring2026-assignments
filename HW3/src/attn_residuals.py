"""
Section 2: Attention Residuals
Implement prenorm_residual_layer, full_attn_res, and block_attn_res.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def prenorm_residual_layer(
    h: torch.Tensor,
    f: nn.Module,
    norm: nn.Module,
) -> torch.Tensor:
    """
    Apply a single PreNorm residual update: h -> h + f(norm(h)).

    Args:
        h: input tensor of shape (B, T, d).
        f: sublayer module (e.g. attention or MLP block).
        norm: normalization module (e.g. RMSNorm).

    Returns:
        Updated tensor of shape (B, T, d).
    """
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Implement PreNorm residual: h + f(norm(h))")
    # ─── END YOUR CODE ────────────────────────────────────────────────


def full_attn_res(
    prev_outputs: List[torch.Tensor],
    query: torch.Tensor,
    norm: nn.Module,
) -> torch.Tensor:
    """
    Compute Full Attention Residuals: depth-wise softmax attention over
    all previous layer outputs using a learnable pseudo-query.

    Args:
        prev_outputs: list of l tensors, each of shape (B, T, d),
                      holding v_0, ..., v_{l-1} (previous layer outputs).
        query: learnable pseudo-query w_l of shape (d,).
        norm: RMSNorm module applied to the keys before scoring.

    Returns:
        h_l of shape (B, T, d): the input to layer l, computed as
        sum_i alpha_{i->l} * v_i, where alpha is the softmax over
        depth of w_l^T @ RMSNorm(k_i) for each token.

    Notes:
        - Keys equal values: k_i = v_i.
        - The softmax is over the depth dimension (not sequence).
        - The attention weights are per-token (each token has its own
          keys), but the query is shared across tokens.
        - Implement WITHOUT an explicit Python loop over the l previous
          layers -- stack them and use einsum + softmax.
    """
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Implement Full AttnRes (depth-wise softmax attention)")
    # ─── END YOUR CODE ────────────────────────────────────────────────


def block_attn_res(
    block_summaries: List[torch.Tensor],
    partial_block: torch.Tensor,
    query: torch.Tensor,
    proj: nn.Module,
    norm: nn.Module,
) -> torch.Tensor:
    """
    Compute Block Attention Residuals: softmax attention over block-level
    summaries instead of all individual layer outputs.

    Args:
        block_summaries: list of (N-1) completed block summaries
                         b_0, ..., b_{N-2}, each of shape (B, T, d).
        partial_block: running intra-block partial sum b_n^{i-1}
                       for the current (in-progress) block, shape (B, T, d).
        query: learnable pseudo-query w_l of shape (d,).
        proj: nn.Linear projection applied to stacked keys before norm.
        norm: RMSNorm module applied after projection.

    Returns:
        Aggregated output of shape (B, T, d).

    Notes:
        - Stack keys as [b_0, ..., b_{N-2}, partial_block].
        - Project with proj, apply norm, then compute softmax attention
          with query, exactly as in full_attn_res but over block
          summaries rather than individual layer outputs.
    """
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Implement Block AttnRes (attention over block summaries)")
    # ─── END YOUR CODE ────────────────────────────────────────────────
