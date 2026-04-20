"""
Section 1: Distributed Training
Implement ring_allreduce and simulate_ddp_step.
"""

import torch
import torch.nn as nn
from typing import List, Callable, Tuple
import copy


def ring_allreduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Simulate ring allreduce across N GPUs.

    Args:
        tensors: list of N tensors (one per GPU), all the same shape.

    Returns:
        list of N tensors, each containing the averaged gradient.
        All output tensors must be identical (within atol=1e-6).
    """
    N = len(tensors)
    assert N >= 1, "Need at least one tensor"
    if N == 1:
        return [tensors[0].clone()]

    # Work on clones so we don't modify the inputs
    bufs = [t.clone() for t in tensors]

    # Step 1: Split each tensor into N equal-sized chunks
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Split each buffer into N chunks")
    # ─── END YOUR CODE ────────────────────────────────────────────────

    # Step 2: Scatter-reduce (N-1 rounds)
    # In round 0, GPU i sends chunk i. In subsequent rounds, each GPU
    # forwards the chunk it just received. The receiver ADDS the incoming
    # values to its own copy.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Implement the scatter-reduce phase")
    # ─── END YOUR CODE ────────────────────────────────────────────────

    # Step 3: Allgather (N-1 rounds)
    # Same ring pattern, but receivers COPY instead of adding.
    # After this phase every GPU has all N fully-reduced chunks.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Implement the allgather phase")
    # ─── END YOUR CODE ────────────────────────────────────────────────

    # Step 4: Average — divide all chunks by N, then reconstruct
    # full tensors by concatenating chunks.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Divide by N and reconstruct full tensors")
    # ─── END YOUR CODE ────────────────────────────────────────────────


def simulate_ddp_step(
    model_fn: Callable[[], nn.Module],
    optimizer_fn: Callable[[any], torch.optim.Optimizer],
    data_shards: List[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable,
) -> List[nn.Module]:
    """
    Simulate one DDP training step across N GPUs.

    Args:
        model_fn: callable that returns a freshly initialized model.
        optimizer_fn: callable that takes model parameters and returns an optimizer.
        data_shards: list of N (input, target) tuples, one per GPU.
        loss_fn: loss function, e.g. nn.CrossEntropyLoss().

    Returns:
        list of N models, all with identical parameters after the step.
    """
    N = len(data_shards)

    # Step 1: Create N identical (model, optimizer) pairs.
    # All models must start with the SAME weights.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Create N model/optimizer pairs with identical initial weights")
    # ─── END YOUR CODE ────────────────────────────────────────────────

    # Step 2: Forward + backward on each model with its own data shard.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Run forward + backward on each model")
    # ─── END YOUR CODE ────────────────────────────────────────────────

    # Step 3: Collect gradients from all models (flatten into one tensor
    # per model), call ring_allreduce, then load averaged gradients back.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("Collect gradients, allreduce, load back")
    # ─── END YOUR CODE ────────────────────────────────────────────────

    # Step 4: Call optimizer.step() on each model.
    # ─── YOUR CODE HERE ───────────────────────────────────────────────
    raise NotImplementedError("optimizer.step() on each model")
    # ─── END YOUR CODE ────────────────────────────────────────────────
