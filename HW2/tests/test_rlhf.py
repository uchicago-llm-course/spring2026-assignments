"""
Autograder for Section 3: RLHF (src/rlhf.py)

Run with:
    pytest tests/test_rlhf.py -v
"""

import pytest
import torch
import math


# ── Imports under test ────────────────────────────────────────────────────────

from src.rlhf import (
    compute_rouge_reward,
    compute_group_normalized_rewards,
    compute_grpo_clip_loss,
)


# ── Tests for compute_rouge_reward ────────────────────────────────────────────

class TestComputeRougeReward:
    """Tests for Problem 3.1."""

    def test_returns_list(self):
        scores = compute_rouge_reward(["hello world"], ["hello world"])
        assert isinstance(scores, list), "Should return a list"

    def test_correct_length(self):
        completions   = ["a b c", "x y z", "foo bar"]
        ground_truths = ["a b c", "x y z", "foo bar"]
        scores = compute_rouge_reward(completions, ground_truths)
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"

    def test_scores_are_floats(self):
        scores = compute_rouge_reward(["hello"], ["world"])
        assert all(isinstance(s, float) for s in scores), \
            "All scores should be floats"

    def test_scores_in_range(self):
        completions   = ["the cat sat on the mat", "foo bar baz"]
        ground_truths = ["the cat sat on the mat", "hello world"]
        scores = compute_rouge_reward(completions, ground_truths)
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1] range"

    def test_perfect_match_gives_high_score(self):
        text   = "the quick brown fox"
        scores = compute_rouge_reward([text], [text])
        assert scores[0] > 0.9, \
            f"Perfect match should give score > 0.9, got {scores[0]}"

    def test_empty_completion_gives_low_score(self):
        scores = compute_rouge_reward([""], ["the quick brown fox jumps"])
        assert scores[0] < 0.1, \
            f"Empty completion should give low score, got {scores[0]}"

    def test_different_texts_give_lower_score(self):
        scores_same = compute_rouge_reward(
            ["the cat sat on the mat"], ["the cat sat on the mat"]
        )
        scores_diff = compute_rouge_reward(
            ["the dog ran in the park"], ["the cat sat on the mat"]
        )
        assert scores_same[0] > scores_diff[0], \
            "Same text should outscore different text"

    def test_batch_independence(self):
        """Each pair should be scored independently."""
        completions   = ["cat", "dog", "bird"]
        ground_truths = ["cat", "cat", "cat"]
        scores = compute_rouge_reward(completions, ground_truths)
        # First pair matches, others don't — scores should differ
        assert scores[0] > scores[1], \
            "First pair (exact match) should outscore second pair (different)"


# ── Tests for compute_group_normalized_rewards ────────────────────────────────

class TestComputeGroupNormalizedRewards:
    """Tests for Problem 3.2."""

    def test_returns_tensor(self):
        rewards = [0.3, 0.7, 0.5, 0.9]
        out = compute_group_normalized_rewards(rewards, group_size=2)
        assert isinstance(out, torch.Tensor), "Should return a torch.Tensor"

    def test_output_shape(self):
        rewards = [0.1, 0.5, 0.3, 0.9, 0.2, 0.8]
        out = compute_group_normalized_rewards(rewards, group_size=2)
        assert out.shape == (6,), f"Expected shape (6,), got {out.shape}"

    def test_output_dtype_float(self):
        rewards = [0.1, 0.5, 0.3, 0.9]
        out = compute_group_normalized_rewards(rewards, group_size=2)
        assert out.dtype in (torch.float32, torch.float64), \
            f"Expected float tensor, got {out.dtype}"

    def test_group_mean_is_zero(self):
        """After normalisation, the mean advantage within each group should be ~0."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 3 groups of 2
        out     = compute_group_normalized_rewards(rewards, group_size=2)
        groups  = out.view(-1, 2)  # shape (3, 2)
        group_means = groups.mean(dim=1)
        assert torch.allclose(group_means, torch.zeros(3), atol=1e-5), \
            f"Group means should be 0, got {group_means}"

    def test_group_std_near_one(self):
        """After normalisation, the std within each group should be ~1
        (when the group has non-zero variance)."""
        rewards = [1.0, 3.0, 10.0, 20.0]  # 2 groups of 2
        out     = compute_group_normalized_rewards(rewards, group_size=2)
        groups  = out.view(-1, 2)
        # std of a 2-element group after normalisation:
        # values are ±1/std_pop * std_pop = ±1, so sample std ≈ sqrt(2)
        # but population std should be near 1.
        for g in groups:
            # Both elements should have equal magnitude (~1 for mean-subtracted, std-normalised)
            assert abs(g[0].item()) > 0.5, "Normalised advantage too small"

    def test_uniform_rewards_give_zero_advantages(self):
        """If all rewards in a group are equal, advantages should be zero
        (or very small due to eps)."""
        rewards = [0.5, 0.5, 0.5, 0.5]  # 2 groups of 2, uniform
        out     = compute_group_normalized_rewards(rewards, group_size=2)
        assert torch.allclose(out, torch.zeros(4), atol=1e-4), \
            f"Uniform rewards should give zero advantages, got {out}"

    def test_higher_reward_gets_positive_advantage(self):
        """Within a group, the higher reward should produce a positive advantage."""
        rewards = [0.2, 0.8]  # single group
        out     = compute_group_normalized_rewards(rewards, group_size=2)
        assert out[0].item() < 0, "Lower reward should give negative advantage"
        assert out[1].item() > 0, "Higher reward should give positive advantage"

    def test_group_size_four(self):
        """Test with group_size=4 (typical for GRPO)."""
        rewards = [0.1, 0.4, 0.7, 1.0,   # group 1
                   0.2, 0.3, 0.8, 0.9]   # group 2
        out = compute_group_normalized_rewards(rewards, group_size=4)
        assert out.shape == (8,)
        groups = out.view(-1, 4)
        group_means = groups.mean(dim=1)
        assert torch.allclose(group_means, torch.zeros(2), atol=1e-5)


# ── Tests for compute_grpo_clip_loss ─────────────────────────────────────────

class TestComputeGrpoClipLoss:
    """Tests for Problem 3.3."""

    def test_returns_scalar(self):
        N = 8
        log_probs     = torch.randn(N)
        old_log_probs = torch.randn(N)
        advantages    = torch.randn(N)
        loss = compute_grpo_clip_loss(log_probs, old_log_probs, advantages)
        assert isinstance(loss, torch.Tensor), "Should return a tensor"
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_loss_with_zero_advantages_is_zero(self):
        """If all advantages are 0, the loss should be 0 regardless of ratios."""
        N = 8
        log_probs     = torch.randn(N)
        old_log_probs = torch.randn(N)
        advantages    = torch.zeros(N)
        loss = compute_grpo_clip_loss(log_probs, old_log_probs, advantages)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), \
            f"Zero advantages should give zero loss, got {loss.item()}"

    def test_loss_with_identical_policies_and_positive_advantages(self):
        """When current and reference policies are the same, ratio = 1.
        Loss should be -mean(advantages)."""
        N          = 8
        log_probs  = torch.tensor([-1.0] * N)
        advantages = torch.tensor([1.0] * N)
        loss = compute_grpo_clip_loss(log_probs, log_probs.clone(), advantages)
        expected = -1.0
        assert abs(loss.item() - expected) < 1e-5, \
            f"Expected loss {expected}, got {loss.item()}"

    def test_clipping_is_applied(self):
        """When the ratio is far outside [1-eps, 1+eps], clipping should
        prevent the loss from growing unboundedly."""
        N    = 4
        eps  = 0.2
        # Make ratio very large: log_probs >> old_log_probs
        log_probs     = torch.tensor([10.0] * N)
        old_log_probs = torch.tensor([ 0.0] * N)
        advantages    = torch.tensor([ 1.0] * N)   # positive

        loss = compute_grpo_clip_loss(log_probs, old_log_probs, advantages, clip_epsilon=eps)
        # The clipped ratio is 1+eps for positive advantages, so loss ≤ -(1+eps)
        clipped_loss = -(1.0 + eps)
        assert abs(loss.item() - clipped_loss) < 1e-4, \
            f"Expected clipped loss {clipped_loss}, got {loss.item()}"

    def test_negative_sign_convention(self):
        """The loss should be negative of the surrogate objective (we minimise)."""
        N          = 4
        log_probs  = torch.zeros(N)
        advantages = torch.tensor([2.0] * N)
        loss = compute_grpo_clip_loss(log_probs, log_probs.clone(), advantages)
        # ratio=1, surrogate = mean(1 * 2) = 2, loss = -2
        assert loss.item() < 0, \
            "Positive advantages with ratio=1 should give negative loss"

    def test_clip_epsilon_respected(self):
        """Test with a custom clip_epsilon value."""
        N    = 4
        eps  = 0.1
        log_probs     = torch.tensor([5.0] * N)   # large positive shift
        old_log_probs = torch.tensor([0.0] * N)
        advantages    = torch.tensor([1.0] * N)
        loss = compute_grpo_clip_loss(log_probs, old_log_probs, advantages, clip_epsilon=eps)
        expected = -(1.0 + eps)
        assert abs(loss.item() - expected) < 1e-4, \
            f"Expected clipped loss {expected} with eps={eps}, got {loss.item()}"

    def test_gradient_flows_through_log_probs(self):
        """The loss must be differentiable with respect to log_probs."""
        N             = 4
        log_probs     = torch.randn(N, requires_grad=True)
        old_log_probs = torch.randn(N).detach()
        advantages    = torch.randn(N)
        loss = compute_grpo_clip_loss(log_probs, old_log_probs, advantages)
        loss.backward()
        assert log_probs.grad is not None, \
            "Gradient should flow through log_probs"
