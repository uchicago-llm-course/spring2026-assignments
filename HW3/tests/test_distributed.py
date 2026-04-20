"""
Autograder for Section 1: Distributed Training.
Do not modify this file.
"""

import torch
import torch.nn as nn
import pytest
import copy

from src.distributed import ring_allreduce, simulate_ddp_step


# ── Helpers ──────────────────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """Small model for testing."""
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def naive_allreduce(tensors):
    """Reference: simple average by summing and dividing."""
    avg = torch.stack(tensors).mean(dim=0)
    return [avg.clone() for _ in tensors]


def make_model_fn(seed=42, input_dim=8, hidden_dim=16, output_dim=4):
    """Return a model_fn that always produces the same initial weights."""
    def model_fn():
        torch.manual_seed(seed)
        return TinyMLP(input_dim, hidden_dim, output_dim)
    return model_fn


# ── Tests for ring_allreduce ─────────────────────────────────────────────────

class TestRingAllreduce:
    def test_two_gpus(self):
        """Basic test with N=2."""
        torch.manual_seed(0)
        tensors = [torch.randn(12) for _ in range(2)]
        result = ring_allreduce(tensors)
        expected = naive_allreduce(tensors)

        assert len(result) == 2
        for i in range(2):
            assert torch.allclose(result[i], expected[i], atol=1e-6), \
                f"GPU {i} result doesn't match naive average"

    def test_four_gpus(self):
        """Test with N=4 (matches the exposition example)."""
        torch.manual_seed(1)
        tensors = [torch.randn(20) for _ in range(4)]
        result = ring_allreduce(tensors)
        expected = naive_allreduce(tensors)

        assert len(result) == 4
        for i in range(4):
            assert torch.allclose(result[i], expected[i], atol=1e-6), \
                f"GPU {i} result doesn't match naive average"

    def test_eight_gpus(self):
        """Test with N=8."""
        torch.manual_seed(2)
        tensors = [torch.randn(64) for _ in range(8)]
        result = ring_allreduce(tensors)
        expected = naive_allreduce(tensors)

        assert len(result) == 8
        for i in range(8):
            assert torch.allclose(result[i], expected[i], atol=1e-6), \
                f"GPU {i} result doesn't match naive average"

    def test_all_outputs_identical(self):
        """All GPUs must end up with the exact same tensor."""
        torch.manual_seed(3)
        tensors = [torch.randn(32) for _ in range(5)]
        result = ring_allreduce(tensors)

        for i in range(1, len(result)):
            assert torch.allclose(result[0], result[i], atol=1e-6), \
                f"GPU 0 and GPU {i} have different results"

    def test_single_gpu(self):
        """Edge case: N=1 should return the input unchanged."""
        t = torch.randn(10)
        result = ring_allreduce([t])
        assert len(result) == 1
        assert torch.allclose(result[0], t, atol=1e-6)

    def test_2d_tensors(self):
        """Should work with multi-dimensional tensors."""
        torch.manual_seed(4)
        tensors = [torch.randn(4, 8) for _ in range(3)]
        result = ring_allreduce(tensors)
        expected = naive_allreduce(tensors)

        for i in range(3):
            assert result[i].shape == tensors[0].shape
            assert torch.allclose(result[i], expected[i], atol=1e-6)

    def test_does_not_modify_input(self):
        """Input tensors should not be modified."""
        torch.manual_seed(5)
        tensors = [torch.randn(16) for _ in range(4)]
        originals = [t.clone() for t in tensors]
        ring_allreduce(tensors)

        for i in range(4):
            assert torch.allclose(tensors[i], originals[i], atol=1e-8), \
                f"Input tensor {i} was modified"

    def test_large_tensor(self):
        """Test with a larger tensor to catch chunking edge cases."""
        torch.manual_seed(6)
        tensors = [torch.randn(1000) for _ in range(7)]
        result = ring_allreduce(tensors)
        expected = naive_allreduce(tensors)

        for i in range(7):
            assert torch.allclose(result[i], expected[i], atol=1e-5), \
                f"GPU {i} result doesn't match naive average for large tensor"


# ── Tests for simulate_ddp_step ──────────────────────────────────────────────

class TestSimulateDDPStep:
    def _run_ddp_step(self, N=4, input_dim=8, output_dim=4, seed=42):
        """Helper: run one DDP step and one naive-average reference step."""
        model_fn = make_model_fn(seed=seed, input_dim=input_dim, output_dim=output_dim)
        optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Generate random data shards
        torch.manual_seed(seed + 100)
        data_shards = [
            (torch.randn(4, input_dim), torch.randint(0, output_dim, (4,)))
            for _ in range(N)
        ]

        # Run student's DDP step
        models = simulate_ddp_step(model_fn, optimizer_fn, data_shards, loss_fn)

        # Compute reference: one model, naive gradient average, one step
        ref_model = model_fn()
        ref_optimizer = optimizer_fn(ref_model.parameters())

        # Forward+backward on all shards, accumulate gradients
        ref_optimizer.zero_grad()
        for inp, tgt in data_shards:
            loss = loss_fn(ref_model(inp), tgt)
            loss.backward()
        # Average gradients (backward accumulated the sum, divide by N)
        for p in ref_model.parameters():
            if p.grad is not None:
                p.grad.div_(N)
        ref_optimizer.step()

        return models, ref_model

    def test_all_models_identical(self):
        """All N models must have identical parameters after the step."""
        models, _ = self._run_ddp_step(N=4)
        params_0 = list(models[0].parameters())
        for i in range(1, len(models)):
            params_i = list(models[i].parameters())
            for p0, pi in zip(params_0, params_i):
                assert torch.allclose(p0, pi, atol=1e-6), \
                    f"Model 0 and model {i} have different parameters"

    def test_matches_naive_average(self):
        """DDP step result must match naive gradient averaging."""
        models, ref_model = self._run_ddp_step(N=4)
        for p_ddp, p_ref in zip(models[0].parameters(), ref_model.parameters()):
            assert torch.allclose(p_ddp, p_ref, atol=1e-5), \
                "DDP step doesn't match naive gradient average reference"

    def test_two_gpus(self):
        """Test with N=2."""
        models, ref_model = self._run_ddp_step(N=2)
        for p_ddp, p_ref in zip(models[0].parameters(), ref_model.parameters()):
            assert torch.allclose(p_ddp, p_ref, atol=1e-5)

    def test_eight_gpus(self):
        """Test with N=8."""
        models, ref_model = self._run_ddp_step(N=8)
        for p_ddp, p_ref in zip(models[0].parameters(), ref_model.parameters()):
            assert torch.allclose(p_ddp, p_ref, atol=1e-5)

    def test_weights_actually_changed(self):
        """Verify that the optimizer step actually updated the weights."""
        model_fn = make_model_fn(seed=42)
        initial_model = model_fn()
        initial_params = [p.clone() for p in initial_model.parameters()]

        models, _ = self._run_ddp_step(N=4, seed=42)
        for p_init, p_after in zip(initial_params, models[0].parameters()):
            assert not torch.allclose(p_init, p_after, atol=1e-8), \
                "Weights didn't change — optimizer step may not have run"
