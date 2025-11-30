"""Test PyTorch integration with chen.torch"""

import pytest
import numpy as np

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="PyTorch not installed. Install with: uv sync --extra torch")

import chen
from chen.torch import sig_torch


def test_torch_forward():
    """Test forward pass with PyTorch"""
    path = torch.randn(10, 3, dtype=torch.float64)
    sig = sig_torch(path, m=3)

    assert isinstance(sig, torch.Tensor), "Output should be torch.Tensor"
    assert sig.shape[0] > 0, "Signature should not be empty"

    # Expected size: 3 + 9 + 27 = 39
    expected_size = 3 + 3**2 + 3**3
    assert sig.shape[0] == expected_size, f"Expected size {expected_size}, got {sig.shape[0]}"


def test_torch_backward():
    """Test backward pass (autograd)"""
    path = torch.randn(10, 3, requires_grad=True, dtype=torch.float64)
    sig = sig_torch(path, m=2)

    # Compute a scalar loss
    loss = sig.sum()
    loss.backward()

    assert path.grad is not None, "Gradients should be computed"
    assert path.grad.shape == path.shape, "Gradient shape should match input shape"
    assert not torch.isnan(path.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(path.grad).any(), "Gradients should not contain Inf"


def test_torch_vs_numpy():
    """Test that torch version matches numpy version"""
    path_np = np.random.randn(15, 4)

    # Compute with numpy
    sig_np = chen.sig(path_np, m=2)

    # Compute with torch
    path_torch = torch.from_numpy(path_np)
    sig_torch_result = sig_torch(path_torch, m=2).detach().numpy()

    # Should match
    assert np.allclose(sig_np, sig_torch_result, rtol=1e-10), \
        "PyTorch and NumPy results should match"


def test_torch_gradient_finite_difference():
    """Test gradients using finite difference approximation"""
    torch.manual_seed(42)
    path = torch.randn(5, 2, requires_grad=True, dtype=torch.float64)

    # Compute gradient with autograd
    sig = sig_torch(path, m=2)
    loss = sig.sum()
    loss.backward()
    grad_autograd = path.grad.clone()

    # Compute gradient with finite differences
    eps = 1e-6
    grad_fd = torch.zeros_like(path)

    with torch.no_grad():
        for i in range(path.shape[0]):
            for j in range(path.shape[1]):
                # Forward perturbation
                path_plus = path.clone()
                path_plus[i, j] += eps
                sig_plus = sig_torch(path_plus, m=2)

                # Backward perturbation
                path_minus = path.clone()
                path_minus[i, j] -= eps
                sig_minus = sig_torch(path_minus, m=2)

                # Central difference
                grad_fd[i, j] = (sig_plus.sum() - sig_minus.sum()) / (2 * eps)

    # Compare
    max_diff = (grad_autograd - grad_fd).abs().max().item()
    assert max_diff < 1e-4, f"Gradient mismatch: max diff = {max_diff}"
