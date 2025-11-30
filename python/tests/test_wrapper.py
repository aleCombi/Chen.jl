"""Test chen Python wrapper functionality"""

import pytest
import numpy as np
import chen


def test_basic_signature():
    """Test basic signature computation"""
    path = np.random.randn(100, 5)
    sig = chen.sig(path, 3)

    assert sig.shape[0] > 0, "Signature should not be empty"
    assert isinstance(sig, np.ndarray), "Signature should be numpy array"

    # Expected size: 5 + 25 + 125 = 155
    expected_size = 5 + 5**2 + 5**3
    assert len(sig) == expected_size, f"Expected size {expected_size}, got {len(sig)}"


def test_logsignature():
    """Test log-signature computation"""
    path = np.random.randn(100, 5)
    basis = chen.prepare_logsig(path.shape[1], 3)
    logsig = chen.logsig(path, basis)

    assert logsig.shape[0] > 0, "Log-signature should not be empty"
    assert isinstance(logsig, np.ndarray), "Log-signature should be numpy array"
    # Log-signature should be smaller than signature due to Lyndon basis
    sig = chen.sig(path, 3)
    assert len(logsig) < len(sig), "Log-signature should be more compact than signature"


def test_float32_support():
    """Test that float32 input is properly handled"""
    path32 = np.random.randn(50, 3).astype(np.float32)
    sig32 = chen.sig(path32, 2)

    assert isinstance(sig32, np.ndarray), "Output should be numpy array"
    # Julia converts to Float64 by default, which is fine
    assert sig32.dtype in [np.float32, np.float64], f"Unexpected dtype: {sig32.dtype}"


@pytest.mark.slow
def test_performance():
    """Test performance on larger input (marked as slow test)"""
    import time

    path_large = np.random.randn(1000, 10)

    # Warmup
    _ = chen.sig(path_large, 5)

    # Time it
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        sig = chen.sig(path_large, 5)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t_ms = min(times) * 1000
    # Just verify it completes; don't assert on timing (hardware-dependent)
    assert t_ms > 0, "Performance test should complete"
