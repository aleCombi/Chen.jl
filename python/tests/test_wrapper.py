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


def test_batch_signatures():
    """Test batch signature computation with 3D input"""
    np.random.seed(42)
    batch_size = 10
    n_points = 50
    dim = 3
    m = 2

    # Create batch of paths: (N, d, B)
    paths = np.random.randn(n_points, dim, batch_size)

    # Compute batch signatures
    sigs_batch = chen.sig(paths, m, threaded=True)

    # Expected shape: (sig_length, batch_size)
    expected_sig_len = dim + dim**2  # 3 + 9 = 12
    assert sigs_batch.shape == (expected_sig_len, batch_size), \
        f"Expected shape ({expected_sig_len}, {batch_size}), got {sigs_batch.shape}"

    # Verify each batch element matches single-path computation
    for i in range(batch_size):
        sig_single = chen.sig(paths[:, :, i], m)
        np.testing.assert_allclose(sigs_batch[:, i], sig_single, rtol=1e-10,
                                   err_msg=f"Batch element {i} doesn't match single computation")


def test_batch_signatures_sequential():
    """Test batch signature computation with threading disabled"""
    np.random.seed(42)
    batch_size = 5
    n_points = 30
    dim = 2
    m = 3

    paths = np.random.randn(n_points, dim, batch_size)

    # Compute with threaded=False
    sigs_sequential = chen.sig(paths, m, threaded=False)

    # Compute with threaded=True
    sigs_threaded = chen.sig(paths, m, threaded=True)

    # Both should give identical results
    np.testing.assert_allclose(sigs_sequential, sigs_threaded, rtol=1e-12,
                               err_msg="Sequential and threaded batch results differ")


def test_batch_logsignatures():
    """Test batch log-signature computation with 3D input"""
    np.random.seed(123)
    batch_size = 8
    n_points = 40
    dim = 3
    m = 2

    # Create batch of paths: (N, d, B)
    paths = np.random.randn(n_points, dim, batch_size)

    # Prepare basis
    basis = chen.prepare_logsig(dim, m)

    # Compute batch log-signatures
    logsigs_batch = chen.logsig(paths, basis, threaded=True)

    # Verify shape: (num_lyndon_words, batch_size)
    assert logsigs_batch.shape[1] == batch_size, \
        f"Expected batch size {batch_size}, got {logsigs_batch.shape[1]}"
    assert logsigs_batch.ndim == 2, "Batch logsig should be 2D"

    # Verify each batch element matches single-path computation
    for i in range(batch_size):
        logsig_single = chen.logsig(paths[:, :, i], basis)
        np.testing.assert_allclose(logsigs_batch[:, i], logsig_single, rtol=1e-10,
                                   err_msg=f"Batch element {i} doesn't match single computation")


def test_batch_dtype_preservation():
    """Test that float32 dtype is preserved in batch processing"""
    np.random.seed(789)
    batch_size = 5
    n_points = 30
    dim = 2
    m = 2

    # Create float32 batch
    paths_float32 = np.random.randn(n_points, dim, batch_size).astype(np.float32)

    # Compute signatures
    sigs = chen.sig(paths_float32, m, threaded=True)

    # Should preserve float32
    assert sigs.dtype == np.float32, f"Expected float32, got {sigs.dtype}"

    # Test with float64
    paths_float64 = paths_float32.astype(np.float64)
    sigs64 = chen.sig(paths_float64, m, threaded=True)
    assert sigs64.dtype == np.float64, f"Expected float64, got {sigs64.dtype}"


def test_batch_logsig_dtype_preservation():
    """Test that dtype is preserved in batch log-signature computation"""
    np.random.seed(456)
    batch_size = 5
    n_points = 30
    dim = 2
    m = 2

    # Create float32 batch
    paths_float32 = np.random.randn(n_points, dim, batch_size).astype(np.float32)
    basis = chen.prepare_logsig(dim, m)

    # Compute log-signatures
    logsigs = chen.logsig(paths_float32, basis, threaded=True)

    # Should preserve float32
    assert logsigs.dtype == np.float32, f"Expected float32, got {logsigs.dtype}"


def test_single_path_still_works():
    """Test that single path computation still works after adding batch support"""
    np.random.seed(111)
    path = np.random.randn(50, 3)

    # These should still work without threaded parameter
    sig = chen.sig(path, 2)
    assert sig.ndim == 1, "Single path signature should be 1D"

    basis = chen.prepare_logsig(3, 2)
    logsig = chen.logsig(path, basis)
    assert logsig.ndim == 1, "Single path log-signature should be 1D"


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
