import numpy as np
import pytest
import chen


def test_rolling_sig_correctness_and_shape():
    np.random.seed(0)
    path = np.random.randn(12, 3)
    m = 2
    window = 5
    stride = 1

    sigs = chen.rolling_sig(path, m, window, stride=stride, threaded=False)

    expected_sig_len = 3 + 3**2  # 12
    expected_windows = (path.shape[0] - window) // stride + 1
    assert sigs.shape == (expected_sig_len, expected_windows)

    # Compare each window with direct sig
    for i in range(expected_windows):
        start = i * stride
        end = start + window
        sig_single = chen.sig(path[start:end, :], m)
        np.testing.assert_allclose(sigs[:, i], sig_single, rtol=1e-10)


def test_rolling_sig_stride():
    np.random.seed(1)
    path = np.random.randn(20, 2)
    m = 3
    window = 6
    stride = 2

    sigs = chen.rolling_sig(path, m, window, stride=stride, threaded=True)
    expected_windows = (path.shape[0] - window) // stride + 1
    assert sigs.shape[1] == expected_windows

    # Spot check a couple of windows
    for idx in (0, expected_windows - 1):
        start = idx * stride
        end = start + window
        sig_single = chen.sig(path[start:end, :], m)
        np.testing.assert_allclose(sigs[:, idx], sig_single, rtol=1e-10)


def test_rolling_sig_validation():
    path = np.random.randn(5, 2)
    with pytest.raises(Exception):
        chen.rolling_sig(path, 2, 10)
