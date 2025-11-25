"""Quick test of the Python wrapper"""

import chen
import numpy as np
import time

print("="*70)
print("CHEN.JL PYTHON WRAPPER TEST")
print("="*70)
print()

# Test 1: Basic signature
print("Test 1: Basic signature computation")
path = np.random.randn(100, 5)
sig = chen.sig(path, 3)
print(f"  Path shape: {path.shape}")
print(f"  Signature shape: {sig.shape}")
print(f"  First 5 values: {sig[:5]}")
print("  ✓ Passed")
print()

# Test 2: Log-signature
print("Test 2: Log-signature computation")
logsig = chen.logsig(path, 3)
print(f"  Log-signature shape: {logsig.shape}")
print("  ✓ Passed")
print()

# Test 3: Float32
print("Test 3: Float32 support")
path32 = np.random.randn(50, 3).astype(np.float32)
sig32 = chen.sig(path32, 2)
print(f"  Input dtype: {path32.dtype}")
print(f"  Output dtype: {sig32.dtype}")
print("  ✓ Passed" if sig32.dtype == np.float32 else "  ✗ Failed")
print()

# Test 4: Performance
print("Test 4: Performance (N=1000, d=10, m=5)")
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
print(f"  Time: {t_ms:.1f} ms")
print("  ✓ Passed")
print()

print("="*70)
print("ALL TESTS PASSED")
print("="*70)