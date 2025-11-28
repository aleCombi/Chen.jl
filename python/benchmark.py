import chen
import numpy as np
import time

# --- Imports with checks ---
try:
    import iisignature
    HAS_IISIG = True
except ImportError:
    HAS_IISIG = False

try:
    import pysiglib
    HAS_PYSIGLIB = True
except ImportError:
    HAS_PYSIGLIB = False

# --- Setup ---
N, d, m = 1000, 5, 7
# Ensure C-contiguous float64 array
path = np.ascontiguousarray(np.random.randn(N, d), dtype=np.float64)

print(f"Benchmarking: N={N}, d={d}, m={m}")
print("=" * 60)

# --- 1. ChenSignatures (optimized) ---
# Warmup
_ = chen.sig(path, m)

times = []
for _ in range(20):
    t0 = time.perf_counter()
    chen.sig(path, m)
    times.append(time.perf_counter() - t0)
t_chen = min(times) * 1000

print(f"chen (optimized):     {t_chen:.1f} ms")

# --- 3. iisignature ---
if HAS_IISIG:
    # Warmup
    _ = iisignature.sig(path, m)
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        iisignature.sig(path, m)
        times.append(time.perf_counter() - t0)
    t_iisig = min(times) * 1000
    
    print(f"iisignature:          {t_iisig:.1f} ms  (vs chen: {t_iisig/t_chen:.2f}x)")
else:
    print("iisignature:          Not installed (skipped)")

# --- 4. pysiglib ---
if HAS_PYSIGLIB:
    # Warmup
    _ = pysiglib.signature(path, m)
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        pysiglib.signature(path, m)
        times.append(time.perf_counter() - t0)
    t_pysiglib = min(times) * 1000
    
    print(f"pysiglib:             {t_pysiglib:.1f} ms  (vs chen: {t_pysiglib/t_chen:.2f}x)")
else:
    print("pysiglib:             Not installed (skipped)")

print("=" * 60)
print("\nNotes:")
print("  - 'chen (optimized)' uses the default fast implementation")
print("  - 'chen (enzyme)' uses the Enzyme-compatible differentiable version")
print("  - Speedup/slowdown shown relative to chen (optimized)")