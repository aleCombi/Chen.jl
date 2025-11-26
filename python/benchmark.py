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
# Ensure C-contiguous array (pysiglib is picky about memory layout)
path = np.ascontiguousarray(np.random.randn(N, d))

print(f"Benchmarking: N={N}, d={d}, m={m}")
print("-" * 40)

# --- 1. Chen ---
# Warmup
_ = chen.sig(path, m)

times = []
for _ in range(20):
    t0 = time.perf_counter()
    chen.sig(path, m)
    times.append(time.perf_counter() - t0)
t_chen = min(times) * 1000

print(f"chen:        {t_chen:.1f} ms")

# --- 2. iisignature ---
if HAS_IISIG:
    # Warmup
    _ = iisignature.sig(path, m)
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        iisignature.sig(path, m)
        times.append(time.perf_counter() - t0)
    t_iisig = min(times) * 1000
    
    print(f"iisignature: {t_iisig:.1f} ms  (Speedup: {t_iisig/t_chen:.2f}x)")
else:
    print("iisignature: Not installed (skipped)")

# --- 3. pysiglib ---
if HAS_PYSIGLIB:
    # Warmup
    _ = pysiglib.signature(path, m)
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        pysiglib.signature(path, m)
        times.append(time.perf_counter() - t0)
    t_pysiglib = min(times) * 1000
    
    print(f"pysiglib:    {t_pysiglib:.1f} ms  (Speedup: {t_pysiglib/t_chen:.2f}x)")
else:
    print("pysiglib:    Not installed (skipped)")