import chen
import numpy as np
import time

try:
    import iisignature
    HAS_IISIG = True
except:
    HAS_IISIG = False

N, d, m = 1000, 10, 5
path = np.random.randn(N, d)

# Chen
_ = chen.sig(path, m)
times = []
for _ in range(20):
    t0 = time.perf_counter()
    chen.sig(path, m)
    times.append(time.perf_counter() - t0)
t_chen = min(times) * 1000

print(f"chen:        {t_chen:.1f} ms")

if HAS_IISIG:
    _ = iisignature.sig(path, m)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        iisignature.sig(path, m)
        times.append(time.perf_counter() - t0)
    t_iisig = min(times) * 1000
    
    print(f"iisignature: {t_iisig:.1f} ms")
    print(f"Speedup:     {t_iisig/t_chen:.2f}x")