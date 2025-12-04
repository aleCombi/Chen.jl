"""
Compare ChenSignatures against pySigLib for correctness and performance.

IMPORTANT: Different array layout conventions:
- ChenSignatures: (N, d, B) where N=length, d=dimension, B=batch_size
- pySigLib: (B, N, d) where B=batch_size, N=length, d=dimension

Threading control:
- Julia: JULIA_NUM_THREADS environment variable (must be set BEFORE Python starts)
- pySigLib: n_jobs parameter (-1 = all threads, 1 = serial)

Usage:
  # Single-threaded comparison (most fair for benchmarking):
  set JULIA_NUM_THREADS=1 && python compare_pysiglib.py     (Windows)
  JULIA_NUM_THREADS=1 python compare_pysiglib.py            (Linux/Mac)

  # Multi-threaded comparison with 8 threads:
  set JULIA_NUM_THREADS=8 && python compare_pysiglib.py     (Windows)
  JULIA_NUM_THREADS=8 python compare_pysiglib.py            (Linux/Mac)

  # Full parallelism (uses all available cores):
  set JULIA_NUM_THREADS=16 && python compare_pysiglib.py    (Windows)
  JULIA_NUM_THREADS=16 python compare_pysiglib.py           (Linux/Mac)

Note: By default, if JULIA_NUM_THREADS is not set, the script attempts to set it
to the CPU count, but this only works if juliacall hasn't been initialized yet.
For reliable thread control, always set JULIA_NUM_THREADS before running Python.
"""

import os

# Set JULIA_NUM_THREADS before importing chen (Julia initialization happens on import)
# This can be controlled via environment variable or defaults to CPU count
if "JULIA_NUM_THREADS" not in os.environ:
    import multiprocessing
    # Default to all available threads
    os.environ["JULIA_NUM_THREADS"] = str(multiprocessing.cpu_count())

import time
import numpy as np
import chen

try:
    from pysiglib import signature as pysiglib_signature
    PYSIGLIB_AVAILABLE = True
except ImportError:
    PYSIGLIB_AVAILABLE = False
    print("WARNING: pySigLib not available, skipping comparison")


def get_julia_thread_count() -> int:
    """Get the number of threads Julia is using."""
    from juliacall import Main as jl
    return int(jl.Threads.nthreads())


def generate_paths_chen(B: int, L: int, d: int, dtype=np.float64) -> np.ndarray:
    """Generate paths in ChenSignatures format: (L, d, B)"""
    return np.random.randn(L, d, B).astype(dtype)


def chen_to_pysiglib(paths: np.ndarray) -> np.ndarray:
    """
    Convert ChenSignatures format (L, d, B) to pySigLib format (B, L, d).

    Args:
        paths: Array of shape (L, d, B)

    Returns:
        Array of shape (B, L, d)
    """
    # (L, d, B) -> (B, L, d)
    return np.transpose(paths, (2, 0, 1))


def pysiglib_to_chen(result: np.ndarray) -> np.ndarray:
    """
    Convert pySigLib result format (B, sig_len) to ChenSignatures format (sig_len, B).

    Args:
        result: Array of shape (B, sig_len)

    Returns:
        Array of shape (sig_len, B)
    """
    return result.T


def benchmark_and_compare(B: int, L: int, d: int, N: int, n_trials: int = 10, n_jobs: int = None):
    """
    Benchmark both implementations and check for correctness.

    Args:
        B: Batch size
        L: Path length
        d: Dimension
        N: Truncation level
        n_trials: Number of benchmark trials
        n_jobs: Number of threads for pySigLib (None = match Julia's thread count)
    """
    # Get Julia's thread count for fair comparison
    julia_threads = get_julia_thread_count()

    # Default n_jobs to match Julia's thread count
    if n_jobs is None:
        n_jobs = julia_threads

    print(f"\n{'='*80}")
    print(f"Test case: B={B}, L={L}, d={d}, N={N}")
    print(f"Threading: Julia={julia_threads} threads, pySigLib n_jobs={n_jobs}")
    print(f"{'='*80}")

    # Generate test data in ChenSignatures format (use float32 for pySigLib compatibility)
    np.random.seed(42)
    paths_chen = generate_paths_chen(B, L, d, dtype=np.float32)

    # Convert to pySigLib format
    paths_pysig = chen_to_pysiglib(paths_chen).astype(np.float32)

    print(f"\nArray shapes:")
    print(f"  ChenSignatures input: {paths_chen.shape} (L, d, B)")
    print(f"  pySigLib input:       {paths_pysig.shape} (B, L, d)")

    # =========================================================================
    # Test ChenSignatures
    # =========================================================================
    print(f"\n{'-'*80}")
    print("ChenSignatures (threaded):")
    print(f"{'-'*80}")

    # Warmup
    _ = chen.sig(paths_chen, N, threaded=True)

    # Benchmark
    chen_times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        result_chen = chen.sig(paths_chen, N, threaded=True)
        t1 = time.perf_counter()
        chen_times.append(t1 - t0)

    chen_min = np.min(chen_times)
    chen_mean = np.mean(chen_times)
    chen_std = np.std(chen_times)

    print(f"  Output shape: {result_chen.shape} (sig_len, B)")
    print(f"  Min time:  {chen_min:.4f}s")
    print(f"  Mean time: {chen_mean:.4f}s")
    print(f"  Std time:  {chen_std:.4f}s")

    # =========================================================================
    # Test pySigLib
    # =========================================================================
    if not PYSIGLIB_AVAILABLE:
        print("\npySigLib not available - skipping")
        return

    print(f"\n{'-'*80}")
    print(f"pySigLib (CPU, n_jobs={n_jobs}):")
    print(f"{'-'*80}")

    # Warmup
    _ = pysiglib_signature(paths_pysig, N, n_jobs=n_jobs)

    # Benchmark
    pysig_times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        result_pysig = pysiglib_signature(paths_pysig, N, n_jobs=n_jobs)
        t1 = time.perf_counter()
        pysig_times.append(t1 - t0)

    pysig_min = np.min(pysig_times)
    pysig_mean = np.mean(pysig_times)
    pysig_std = np.std(pysig_times)

    print(f"  Output shape: {result_pysig.shape} (B, sig_len)")
    print(f"  Min time:  {pysig_min:.4f}s")
    print(f"  Mean time: {pysig_mean:.4f}s")
    print(f"  Std time:  {pysig_std:.4f}s")

    # =========================================================================
    # Compare results
    # =========================================================================
    print(f"\n{'-'*80}")
    print("Correctness check:")
    print(f"{'-'*80}")

    # Convert pySigLib result to ChenSignatures format for comparison
    result_pysig_transposed = pysiglib_to_chen(result_pysig)

    print(f"  ChenSignatures result shape: {result_chen.shape}")
    print(f"  pySigLib result shape (transposed): {result_pysig_transposed.shape}")

    # pySigLib includes constant term (1) as first element, ChenSignatures doesn't
    # Remove the constant term from pySigLib result
    result_pysig_no_const = result_pysig_transposed[1:, :]

    print(f"  pySigLib without constant: {result_pysig_no_const.shape}")

    # Check if results match
    max_diff = np.max(np.abs(result_chen - result_pysig_no_const))
    rel_diff = max_diff / (np.max(np.abs(result_chen)) + 1e-10)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    # Use a more lenient tolerance for numerical differences (float32 has ~7 digits precision)
    if np.allclose(result_chen, result_pysig_no_const, rtol=1e-5, atol=1e-2):
        print(f"  [OK] Results match (within tolerance)")
    else:
        print(f"  [FAIL] Results differ significantly!")
        print(f"    Sample ChenSignatures values: {result_chen[:5, 0]}")
        print(f"    Sample pySigLib values:       {result_pysig_no_const[:5, 0]}")

    # =========================================================================
    # Performance comparison
    # =========================================================================
    print(f"\n{'-'*80}")
    print("Performance comparison:")
    print(f"{'-'*80}")

    speedup = pysig_min / chen_min
    print(f"  pySigLib min time:       {pysig_min:.4f}s")
    print(f"  ChenSignatures min time: {chen_min:.4f}s")

    if speedup > 1:
        print(f"  ChenSignatures is {speedup:.2f}x FASTER than pySigLib")
    else:
        print(f"  pySigLib is {1/speedup:.2f}x FASTER than ChenSignatures")


def main():
    """Run comparison benchmarks."""
    print("="*80)
    print("ChenSignatures vs pySigLib Comparison")
    print("="*80)

    if not PYSIGLIB_AVAILABLE:
        print("\nERROR: pySigLib not available")
        print("Install with: pip install pysiglib")
        return

    # Show threading configuration
    julia_threads = get_julia_thread_count()
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    print(f"\nSystem info:")
    print(f"  CPU cores: {cpu_count}")
    print(f"  Julia threads: {julia_threads}")
    print(f"  JULIA_NUM_THREADS env: {os.environ.get('JULIA_NUM_THREADS', 'not set')}")

    # Warn if Julia threads don't match environment variable
    env_threads = os.environ.get('JULIA_NUM_THREADS')
    if env_threads and int(env_threads) != julia_threads:
        print(f"\n  WARNING: Julia is using {julia_threads} threads but JULIA_NUM_THREADS={env_threads}")
        print(f"  This means Julia was already initialized before the environment variable was set.")
        print(f"  To change thread count, restart Python with JULIA_NUM_THREADS set beforehand.")
        print(f"  Example: set JULIA_NUM_THREADS={env_threads} && python compare_pysiglib.py")

    print(f"\nNote: Thread counts are matched between Julia and pySigLib for fair comparison")

    # Test cases from the reference paper + edge cases
    test_cases = [
        (1, 256, 4, 6),      # Edge case: batch size 1
        (128, 256, 4, 6),    # Small case
        (128, 512, 8, 5),    # Medium case
        (128, 1024, 16, 4),  # Large case
    ]

    for B, L, d, N in test_cases:
        benchmark_and_compare(B, L, d, N, n_trials=10)

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
