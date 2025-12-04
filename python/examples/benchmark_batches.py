"""
Benchmark batch signature computation for ChenSignatures.

Reference timings from literature (Forward pass, CPU):
| (B, L, d, N)      | esig   | iisignature | pySigLib | signatory | ChenSig (target) |
|-------------------|--------|-------------|----------|-----------|------------------|
| (128, 256, 4, 6)  | 1.1310 | 0.4104      | 0.0482   | 0.0558    | 0.0110           |
| (128, 512, 8, 5)  | 11.481 | 4.7908      | 0.3673   | 0.4512    | 0.0896           |
| (128, 1024, 16, 4)| 34.784 | 14.3296     | 1.1512   | 2.2121    | 0.2988           |

Where:
- B = batch size
- L = path length (number of time points)
- d = spatial dimension
- N = truncation level (signature depth)
"""

import time
import numpy as np
import chen
from typing import Tuple


def generate_random_paths(B: int, L: int, d: int, dtype=np.float64) -> np.ndarray:
    """
    Generate random paths for benchmarking.

    Args:
        B: Batch size
        L: Path length (number of time points)
        d: Spatial dimension
        dtype: numpy dtype for the arrays

    Returns:
        Array of shape (L, d, B)
    """
    return np.random.randn(L, d, B).astype(dtype)


def benchmark_sig_batch(
    B: int,
    L: int,
    d: int,
    N: int,
    n_warmup: int = 3,
    n_trials: int = 10,
    threaded: bool = True,
    dtype=np.float64
) -> Tuple[float, float, float]:
    """
    Benchmark signature computation for a batch of paths.

    Args:
        B: Batch size
        L: Path length
        d: Dimension
        N: Truncation level
        n_warmup: Number of warmup runs
        n_trials: Number of timing trials
        threaded: Use multi-threading
        dtype: numpy dtype

    Returns:
        (min_time, mean_time, std_time) in seconds
    """
    # Generate test data
    paths = generate_random_paths(B, L, d, dtype)

    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        _ = chen.sig(paths, N, threaded=threaded)

    # Benchmark
    print(f"  Running benchmark ({n_trials} trials)...")
    times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        result = chen.sig(paths, N, threaded=threaded)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"    Trial {i+1}/{n_trials}: {elapsed:.4f}s")

    times = np.array(times)
    return np.min(times), np.mean(times), np.std(times)


def print_result_row(params: Tuple[int, int, int, int],
                     time_min: float,
                     time_mean: float,
                     time_std: float,
                     reference: float = None):
    """Print a formatted result row."""
    B, L, d, N = params
    result = f"({B:3d}, {L:4d}, {d:2d}, {N}) | {time_min:7.4f} | {time_mean:7.4f} | {time_std:7.4f}"

    if reference is not None:
        speedup = reference / time_min
        result += f" | {reference:7.4f} | {speedup:6.2f}x"

    print(result)


def main():
    """Run the benchmark suite."""
    print("=" * 80)
    print("ChenSignatures Batch Signature Benchmark")
    print("=" * 80)
    print()

    # Test configurations from the reference paper
    test_cases = [
        # (B, L, d, N, reference_time)
        (128, 256, 4, 6, 0.0110),   # Small case
        (128, 512, 8, 5, 0.0896),   # Medium case
        (128, 1024, 16, 4, 0.2988), # Large case
    ]

    print("Configuration:")
    print(f"  dtype: float64")
    print(f"  warmup: 3 runs")
    print(f"  trials: 10 runs")
    print()

    # Float64 results
    print("Results (float64, threaded=True):")
    print("-" * 80)
    print(" (B,    L,  d, N) |   Min   |  Mean   |   Std   | Ref(ms) | Speedup")
    print("-" * 80)

    results_f64 = []
    for B, L, d, N, ref_ms in test_cases:
        print(f"\nBenchmarking (B={B}, L={L}, d={d}, N={N})...")
        min_t, mean_t, std_t = benchmark_sig_batch(B, L, d, N, threaded=True, dtype=np.float64)
        results_f64.append((min_t, mean_t, std_t))
        ref_sec = ref_ms  # Already in seconds based on the table
        print_result_row((B, L, d, N), min_t, mean_t, std_t, ref_sec)

    print()
    print("=" * 80)

    # Float32 results
    print("\nResults (float32, threaded=True):")
    print("-" * 80)
    print(" (B,    L,  d, N) |   Min   |  Mean   |   Std   ")
    print("-" * 80)

    for i, (B, L, d, N, _) in enumerate(test_cases):
        print(f"\nBenchmarking (B={B}, L={L}, d={d}, N={N})...")
        min_t, mean_t, std_t = benchmark_sig_batch(B, L, d, N, threaded=True, dtype=np.float32)
        print_result_row((B, L, d, N), min_t, mean_t, std_t)

    print()
    print("=" * 80)

    # Sequential vs threaded comparison (float64 only)
    print("\nThreaded vs Sequential Comparison (float64, first test case):")
    print("-" * 80)
    B, L, d, N, _ = test_cases[0]

    print(f"\nBenchmarking (B={B}, L={L}, d={d}, N={N}) - Sequential...")
    min_seq, mean_seq, std_seq = benchmark_sig_batch(B, L, d, N, threaded=False, dtype=np.float64)

    print(f"\nBenchmarking (B={B}, L={L}, d={d}, N={N}) - Threaded...")
    min_thr, mean_thr, std_thr = benchmark_sig_batch(B, L, d, N, threaded=True, dtype=np.float64)

    print()
    print(f"Sequential: {min_seq:.4f}s (min), {mean_seq:.4f}s (mean)")
    print(f"Threaded:   {min_thr:.4f}s (min), {mean_thr:.4f}s (mean)")
    print(f"Speedup:    {min_seq/min_thr:.2f}x")

    print()
    print("=" * 80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
