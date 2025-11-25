# Chen.jl Benchmarks

This directory contains the performance benchmarking suite for **Chen.jl**, comparing it directly against the Python state-of-the-art library **iisignature**.

## 📋 Prerequisites

To run these benchmarks, you need:

1.  **Julia** (v1.10+)
2.  **Python** (v3.9+)
3.  **[uv](https://github.com/astral-sh/uv)**: An extremely fast Python package manager.
    *   *Installation:* `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or `irm https://astral.sh/uv/install.ps1 | iex` (Windows).

> **Why `uv`?** The benchmark scripts automatically bootstrap an isolated, reproducible Python environment with exact versions of `numpy` and `iisignature` without polluting your global system.

## 🚀 Running the Benchmarks

The main orchestrator script is `run_benchmark.py`. It handles:
1.  Setting up the Python environment.
2.  Running the Julia benchmarks (`benchmark.jl`).
3.  Running the Python benchmarks (`benchmark.py`).
4.  Merging results and calculating speedup ratios.

To run the full suite:

```bash
# If you have python installed:
python run_benchmark.py

# Or using uv directly:
uv run run_benchmark.py
```

### Output
Results are saved to the `runs/` directory as CSV files:
*   `run_julia_TIMESTAMP.csv`: Raw Julia timings.
*   `run_python_TIMESTAMP.csv`: Raw Python timings.
*   `comparison_TIMESTAMP.csv`: **The final report containing speedup ratios.**

## ⚙️ Configuration

Benchmarks are configured via `benchmark_config.yaml`. You can modify this file to change the sweep parameters:

```yaml
# benchmark_config.yaml

path_kind: "linear"   # "linear" or "sin"
Ns: [100, 1000]       # Path lengths (number of time steps)
Ds: [2, 6, 20]        # Dimensions
Ms: [3, 5]            # Truncation levels
repeats: 5            # Samples per benchmark
runs_dir: "runs"      # Output directory
```

## ✅ Verification

Speed is useless without correctness. Use `check_signatures.py` to numerically validate `Chen.jl` output against `iisignature`.

```bash
uv run check_signatures.py
```

This script generates random paths based on the config, computes signatures in both languages, and reports the $L_2$ difference and maximum absolute error.

## 📂 File Structure

*   **`run_benchmark.py`**: Main entry point. Automates the entire process.
*   **`benchmark_config.yaml`**: Configuration for dimensions, levels, and path types.
*   **`benchmark.jl`**: Julia benchmark loop (uses `BenchmarkTools.jl`).
*   **`benchmark.py`**: Python benchmark loop (uses `time` and `tracemalloc`).
*   **`check_signatures.py`**: Validation script to ensure numerical accuracy.
*   **`sigcheck.jl`**: Helper script called by `check_signatures.py` to get single-point Julia results.
*   **`generate_fixtures.py`**: Helper to generate hardcoded test vectors for the main test suite.