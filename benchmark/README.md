# Chen.jl Benchmarks

This directory contains the performance benchmarking suite for **Chen.jl**, comparing it against Python state-of-the-art libraries: **iisignature** and **pysiglib** (pysiglib).

## 📋 Prerequisites

To run these benchmarks, you need:

1.  **Julia** (v1.10+)
2.  **Python** (v3.9+)
3.  **[uv](https://github.com/astral-sh/uv)**: An extremely fast Python package manager.
    *   *Installation:* `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or `irm https://astral.sh/uv/install.ps1 | iex` (Windows).

> **Why `uv`?** The benchmark scripts automatically bootstrap an isolated, reproducible Python environment with exact versions of `numpy`, `iisignature`, and `pysiglib` (pysiglib) without polluting your global system.

## 🚀 Running the Benchmarks

The main orchestrator script is `run_benchmark.py`. It handles:
1.  Setting up the Python environment.
2.  Running the Julia benchmarks (`benchmark.jl`).
3.  Running the Python benchmarks (`benchmark.py`) for both iisignature and pysiglib.
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
*   `run_python_TIMESTAMP.csv`: Raw Python timings (includes both libraries).
*   `comparison_TIMESTAMP.csv`: **The final report containing speedup ratios for each Python library.**

The comparison CSV includes a `python_library` column to distinguish between `iisignature` and `pysiglib` results.

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
logsig_method: "S"    # iisignature log-sig method: "O" or "S"
```

**Note:** The `logsig_method` parameter only applies to iisignature. pysiglib uses its own default method.

## ✅ Verification

Speed is useless without correctness. Use `check_signatures.py` to numerically validate `Chen.jl` output against both `iisignature` and `pysiglib`.

```bash
uv run check_signatures.py
```

This script:
- Generates random paths based on the config
- Computes signatures in Julia (Chen.jl)
- Computes signatures in Python using iisignature
- Computes signatures in Python using pysiglib
- Reports the L₂ difference and maximum absolute error for each comparison

The output CSV includes a `python_library` column to show which library was used for each comparison.

## 📂 File Structure

*   **`run_benchmark.py`**: Main entry point. Automates the entire benchmarking process.
*   **`benchmark_config.yaml`**: Configuration for dimensions, levels, and path types.
*   **`benchmark.jl`**: Julia benchmark loop (uses `BenchmarkTools.jl`).
*   **`benchmark.py`**: Python benchmark loop (uses `time` and `tracemalloc`). Tests both iisignature and pysiglib.
*   **`check_signatures.py`**: Validation script to ensure numerical accuracy against both Python libraries.
*   **`sigcheck.jl`**: Helper script called by `check_signatures.py` to get single-point Julia results.
*   **`generate_fixtures.py`**: Helper to generate hardcoded test vectors for the main test suite (uses iisignature as reference).
*   **`pyproject.toml`**: Python dependencies including iisignature, numpy, and pysiglib (pysiglib).

## 📊 Benchmark Results Format

### Julia Output (`run_julia_TIMESTAMP.csv`)
```csv
N,d,m,path_kind,operation,t_ms,alloc_KiB
100,2,3,linear,signature,12.5,48.2
```

### Python Output (`run_python_TIMESTAMP.csv`)
```csv
N,d,m,path_kind,operation,library,t_ms,alloc_KiB
100,2,3,linear,signature,iisignature,18.3,64.1
100,2,3,linear,signature,pysiglib,15.7,52.3
```

### Comparison Output (`comparison_TIMESTAMP.csv`)
```csv
N,d,m,path_kind,operation,python_library,t_ms_julia,t_ms_python,speed_ratio_python_over_julia,alloc_KiB_julia,alloc_KiB_python
100,2,3,linear,signature,iisignature,12.5,18.3,1.46,48.2,64.1
100,2,3,linear,signature,pysiglib,12.5,15.7,1.26,48.2,52.3
```

The `speed_ratio_python_over_julia` shows how many times slower (>1.0) or faster (<1.0) the Python library is compared to Chen.jl.

## 🔬 Library Comparison

| Library | Strength | Notes |
|---------|----------|-------|
| **iisignature** | Well-established, reference implementation | Unmaintained, Python ≤3.9 only |
| **pysiglib** (pysiglib) | Modern, actively maintained | Newer implementation, supports Python 3.9+ |
| **Chen.jl** | Highest performance, modern Julia | This library |

## 💡 Tips

1. **First run is slow**: The first benchmark run will install Julia packages. Subsequent runs are much faster.
2. **Warmup matters**: Both Julia and Python benchmarks include warmup iterations to ensure fair comparisons.
3. **Memory tracking**: The benchmarks track peak memory usage to compare allocation efficiency.
4. **Library availability**: If a Python library is not available, benchmarks will skip it gracefully and report which libraries were tested.

## 🐛 Troubleshooting

**Problem**: `iisignature` or `pysiglib` not found
```bash
# Manually install dependencies
uv add iisignature numpy pysiglib
```

**Problem**: Julia package installation fails
```bash
# Pre-install Julia packages
cd benchmark
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Problem**: Different results between libraries
- Check `check_signatures.py` output for detailed error metrics
- Small numerical differences (< 1e-8) are expected due to different implementations
- Large differences indicate a potential bug

## 📝 Adding New Test Cases

To add new benchmark scenarios:

1. Edit `benchmark_config.yaml` to add new N/D/M values
2. (Optional) Add new path generators in `benchmark.py` and `benchmark.jl`
3. Run `python run_benchmark.py`

## 🎯 Performance Goals

Chen.jl aims to be:
- **Faster than iisignature** by 1.5-2x for typical use cases
- **Competitive with pysiglib** (within 0.8-1.2x)
- **Memory efficient** with lower allocation overhead