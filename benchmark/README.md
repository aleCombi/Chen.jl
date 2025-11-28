# ChenSignatures.jl Benchmarks

Performance benchmarking and validation suite for **ChenSignatures.jl**, comparing it against Python libraries:
- **iisignature** (industry standard)
- **pysiglib** (PyTorch-based)

Features:
- End-to-end **runtime and allocation benchmarks** (Julia vs Python)
- **Scaling analysis** in N (path length), d (dimension), m (signature level)
- **Correctness validation** against reference implementations
- **Multiple dispatch benchmarking** (Matrix vs Vector{SVector})

---

## 📋 Prerequisites

To run the benchmarks, you need:

1. **Julia** ≥ 1.10
2. **Python** ≥ 3.9
3. **[uv](https://github.com/astral-sh/uv)** – fast Python package manager

Install `uv`:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

---

## 🚀 Quick Start

### Run Everything (Orchestrated)

```bash
cd benchmark
uv run compare_benchmarks.py
```

This will:
1. Run Julia benchmarks (both Matrix and Vector{SVector} dispatches)
2. Run Python benchmarks (iisignature and pysiglib)
3. Generate comparison CSV and performance plots
4. Create a timestamped folder in `runs/benchmark_comparison_*/`

### Run Individual Benchmarks

```bash
# Julia only
julia benchmark.jl
# → Creates runs/benchmark_julia_TIMESTAMP/

# Python only
uv run benchmark.py
# → Creates runs/benchmark_python_TIMESTAMP/

# Validation (correctness check)
uv run check_signatures.py
# → Creates runs/signature_check_TIMESTAMP/
```

---

## 📁 Folder Structure

### Run Folder Organization

Each script creates a timestamped run folder with a descriptive prefix:

| Prefix | Created By | Contents |
|--------|-----------|----------|
| `benchmark_julia_*` | `benchmark.jl` | Julia benchmarks (standalone) |
| `benchmark_python_*` | `benchmark.py` | Python benchmarks (standalone) |
| `benchmark_comparison_*` | `compare_benchmarks.py` | Full comparison with plots |
| `signature_check_*` | `check_signatures.py` | Validation results |

### Example: Comparison Run

```
runs/benchmark_comparison_20251128_191349/
├── benchmark_config.yaml      # Config snapshot (reproducibility)
├── config_resolved.json       # Resolved configuration
├── run_metadata.json          # Timestamps, run type
├── julia_results.csv          # Julia benchmark results
├── python_results.csv         # Python benchmark results
├── comparison.csv             # Head-to-head comparison
├── comparison_3x2.png         # Performance plots (3 params × 2 ops)
├── julia_stdout.log           # Execution logs
├── julia_stderr.log
├── python_stdout.log
├── python_stderr.log
└── SUMMARY.txt                # Human-readable summary with speedup stats
```

### Example: Standalone Run

```
runs/benchmark_julia_20251128_143022/
├── benchmark_config.yaml
├── config_resolved.txt
├── run_metadata.txt
├── results.csv                # Benchmark results
└── SUMMARY.txt                # Summary statistics
```

---

## ⚙️ Configuration

Edit `benchmark_config.yaml` to customize benchmarks:

```yaml
path_kind: "sin"               # "linear" or "sin"

# Scaling parameters
Ns: [200, 1000, 2000, 10000]  # Path lengths
Ds: [2, 3, 5, 7, 10]           # Dimensions
Ms: [2, 3, 4, 5, 6]            # Signature levels

operations: ["signature", "logsignature"]
runs_dir: "runs"
repeats: 10                    # Statistical samples per benchmark
logsig_method: "S"             # iisignature method: "S" or "O"
```

**Note:** Large configurations (high N × d × m) can take significant time. Start small for testing.

---

## 📊 Output Format

### Unified CSV Schema

All benchmarks use the same schema for easy comparison:

```csv
N,d,m,path_kind,operation,language,library,method,path_type,t_ms,alloc_KiB
1000,3,4,sin,signature,julia,ChenSignatures.jl,signature_path,Matrix,12.5,245.2
1000,3,4,sin,signature,julia,ChenSignatures.jl,signature_path,Vector{SVector},11.8,180.0
1000,3,4,sin,signature,python,iisignature,sig,ndarray,15.3,312.8
1000,3,4,sin,signature,python,pysiglib,signature,ndarray,142.1,890.2
```

**Columns:**
- `N`, `d`, `m`: Problem parameters
- `path_kind`: Path type (`linear` or `sin`)
- `operation`: `signature` or `logsignature`
- `language`: `julia` or `python`
- `library`: Implementation name
- `method`: API function called
- `path_type`: Input format (`Matrix`, `Vector{SVector}`, `ndarray`)
- `t_ms`: Time in milliseconds (best of `repeats` runs)
- `alloc_KiB`: Peak memory allocation in KiB

### Comparison CSV

The orchestrator produces `comparison.csv` with speedup ratios:

```csv
N,d,m,path_kind,operation,julia_library,julia_path_type,python_library,t_ms_julia,t_ms_python,speed_ratio_python_over_julia,...
```

**Interpretation:** `speed_ratio > 1.0` means Julia is faster.

### Validation CSV

`check_signatures.py` produces `validation_results.csv`:

```csv
N,d,m,path_kind,operation,python_library,len_sig,max_abs_diff,l2_diff,rel_l2_diff,status
```

**Status:** `OK` (passed) or `FAIL` (numerical difference exceeds tolerance).

---

## 🔧 Advanced Usage

### Custom Output Location (Orchestrator Mode)

Scripts can be called with ENV variable override:

```bash
# Direct output to specific file
BENCHMARK_OUT_CSV=/tmp/my_results.csv julia benchmark.jl
```

This is used internally by `compare_benchmarks.py` to coordinate outputs into a single comparison folder.

### Dual Mode Operation

Each script operates in two modes:

1. **Standalone:** Creates its own folder with full logging
   ```bash
   julia benchmark.jl  # → runs/benchmark_julia_*/
   ```

2. **Orchestrated:** Writes to orchestrator's folder
   ```bash
   # Called by compare_benchmarks.py with ENV override
   BENCHMARK_OUT_CSV=path/to/output.csv julia benchmark.jl
   ```

---

## 📈 Interpreting Results

### Performance Comparison

From `SUMMARY.txt` in comparison runs:

```
ChenSignatures.jl(Matrix) vs iisignature: avg speedup = 5.00x
ChenSignatures.jl(Vector{SVector}) vs iisignature: avg speedup = 5.40x
```

- **Matrix vs Vector{SVector}:** Vector{SVector} is typically 5-10% faster
- **vs iisignature:** Julia is 5-7× faster
- **vs pysiglib:** Julia is 25-30× faster

### Validation

From `SUMMARY.txt` in validation runs:

```
Total tests: 24
Passed:      24
Failed:      0
Pass rate:   100.0%
```

All implementations should agree within numerical tolerance (`rel_err < 1e-7`).

---

## 🛠️ Development

### Adding a New Library

1. **Python:** Add benchmark function to `benchmark.py`
2. **Julia:** Add dispatch to `benchmark.jl`
3. Schema automatically handles new entries

### Modifying Path Generators

Edit `common.py` (Python) or `benchmark.jl` (Julia):
- Add function: `make_path_custom(d, N)`
- Update `make_path()` dispatcher
- Add to config: `path_kind: "custom"`

---

## 📝 Files Overview

| File | Purpose |
|------|---------|
| `benchmark.jl` | Julia benchmark runner |
| `benchmark.py` | Python benchmark runner |
| `check_signatures.py` | Correctness validation |
| `compare_benchmarks.py` | Orchestrator (runs all + comparison) |
| `common.py` | Shared Python utilities (config, paths, logging) |
| `generate_fixtures.py` | Generate test fixtures for unit tests |
| `sigcheck.jl` | Helper for validation (called by Python) |
| `benchmark_config.yaml` | Configuration file |

---

## 🐛 Troubleshooting

### "iisignature not available"

```bash
cd benchmark
uv add iisignature
```

### "Julia Project.toml not found"

Run from repo root or set `JULIA_PROJECT`:
```bash
JULIA_PROJECT=. julia benchmark/benchmark.jl
```

### Plots not showing / Matplotlib error

```bash
uv add matplotlib
```

### Benchmarks taking too long

Reduce grid size in `benchmark_config.yaml`:
```yaml
Ns: [200, 1000]    # Fewer values
Ds: [2, 3]
Ms: [2, 3, 4]
repeats: 5         # Fewer samples
```

---

## 📚 References

- **ChenSignatures.jl:** Main library being benchmarked
- **iisignature:** [GitHub](https://github.com/bottler/iisignature)
- **pysiglib:** [GitHub](https://github.com/crispitagorico/pysiglib)

---

## 📄 License

Same as parent project (ChenSignatures.jl).