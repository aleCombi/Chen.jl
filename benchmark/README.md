# Internal Benchmarks

This directory contains **pure Julia** benchmarks for development and regression testing of **ChenSignatures.jl**.

It allows maintainers to quickly check the performance of the library without setting up Python environments or external dependencies.

> 📊 **Cross-Language Benchmarks:**  
> For comprehensive comparisons against **iisignature** (C++) and **pysiglib** (PyTorch), please see the dedicated benchmark repository:  
> 👉 **[aleCombi/sig-benchmarks](https://github.com/aleCombi/sig-benchmarks)**

---

## 📂 Files

- **[`benchmark.jl`](./benchmark.jl)**: The main script. Runs signatures and log-signatures on generated paths and reports timing/allocations.
- **[`benchmark_config.yaml`](./benchmark_config.yaml)**: Configuration file to control problem sizes ($N, d, m$) and execution parameters.

---

## 🚀 Usage

To run the benchmarks, you only need a working Julia installation.

1. **Instantiate dependencies** (only needs to be done once):
   ```bash
   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   ```

2. **Run the benchmarks**:
   ```bash
   julia --project=benchmark benchmark/benchmark.jl
   ```

### Output

The script will:
1. Print a summary to the console.
2. Create a timestamped folder in `runs/` (e.g., `runs/benchmark_julia_2025.../`) containing:
   - `results.csv`: Detailed timing and allocation data for every configuration.
   - `SUMMARY.txt`: A quick overview of the run.

---

## ⚙️ Configuration

Edit **[`benchmark_config.yaml`](./benchmark_config.yaml)** to customize the sweep:

```yaml
# Grid parameters
Ns: [1000, 10000]      # Path lengths
Ds: [3, 10]            # Dimensions
Ms: [4, 6]             # Signature depths

# Options
path_kind: "sin"       # "linear" or "sin"
repeats: 5             # Samples per test
```

*Tip: Keep the grid small for quick regression checks during development.*