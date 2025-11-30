# ChenSignatures.jl

**ChenSignatures.jl** is a high-performance Julia library for computing **path signatures** and **log-signatures**—core tools in Rough Path Theory, stochastic analysis, and modern deep-learning architectures such as Neural CDEs.

It provides state-of-the-art speed, sparse algebra support, and a clean API for both research and production use.

[![Build Status](https://github.com/aleCombi/ChenSignatures.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/ChenSignatures.jl/actions/workflows/CI.yml?query=branch%3Amaster)  
[![PyPI](https://img.shields.io/pypi/v/chen-signatures)](https://pypi.org/project/chen-signatures/)

---

## 📐 Overview

Path signatures are sequences of iterated integrals that encode the geometric structure of a path.  
They form a powerful, coordinate-free representation used in:

- time-series modelling  
- machine learning (Neural CDEs, RNN augmentation)  
- stochastic analysis  
- control theory  
- anomaly detection  
- feature engineering  

**ChenSignatures.jl** computes both **signatures** and **log-signatures** efficiently and accurately, with full support for dense and sparse tensor algebras.

---

## ✅ Key Features

### Core Functionality
- **Signatures:** Compute truncated path signatures up to any depth.  
- **Log-Signatures:** Lyndon-projected log-signatures for minimal bases.  
- **Tensor Algebra:** Dense and sparse tensor algebra (`SparseTensor`).  
- **Algebraic Operations:** Shuffle product, tensor exp/log, resolvent operators.

### Performance
- Highly optimized kernels using:
  - [`LoopVectorization.jl`](https://github.com/JuliaSIMD/LoopVectorization.jl)  
  - [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl)
- Designed to scale across:
  - long paths (large `N`)  
  - high dimensions (`d`)  
  - large truncation levels (`m`)

### Validation
Benchmarked and numerically cross-validated against:

- **iisignature**  
- **pysiglib**

Full benchmarking and correctness tooling is included (see below).

---

## 🐍 Python API: `chen-signatures`

ChenSignatures.jl powers the Python package:

📦 **PyPI:** https://pypi.org/project/chen-signatures/  
📁 **Source:** [`python/`](./python)

```bash
pip install chen-signatures
```

Features:
- Same API as Julia (`sig`, `logsig`)  
- Works on Python **3.9–3.13**  
- Automatically installs a lightweight Julia runtime via `juliacall`  
- Faster than `iisignature` and comparable to `pysiglib`  

---

## 📊 Benchmark Suite

A reproducible, multi-library benchmark suite lives in:

📁 **[`benchmark/`](./benchmark)**  
📁 **[`python/`](./python)** (Python-side benchmarking tools)

The suite compares:

- ChenSignatures.jl  
- iisignature  
- pysiglib  

and measures:
- runtime  
- memory  
- speed ratios  
- correctness differences  

Run everything with:

```bash
uv run run_benchmark.py
```

This creates structured outputs:

```
runs/
  2025-.../
      config.yaml
      run_julia.csv
      run_python.csv
      comparison.csv
      logs/
      plots/
```

---

## 📦 Dependencies

ChenSignatures.jl uses a minimal, high-performance stack:

- [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl)  
- [`LoopVectorization.jl`](https://github.com/JuliaSIMD/LoopVectorization.jl)  
- `LinearAlgebra`  

---

## 🚀 Example

```julia
using ChenSignatures

path = randn(1000, 5)
sig = signature(path, 4)
logsig = logsignature(path, 4)
```

---

## 🧪 Testing

The repository includes:

- full Julia test suite  
- cross-language correctness checking (`benchmark/check_signatures.py`)  
- automatic fixture generation via Python (`generate_fixtures.py`)  

---

## 📚 Citation

```bibtex
@software{chen_signatures,
  author = {Combi, Alessandro},
  title = {ChenSignatures.jl: High-performance signatures and log-signatures},
  year = {2025},
  url = {https://github.com/aleCombi/ChenSignatures.jl}
}
```

---

## 🤝 Contributing

Contributions welcome!  
Open issues or pull requests here:

👉 https://github.com/aleCombi/ChenSignatures.jl

## Version Management

**For Maintainers:**

Version is managed in `Project.toml` only. Python package reads it automatically.

To release a new version:
1. Edit `Project.toml`: `version = "0.3.0"`
2. Commit: `git commit -am "Bump version to 0.3.0"`
3. Register with Julia General (comment `@JuliaRegistrator register` on PR/commit)
4. TagBot creates `v0.3.0` tag automatically
5. Tag triggers PyPI publish automatically

**Never** edit version in `python/pyproject.toml` - it's computed dynamically.