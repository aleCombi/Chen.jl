# ChenSignatures.jl

**ChenSignatures.jl** is a high-performance Julia library for computing **path signatures** and **log-signatures**—core tools in Rough Path Theory, stochastic analysis, and modern deep-learning architectures such as Neural CDEs.

It provides state-of-the-art speed, sparse algebra support, and a clean API for both research and production use.

[![Build Status](https://github.com/aleCombi/ChenSignatures.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/ChenSignatures.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://aleCombi.github.io/ChenSignatures.jl)
[![PyPI](https://img.shields.io/pypi/v/chen-signatures)](https://pypi.org/project/chen-signatures/)

---

## 📐 Overview

Path signatures are sequences of iterated integrals that encode the geometric structure of a path.
They form a powerful, coordinate-free representation used in:

- financial time series and derivatives pricing
- machine learning (Neural CDEs, feature engineering)
- stochastic analysis and rough path theory  

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
  - [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl)
  - Native Julia SIMD vectorization
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

## 📊 Benchmarks

### Cross-Language Benchmarks

For comprehensive performance comparisons against **iisignature** and **pysiglib**, see the dedicated benchmark repository:

👉 **[aleCombi/sig-benchmarks](https://github.com/aleCombi/sig-benchmarks)**

This external suite provides:
- Isolated environment testing (Python vs Julia)
- Multiple library comparisons
- Detailed performance profiles and visualizations
- Methodologically fair benchmarking

### Internal Julia Benchmarks

For quick development regression testing, see:

📁 **[`benchmark/`](./benchmark)**

Run Julia-only benchmarks:
```bash
julia --project=benchmark benchmark/benchmark.jl
```

---

## 📦 Dependencies

ChenSignatures.jl uses a minimal, high-performance stack:

- [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl)
- `LinearAlgebra`
- [`ChainRulesCore.jl`](https://github.com/JuliaDiff/ChainRulesCore.jl) for AD support
- [`Enzyme.jl`](https://github.com/EnzymeAD/Enzyme.jl) for automatic differentiation  

---

## 🚀 Example

```julia
using ChenSignatures

path = randn(1000, 5)
sig_result = sig(path, 4)

# For logsig, need to prepare basis first
basis = prepare(5, 4)  # dimension=5, level=4
logsig_result = logsig(path, basis)
```

---

## 📖 Documentation

For detailed API documentation, mathematical background, and usage examples, see:

👉 **[ChenSignatures.jl Documentation](https://aleCombi.github.io/ChenSignatures.jl)**

---

## 🧪 Testing

The repository includes:

- Full Julia test suite (run with `julia --project -e 'using Pkg; Pkg.test()'`)
- Cross-language validation fixtures at [`test/validation/`](./test/validation)
- Automatic fixture generation via [`test/validation/generate_fixtures.py`](./test/validation/generate_fixtures.py)  

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

Issues and feedback are welcome!
However, pull requests are not being accepted at this time.

👉 https://github.com/aleCombi/ChenSignatures.jl/issues

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