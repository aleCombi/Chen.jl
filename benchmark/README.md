# ChenSignatures.jl Benchmarks

This directory contains the performance benchmarking and validation suite for **ChenSignatures.jl**, comparing it against Python libraries:

- **iisignature**
- **pysiglib**

It covers:

- End-to-end **runtime and allocation benchmarks** (Julia vs Python).
- **Scaling plots** in \(N, d, m\) for signature and log-signature.
- **Correctness checks** of ChenSignatures.jl against iisignature and pysiglib.

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
