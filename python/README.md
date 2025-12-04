# chen-signatures

Fast rough path signatures for Python, powered by a high-performance Julia backend.

[![PyPI](https://img.shields.io/pypi/v/chen-signatures)](https://pypi.org/project/chen-signatures/)
[![Python](https://img.shields.io/pypi/pyversions/chen-signatures)](https://pypi.org/project/chen-signatures/)

`chen-signatures` brings the speed and numerical stability of Julia’s **ChenSignatures.jl** to Python via `juliacall`, offering a modern, actively maintained alternative to existing signature libraries.

It has been benchmarked against both **iisignature** and **pysiglib**, showing:

- **Comparable performance to pysiglib**, a modern C++/Python implementation  
- **Consistently faster performance than iisignature** across typical configurations  

Full benchmark notebooks and articles will be published separately.

---

## Installation

```bash
pip install chen-signatures
```

On first import, `juliacall` will automatically install a lightweight Julia runtime.
This happens **once per environment**.

---

## Quick Start

```python
import chen
import numpy as np

path = np.random.randn(1000, 10)

# Compute signature
signature = chen.sig(path, m=5)

# Compute log-signature (requires prepare step)
basis = chen.prepare_logsig(d=10, m=5)  # d must match path dimension
logsignature = chen.logsig(path, basis)
```

---

## API

### `sig(path, m)`

Compute a truncated signature up to level `m`.

```python
signature = chen.sig(path, m=3)
```

**Args:**
- `path`: (N, d) array-like where N is path length, d is dimension
- `m`: truncation level (positive integer)

**Returns:** 1D numpy array of signature coefficients

---

### `prepare_logsig(d, m)`

Precompute and cache the Lyndon basis for log-signature computation.

```python
basis = chen.prepare_logsig(d=10, m=5)
```

**Args:**
- `d`: path dimension (positive integer)
- `m`: truncation level (positive integer)

**Returns:** Basis cache object (reusable across multiple paths)

---

### `logsig(path, basis)`

Compute log-signatures using a precomputed Lyndon basis.

```python
basis = chen.prepare_logsig(d=10, m=5)
logsig = chen.logsig(path, basis)
```

**Args:**
- `path`: (N, d) array-like where N is path length, d is dimension
- `basis`: Basis object from `prepare_logsig(d, m)`

**Returns:** 1D numpy array of log-signature coefficients (Lyndon basis projection)

**Note:** The `basis` object should match the dimension of your path. Prepare it once and reuse for multiple paths of the same dimension.

---

## Supported Types

- `float32`, `float64`
- Any NumPy array-like input  
- Contiguous arrays recommended (handled automatically)

---

## Use Cases

- Financial time series and derivatives pricing
- Neural CDEs and machine learning
- Feature engineering and representation learning  

---

## Limitations

- First import is slow (Julia installation)
- CPU-only execution
- Uses more memory than minimal C++ libraries

---

## Requirements

- Python ≥ 3.9  
- NumPy ≥ 1.20  
- ~500MB disk space for Julia runtime

---

## Citation

```bibtex
@software{chen_signatures,
  author = {Combi, Alessandro},
  title = {chen-signatures: Fast rough path signatures for Python},
  year = {2025},
  url = {https://github.com/aleCombi/ChenSignatures.jl}
}
```

---

## Contributing

Issues and feedback are welcome at https://github.com/aleCombi/ChenSignatures.jl/issues

However, pull requests are not being accepted at this time.
