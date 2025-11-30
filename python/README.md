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

signature = chen.sig(path, m=5)
logsignature = chen.logsig(path, m=5)
```

---

## API

### `sig(path, m)`

Compute a truncated signature up to level `m`.

```python
sig = chen.sig(path, m=3)
```

### `logsig(path, m)`

Compute log-signatures using the Lyndon basis.

```python
logsig = chen.logsig(path, m=5)
```

---

## Supported Types

- `float32`, `float64`
- Any NumPy array-like input  
- Contiguous arrays recommended (handled automatically)

---

## Use Cases

- Financial time series  
- Sensor data and IoT  
- Neural CDEs / differential ML  
- Representation learning  
- Anomaly detection  

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

https://github.com/aleCombi/ChenSignatures.jl
