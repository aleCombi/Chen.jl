# chen-signatures

Fast rough path signatures for Python, powered by Julia.

**1.6× faster** than iisignature with support for **modern Python** (3.9-3.13).

[![PyPI](https://img.shields.io/pypi/v/chen-signatures)](https://pypi.org/project/chen-signatures/)
[![Python](https://img.shields.io/pypi/pyversions/chen-signatures)](https://pypi.org/project/chen-signatures/)

## Why chen-signatures?

| Feature | chen-signatures | iisignature |
|---------|-----------------|-------------|
| **Speed** | 187 ms | 299 ms |
| **Python 3.10+** | ✅ Yes | ❌ No (≤3.9 only) |
| **Python 3.13** | ✅ Yes | ❌ No |
| **Autodiff** | ✅ Yes (ForwardDiff) | ❌ No |
| **Maintained** | ✅ Active | ⚠️ Unmaintained |

*Benchmark: N=1000 points, d=10 dims, level=5*

## Installation
```bash
pip install chen-signatures
```

First import will automatically install Julia (via juliacall). Takes ~2 minutes once.

## Quick Start
```python
import chen
import numpy as np

# Your time series data
path = np.random.randn(1000, 10)  # 1000 timepoints, 10 dimensions

# Compute signature
signature = chen.sig(path, m=5)

# Compute log-signature  
logsignature = chen.logsig(path, m=5)
```

## API

### `sig(path, m)`

Compute the truncated signature up to level `m`.

**Parameters:**
- `path` : `(N, d)` numpy array - Path data
- `m` : int - Truncation level

**Returns:** 
- `numpy.ndarray` - Flattened signature coefficients

**Example:**
```python
import chen
import numpy as np

path = np.array([[0., 0.], [1., 0.], [1., 1.]])
sig = chen.sig(path, m=3)
# Returns array of length d + d² + d³ = 2 + 4 + 8 = 14
```

### `logsig(path, m)`

Compute the log-signature projected onto the Lyndon basis.

**Parameters:**
- `path` : `(N, d)` numpy array - Path data
- `m` : int - Truncation level

**Returns:**
- `numpy.ndarray` - Log-signature in Lyndon basis

**Example:**
```python
logsig = chen.logsig(path, m=5)
```

## Supported Types

- `float32` and `float64`
- Any numpy array-like input

## Performance

Production-scale benchmark (N=1000, d=10, m=5):
```python
import chen
import numpy as np
import time

path = np.random.randn(1000, 10)

t0 = time.time()
sig = chen.sig(path, 5)
print(f"Time: {(time.time()-t0)*1000:.1f} ms")
# Output: Time: 187.4 ms
```

Compare to iisignature: **299.3 ms** (1.6× slower)

## Use Cases

- **Financial time series**: Extract signature features from price data
- **Sensor data**: Process multivariate sensor streams  
- **Neural CDEs**: Differentiable features for neural networks
- **Anomaly detection**: Signature-based feature engineering

## Limitations

- **First import is slow** (~2 min to install Julia environment, one-time)
- **Not GPU-accelerated** (CPU only, but very fast)
- **Memory usage**: Uses more RAM than iisignature (but negligible for most applications)

## Requirements

- Python ≥3.9
- NumPy ≥1.20
- ~500MB disk space (for Julia installation)

## Citation

If you use this in research, please cite:
```bibtex
@software{chen_signatures,
  author = {Combi, Alessandro},
  title = {chen-signatures: Fast rough path signatures for Python},
  year = {2025},
  url = {https://github.com/aleCombi/Chen.jl}
}
```

## License

MIT License - see [LICENSE](../LICENSE) file.

## Contributing

Issues and pull requests welcome at [github.com/aleCombi/Chen.jl](https://github.com/aleCombi/Chen.jl)