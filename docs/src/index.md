# ChenSignatures.jl

**ChenSignatures.jl** is a high-performance Julia library for computing **path signatures** and **log-signatures**—core tools in Rough Path Theory, stochastic analysis, and modern machine learning.

## Overview

This package efficiently computes:
- **Signatures**: Truncated iterated integrals up to any level
- **Log-signatures**: Compact representations using Lyndon bases
- **Batch processing**: Multi-threaded computation of multiple paths
- **Automatic differentiation**: Full gradient support via ChainRulesCore and Enzyme

Path signatures are widely used in financial time series, Neural CDEs, stochastic analysis, and machine learning feature engineering.

---

## What is a Path Signature?

Given a path ``\gamma : [0,T] \to \mathbb{R}^d``, the **signature** ``S(\gamma)`` is the collection of iterated integrals:

```math
S(\gamma) = \left(1, S^{(1)}(\gamma), S^{(2)}(\gamma), \ldots, S^{(m)}(\gamma)\right)
```

where each level ``k`` contains ``d^k`` tensor coefficients:

```math
S^{(k)}_{i_1, \ldots, i_k}(\gamma) = \int_{0 < t_1 < \cdots < t_k < T} d\gamma^{i_1}_{t_1} \cdots d\gamma^{i_k}_{t_k}
```

For discrete paths, these are computed recursively using **Chen's identity**:

```math
S(\gamma_{[s,t]}) = S(\gamma_{[s,u]}) \otimes S(\gamma_{[u,t]})
```

**Key property**: Signatures uniquely characterize paths up to tree-like equivalence and are naturally invariant to time reparametrization.

---

## Quick Start

### Computing Signatures

```julia
using ChenSignatures

# Create a 2D path with 100 points
path = randn(100, 2)

# Compute signature up to level 4
s = sig(path, 4)

# Output is a flattened vector: length = 2 + 4 + 8 + 16 = 30
length(s)  # 30
```

### Computing Log-Signatures

The **log-signature** is the logarithm of the signature projected onto the **Lyndon basis**, providing a minimal representation:

```julia
# Precompute Lyndon basis (dimension=3, level=4)
basis = prepare(3, 4)

# Generate a 3D path
path = randn(50, 3)

# Compute log-signature
ls = logsig(path, basis)

# Log-signature is more compact than full signature
println("Log-signature length: ", length(ls))
println("Full signature would be: ", 3 + 9 + 27 + 81, " coefficients")
```

**Tip**: Reuse the `basis` for multiple paths with the same dimension and truncation level.

### Batch Processing

Process multiple paths efficiently with built-in batching:

```julia
# Batch of 100 paths, each 50×3 (50 points, 3 dimensions)
paths = randn(50, 3, 100)

# Compute all signatures at once with multi-threading
sigs = sig(paths, 4)  # Returns matrix of size (signature_length, 100)

# Or log-signatures
basis = prepare(3, 4)
logsigs = logsig(paths, basis, threaded=true)
```

### Path Augmentations

Common preprocessing steps are built in:

```julia
aug = time_augment(path; Tspan=1.0)  # add monotone time as first coordinate
ll  = lead_lag(path)                 # standard lead–lag expansion

sig_time(path, 4)      # = sig(time_augment(path), 4)
sig_leadlag(path, 4)   # = sig(lead_lag(path), 4)
```

---

## Python Wrapper

A Python package **chen-signatures** provides the same high-performance functionality via `juliacall`:

```bash
pip install chen-signatures
```

```python
import chen
import numpy as np

path = np.random.randn(1000, 10)
signature = chen.sig(path, m=5)
```

**Resources**:
- PyPI: [https://pypi.org/project/chen-signatures/](https://pypi.org/project/chen-signatures/)
- Python Documentation: [python/README.md](https://github.com/aleCombi/ChenSignatures.jl/blob/master/python/README.md)

The Python wrapper supports NumPy arrays, PyTorch integration with autograd, and maintains the same performance characteristics as the Julia implementation.

---

## Performance

ChenSignatures.jl is highly performant with optimizations for long paths, high dimensions, and deep truncation levels.

**Benchmarks**: Comprehensive comparisons against **iisignature** and **pysiglib** are available at:
- [aleCombi/sig-benchmarks](https://github.com/aleCombi/sig-benchmarks)

Reproducible benchmark results and detailed performance analysis will be published in the future.

---

## Automatic Differentiation

Full support for automatic differentiation:

```julia
using ChenSignatures
using Zygote

path = randn(50, 3)

# Gradient with respect to path
grad = gradient(p -> sum(sig(p, 4)), path)
```

Both `ChainRulesCore` and `Enzyme.jl` integration are provided.

---

## Links

- **Julia Package**: [GitHub Repository](https://github.com/aleCombi/ChenSignatures.jl)
- **Julia README**: [README.md](https://github.com/aleCombi/ChenSignatures.jl/blob/master/README.md)
- **Python Package**: [PyPI](https://pypi.org/project/chen-signatures/)
- **Python README**: [python/README.md](https://github.com/aleCombi/ChenSignatures.jl/blob/master/python/README.md)
- **Benchmarks**: [sig-benchmarks](https://github.com/aleCombi/sig-benchmarks)

---

## Citation

```bibtex
@software{chen_signatures,
  author = {Combi, Alessandro},
  title = {ChenSignatures.jl: High-performance signatures and log-signatures},
  year = {2025},
  url = {https://github.com/aleCombi/ChenSignatures.jl}
}
```

---

## References

**Path Signatures and Rough Path Theory:**
- K.T. Chen (1957). "Integration of paths, geometric invariants and a generalized Baker-Hausdorff formula."
- T. Lyons, M. Caruana, T. Lévy (2007). "Differential equations driven by rough paths." Lecture Notes in Mathematics, Springer.

**Lyndon Basis:**
- C. Reutenauer (1993). "Free Lie Algebras." Oxford University Press.
