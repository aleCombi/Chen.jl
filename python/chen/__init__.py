# python/chen/__init__.py
"""
chen-signatures: Fast path signatures powered by Julia
"""
from pathlib import Path
import numpy as np

# Import version from our dual-mode version handler
from chen._version import __version__

def _setup_julia_package():
    """
    Configure ChenSignatures Julia package based on environment.
    
    - Development mode: Use local package via path
    - Installed mode: Use General Registry with version matching Python package
    
    Returns:
        bool: True if in development mode, False if in production mode
    """
    import juliapkg
    
    this_file = Path(__file__).resolve()
    python_root = this_file.parent          # python/chen/
    repo_root = python_root.parents[1]      # repo root/
    
    # 1. DEVELOPMENT MODE
    # If the Julia Project.toml exists in the root, link it directly.
    if (repo_root / "Project.toml").exists():
        juliapkg.add(
            "ChenSignatures",
            uuid="4efb4129-5e83-47d2-926d-947c0e6cb76d",
            path=str(repo_root),
            dev=True
        )
        return True

    # 2. INSTALLED / PRODUCTION MODE
    # Use version from __version__ (which came from package metadata or Project.toml)
    juliapkg.add(
        "ChenSignatures",
        uuid="4efb4129-5e83-47d2-926d-947c0e6cb76d",
        version=f"={__version__}"
    )
    return False

# Setup Julia package before importing juliacall
_is_dev = _setup_julia_package()

# Import juliacall - this will use juliapkg to set up the Julia environment
from juliacall import Main as jl

# Load ChenSignatures (juliapkg already added it to the environment)
jl.seval("using ChenSignatures")


# ============================================================================
# Public API
# ============================================================================

def sig(path, m: int) -> np.ndarray:
    """
    Compute the truncated signature of the path up to level m.

    Args:
        path: (N, d) array-like input where N is path length, d is dimension
        m: truncation level (must be positive integer)

    Returns:
        (d + d^2 + ... + d^m,) flattened array of signature coefficients

    Examples:
        >>> import chen
        >>> import numpy as np
        >>> path = np.random.randn(100, 3)
        >>> signature = chen.sig(path, m=4)
        >>> signature.shape
        (120,)  # 3 + 9 + 27 + 81

    Raises:
        ValueError: If path has fewer than 2 points or m is not positive
    """
    # Ensure contiguous float64 array (Julia's default float type)
    arr = np.ascontiguousarray(path, dtype=np.float64)
    
    # Call Julia function
    res = jl.ChenSignatures.sig(arr, m)
    
    # Convert back to numpy
    return np.asarray(res)


def prepare_logsig(d: int, m: int):
    """
    Precompute and cache the Lyndon basis and projection matrix for log-signature.

    This wraps the Julia function `ChenSignatures.prepare(d, m)`, which returns a
    `BasisCache` object containing:
        - d: path dimension
        - m: truncation level
        - lynds: Lyndon words
        - L: projection matrix

    Args:
        d: path dimension (must be a positive integer)
        m: truncation level (must be a positive integer)

    Returns:
        A Julia `BasisCache` object that can be reused in subsequent `logsig` calls.
    """
    if d <= 0:
        raise ValueError("Dimension d must be a positive integer")
    if m <= 0:
        raise ValueError("Truncation level m must be a positive integer")

    # Calls the Julia function:
    #   function prepare(d::Int, m::Int)::BasisCache
    return jl.ChenSignatures.prepare(d, m)


def logsig(path, basis) -> np.ndarray:
    """
    Compute the log-signature of a path using a precomputed basis.

    This wraps the Julia function:
        ChenSignatures.logsig(path::AbstractMatrix, basis::BasisCache)

    It is analogous to `iisignature.logsig(path, prep)`, where `basis` is the
    result of `prepare_logsig(d, m)`.

    Args:
        path: (N, d) array-like input where N is path length, d is dimension.
        basis: Basis object returned by `prepare_logsig(d, m)` (a Julia `BasisCache`).

    Returns:
        1D numpy array of log-signature coefficients (flattened Lyndon basis projection).
    """
    # Convert to contiguous float64 2D array
    arr = np.ascontiguousarray(path, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`path` must be 2D (N, d); got shape {arr.shape}")

    d = arr.shape[1]

    # Sanity check: match Python path dimension with Julia BasisCache.d if available
    basis_d = getattr(basis, "d", None)
    if basis_d is not None and basis_d != d:
        raise ValueError(
            f"Dimension mismatch between path (d={d}) and basis (d={basis_d}). "
            "Did you call prepare_logsig with the correct dimension?"
        )

    # Call Julia:
    #   function logsig(path::AbstractMatrix{T}, basis::BasisCache)::Vector{T}
    res = jl.ChenSignatures.logsig(arr, basis)

    # Convert Julia vector to numpy array
    return np.asarray(res)


# ============================================================================
# Package metadata
# ============================================================================

__all__ = [
    '__version__',
    'sig',
    'logsig',
    'prepare_logsig',
]
