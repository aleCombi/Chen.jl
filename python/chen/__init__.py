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


def logsig(path, m: int) -> np.ndarray:
    """
    Compute the log-signature projected onto the Lyndon basis.

    The log-signature is the logarithm of the signature, projected onto
    a minimal Lyndon basis. This provides a more compact representation.

    Args:
        path: (N, d) array-like input where N is path length, d is dimension
        m: truncation level (must be positive integer)

    Returns:
        Array of log-signature coefficients (typically smaller than signature)

    Examples:
        >>> import chen
        >>> import numpy as np
        >>> path = np.random.randn(100, 3)
        >>> logsignature = chen.logsig(path, m=4)
        >>> logsignature.shape
        (18,)  # Much smaller than sig(path, 4).shape = (120,)

    Notes:
        This function uses a precomputed Lyndon basis for efficiency.
        The basis is cached internally by Julia.
    """
    # Ensure contiguous array
    arr = np.ascontiguousarray(path, dtype=np.float64)
    d = arr.shape[1]

    # Prepare basis (Julia caches this internally)
    basis = jl.ChenSignatures.prepare(d, m)
    
    # Compute log-signature
    res = jl.ChenSignatures.logsig(arr, basis)
    
    # Convert back to numpy
    return np.asarray(res)

def prepare_logsig(d: int, m: int):
    """
    Precompute and cache the Lyndon basis / internal structure for log-signature.

    This is analogous to `iisignature.prepare(d, m)`.

    Args:
        d: path dimension
        m: truncation level (must be positive integer)

    Returns:
        A Julia-side basis object that can be reused in subsequent `logsig_from_basis` calls.
    """
    if m <= 0:
        raise ValueError("Truncation level m must be a positive integer")
    if d <= 0:
        raise ValueError("Dimension d must be a positive integer")

    return jl.ChenSignatures.prepare(d, m)


def logsig(path, basis) -> np.ndarray:
    """
    Compute the log-signature using a precomputed basis.

    This is analogous to `iisignature.logsig(path, prep)`.

    Args:
        path: (N, d) array-like input where N is path length, d is dimension
        basis: object returned by `prepare_logsig(d, m)`

    Returns:
        1D numpy array of log-signature coefficients.
    """
    # Ensure contiguous array
    arr = np.ascontiguousarray(path, dtype=np.float64)

    # Optional: sanity check dimension match (if basis has `d` and `m` fields in Julia)
    # You can skip this if it's too expensive / awkward:
    # d = arr.shape[1]
    # assert basis.d == d, "Path dimension does not match prepared basis"

    res = jl.ChenSignatures.logsig(arr, basis)
    return np.asarray(res)


# ============================================================================
# Package metadata
# ============================================================================

__all__ = [
    '__version__',
    'sig',
    'logsig',
]