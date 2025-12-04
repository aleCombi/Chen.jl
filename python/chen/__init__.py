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

def sig(path, m: int, threaded: bool = True) -> np.ndarray:
    """
    Compute the truncated signature of a path or batch of paths up to level m.

    Args:
        path: Array-like input of shape:
            - (N, d): Single path where N is path length, d is dimension
            - (N, d, B): Batch of B paths (uses multi-threading by default)
        m: Truncation level (must be positive integer)
        threaded: If True (default), use multi-threading for batch processing.
                 Ignored for single path input.

    Returns:
        - Single path (N, d): Returns (d + d^2 + ... + d^m,) array
        - Batch (N, d, B): Returns (d + d^2 + ... + d^m, B) array where
          each column is the signature of one path

    Examples:
        >>> import chen
        >>> import numpy as np

        # Single path
        >>> path = np.random.randn(100, 3)
        >>> signature = chen.sig(path, m=4)
        >>> signature.shape
        (120,)  # 3 + 9 + 27 + 81

        # Batch of paths (with float32)
        >>> paths = np.random.randn(100, 3, 50).astype(np.float32)
        >>> signatures = chen.sig(paths, m=4, threaded=True)
        >>> signatures.shape
        (120, 50)  # (sig_length, batch_size)

    Raises:
        ValueError: If path has fewer than 2 points or m is not positive
    """
    # Convert to numpy array and preserve float dtype (float32/float64)
    arr = np.asarray(path)

    # Default to float64 for integer or non-float types
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    # Ensure contiguous memory layout
    arr = np.ascontiguousarray(arr)

    # Call Julia function (dispatch handles 2D vs 3D)
    # Only pass threaded kwarg for batch (3D) case
    kwargs = {"threaded": threaded} if arr.ndim == 3 else {}
    res = jl.ChenSignatures.sig(arr, m, **kwargs)

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


def logsig(path, basis, threaded: bool = True) -> np.ndarray:
    """
    Compute the log-signature of a path or batch of paths using a precomputed basis.

    This wraps the Julia functions:
        ChenSignatures.logsig(path::AbstractMatrix, basis::BasisCache)
        ChenSignatures.logsig(paths::AbstractArray{T,3}, basis::BasisCache)

    It is analogous to `iisignature.logsig(path, prep)`, where `basis` is the
    result of `prepare_logsig(d, m)`.

    Args:
        path: Array-like input of shape:
            - (N, d): Single path where N is path length, d is dimension
            - (N, d, B): Batch of B paths (uses multi-threading by default)
        basis: Basis object returned by `prepare_logsig(d, m)` (a Julia `BasisCache`).
        threaded: If True (default), use multi-threading for batch processing.
                 Ignored for single path input.

    Returns:
        - Single path (N, d): Returns (L,) array of log-signature coefficients
        - Batch (N, d, B): Returns (L, B) array where each column is the
          log-signature of one path (L = number of Lyndon words)

    Examples:
        >>> import chen
        >>> import numpy as np

        # Single path
        >>> path = np.random.randn(100, 3)
        >>> basis = chen.prepare_logsig(3, 4)
        >>> logsig = chen.logsig(path, basis)
        >>> logsig.shape
        (18,)  # Much more compact than sig (18 vs 120)

        # Batch of paths (with float32)
        >>> paths = np.random.randn(100, 3, 50).astype(np.float32)
        >>> logsigs = chen.logsig(paths, basis, threaded=True)
        >>> logsigs.shape
        (18, 50)  # (num_lyndon_words, batch_size)
    """
    # Convert to numpy array and preserve float dtype (float32/float64)
    arr = np.asarray(path)

    # Default to float64 for integer or non-float types
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    # Ensure contiguous memory layout
    arr = np.ascontiguousarray(arr)

    # Validate dimensions
    if arr.ndim == 2:
        d = arr.shape[1]
    elif arr.ndim == 3:
        d = arr.shape[1]
    else:
        raise ValueError(f"`path` must be 2D (N, d) or 3D (N, d, B); got shape {arr.shape}")

    # Sanity check: match Python path dimension with Julia BasisCache.d if available
    basis_d = getattr(basis, "d", None)
    if basis_d is not None and basis_d != d:
        raise ValueError(
            f"Dimension mismatch between path (d={d}) and basis (d={basis_d}). "
            "Did you call prepare_logsig with the correct dimension?"
        )

    # Call Julia function (dispatch handles 2D vs 3D)
    # Only pass threaded kwarg for batch (3D) case
    kwargs = {"threaded": threaded} if arr.ndim == 3 else {}
    res = jl.ChenSignatures.logsig(arr, basis, **kwargs)

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
