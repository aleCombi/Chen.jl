from pathlib import Path
import numpy as np

def _setup_julia_package():
    """
    Configure ChenSignatures Julia package based on environment.
    
    - Development mode: Use local package via path
    - Installed mode: Use GitHub URL
    """
    import juliapkg
    
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]  # python/chen/__init__.py -> repo root
    
    if (repo_root / "Project.toml").exists():
        # Development mode - use local package
        juliapkg.add(
            "ChenSignatures",
            uuid="4efb4129-5e83-47d2-926d-947c0e6cb76d",
            path=str(repo_root),
            dev=True
        )
        return True
    else:
        # Installed mode - use GitHub URL
        juliapkg.add(
            "ChenSignatures",
            uuid="4efb4129-5e83-47d2-926d-947c0e6cb76d",
            url="https://github.com/aleCombi/ChenSignatures.jl.git"
        )
        return False

# Setup Julia package before importing juliacall
_is_dev = _setup_julia_package()

# Import juliacall - this will use juliapkg to set up the Julia environment
from juliacall import Main as jl

# Load ChenSignatures (juliapkg already added it to the environment)
jl.seval("using ChenSignatures")


def sig(path, m: int) -> np.ndarray:
    """
    Compute the truncated signature of the path up to level m.

    Args:
        path: (N, d) array-like input
        m: truncation level

    Returns:
        (d + d^2 + ... + d^m,) flattened array
    """
    arr = np.ascontiguousarray(path, dtype=np.float64)
    res = jl.ChenSignatures.sig(arr, m)
    
    return np.asarray(res)

def logsig(path, m: int) -> np.ndarray:
    """
    Compute the log-signature projected onto the Lyndon basis.

    Args:
        path: (N, d) array-like input
        m: truncation level

    Returns:
        Array of log-signature coefficients
    """
    arr = np.ascontiguousarray(path)
    d = arr.shape[1]

    basis = jl.ChenSignatures.prepare(d, m)
    res = jl.ChenSignatures.logsig(arr, basis)
    return np.asarray(res)