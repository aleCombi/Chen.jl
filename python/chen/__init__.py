from pathlib import Path
import numpy as np
from importlib.metadata import version, PackageNotFoundError
import re

def _setup_julia_package():
    """
    Configure ChenSignatures Julia package based on environment.
    
    - Development mode: Use local package via path
    - Installed mode: Use General Registry with version matching Python package
    """
    import juliapkg
    
    this_file = Path(__file__).resolve()
    python_root = this_file.parents[1] # python/
    repo_root = this_file.parents[2]   # git root/
    
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
    # Determine the Python package version to enforce strict sync.
    try:
        # Preferred: Get version from installed package metadata
        pkg_version = version("chen-signatures")
        
    except PackageNotFoundError:
        # Fallback: Parse pyproject.toml directly (e.g., running from source without install)
        pyproject_path = python_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding="utf-8")
            # Regex to match: version = "0.2.1"
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                pkg_version = match.group(1)
            else:
                raise RuntimeError(f"Could not parse version from {pyproject_path}")
        else:
            raise RuntimeError(
                "Could not determine package version. "
                "Package is not installed and pyproject.toml was not found."
            )

    # Enforce strict version equality (e.g. "=0.2.1")
    juliapkg.add(
        "ChenSignatures",
        uuid="4efb4129-5e83-47d2-926d-947c0e6cb76d",
        version=f"={pkg_version}"
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