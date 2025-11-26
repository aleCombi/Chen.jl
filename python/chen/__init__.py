from pathlib import Path
import numpy as np
from juliacall import Main as jl

def _find_local_project():
    """
    Detects if we are running from a local development environment
    (i.e. the Chen.jl repository exists two levels up).
    """
    this_file = Path(__file__).resolve()
    # python/chen/__init__.py -> parents[2] is the repo root
    repo_root = this_file.parents[2]
    if (repo_root / "Project.toml").exists():
        return repo_root
    return None


def _ensure_chen_loaded():
    """
    Ensures the Julia backend is installed and loaded.
    - If local dev: `Pkg.develop` the local path.
    - If installed user: `Pkg.add` from GitHub (branch main).
    """
    local = _find_local_project()

    if local is not None:
        proj = local.as_posix()
        # In dev mode, we want to use the local code immediately
        jl.seval(f'import Pkg; Pkg.develop(path="{proj}")')
    else:
        # In production mode, we pull from GitHub.
        # Added rev="main" to ensure users get the latest fixes (like Float32 support)
        jl.seval(
            """
            import Pkg
            Pkg.add(Pkg.PackageSpec(
                url="https://github.com/aleCombi/Chen.jl",
                rev="python_package" 
            ))
            """
        )

    # Load the package into the Julia session
    jl.seval("using Chen")


# Initialize Julia environment on import
_ensure_chen_loaded()


def sig(path, m: int) -> np.ndarray:
    """
    Compute the truncated signature of the path up to level m.
    
    Args:
        path: (N, d) array-like input
        m: truncation level
    Returns:
        (d + d^2 + ... + d^m,) flattened array
    """
    # Ensure memory is contiguous for optimal transfer to Julia
    arr = np.ascontiguousarray(path)
    # Call Julia function
    res = jl.Chen.sig(arr, m)
    # Convert Julia Vector back to NumPy array
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
    
    # We hold the BasisCache object in Python as a managed Julia object
    basis = jl.Chen.prepare(d, m)
    
    res = jl.Chen.logsig(arr, basis)
    return np.asarray(res)