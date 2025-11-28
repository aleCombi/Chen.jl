from pathlib import Path
import numpy as np
from juliacall import Main as jl
from pathlib import Path
import re

def _get_julia_version():
    """Read version from Julia's Project.toml"""
    project_toml = Path(__file__).parent.parent.parent / "Project.toml"
    if not project_toml.exists():
        return None
    
    content = project_toml.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None

def _find_local_project():
    """
    Detect whether we're running from the ChenSignatures.jl repo checkout.

    Layout assumed:
      repo_root/
        Project.toml
        python/
          chen/
            __init__.py  (this file)
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]  # python/chen/__init__.py -> repo root
    if (repo_root / "Project.toml").exists():
        return repo_root
    return None


def _ensure_chen_loaded():
    local = _find_local_project()

    if local is not None:
        # Local development
        proj = local.as_posix()
        jl.seval(f"""
            import Pkg
            if !haskey(Pkg.project().dependencies, "ChenSignatures")
                Pkg.develop(path = "{proj}")
            end
        """)
    else:
        # Installed: pin to version from Julia Project.toml
        version = _get_julia_version()
        if version:
            jl.seval(f"""
                import Pkg
                if !haskey(Pkg.project().dependencies, "ChenSignatures")
                    Pkg.add(name="ChenSignatures", version="{version}")
                end
            """)
        else:
            jl.seval(r"""
                import Pkg
                if !haskey(Pkg.project().dependencies, "ChenSignatures")
                    Pkg.add("ChenSignatures")
                end
            """)

    jl.seval("using ChenSignatures")

# Initialize Julia environment on import
_ensure_chen_loaded()


def sig(path, m: int) -> np.ndarray:
    """
    Compute the truncated signature of the path up to level m.

    Args:
        path: (N, d) array-like input
        m: truncation level
        use_enzyme: If True, use Enzyme-compatible implementation (slower but differentiable).
                    Default is False (uses optimized implementation).

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