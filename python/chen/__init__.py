from pathlib import Path
import numpy as np
from juliacall import Main as jl


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
    """
    Ensure the Julia backend (ChenSignatures.jl) is installed and loaded.

    - Local dev: use Pkg.develop on the repository root.
    - PyPI install: prefer registered package `ChenSignatures`, and
      fall back to adding from GitHub if needed.
    """
    local = _find_local_project()

    if local is not None:
        # Local development: use your local Julia project
        proj = local.as_posix()
        jl.seval(f"""
            import Pkg
            if !haskey(Pkg.project().dependencies, "ChenSignatures")
                Pkg.develop(path = "{proj}")
            end
        """)
    else:
        # Installed-from-PyPI mode
        jl.seval(r"""
            import Pkg
            if !haskey(Pkg.project().dependencies, "ChenSignatures")
                try
                    # Prefer registered package
                    Pkg.add("ChenSignatures")
                catch err
                    @warn "Pkg.add(\"ChenSignatures\") failed, falling back to GitHub url" err
                    Pkg.add(Pkg.PackageSpec(
                        url = "https://github.com/aleCombi/ChenSignatures.jl",
                    ))
                end
            end
        """)

    # Load the Julia module
    jl.seval("using ChenSignatures")


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
    arr = np.ascontiguousarray(path)
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
