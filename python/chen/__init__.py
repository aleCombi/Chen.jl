from pathlib import Path
import numpy as np
from juliacall import Main as jl


def _find_local_project():
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]  # Chen.jl root in dev
    if (repo_root / "Project.toml").exists():
        return repo_root
    return None


def _ensure_chen_loaded():
    local = _find_local_project()

    if local is not None:
        proj = local.as_posix()
        jl.seval(f'import Pkg; Pkg.develop(path="{proj}")')
    else:
        jl.seval(
            """
            import Pkg
            Pkg.add(Pkg.PackageSpec(
                url="https://github.com/aleCombi/Chen.jl"
                # , rev="v0.1.1"  # optional: add when you start tagging
            ))
            """
        )

    jl.seval("using Chen")


_ensure_chen_loaded()


def sig(path, m: int):
    arr = np.asarray(path)
    return np.asarray(jl.Chen.sig(arr, m))


def logsig(path, m: int):
    arr = np.asarray(path)
    d = arr.shape[1]
    basis = jl.Chen.prepare(d, m)
    return np.asarray(jl.Chen.logsig(arr, basis))
