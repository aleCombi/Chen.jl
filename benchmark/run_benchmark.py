# run_benchmarks.py
import csv
import re
import subprocess
import ast
import os
from pathlib import Path
from datetime import datetime
import shutil

import matplotlib.pyplot as plt

# This is the folder where THIS file lives: .../Chen/benchmark
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.yaml"
PYPROJECT = SCRIPT_DIR / "pyproject.toml"

# -------- tiny YAML-ish loader --------

def load_simple_yaml(path: Path) -> dict:
    cfg = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue

            if value.startswith('"') and value.endswith('"'):
                cfg[key] = value[1:-1]
            elif value.startswith("["):
                cfg[key] = ast.literal_eval(value)
            else:
                try:
                    cfg[key] = int(value)
                except ValueError:
                    cfg[key] = value
    return cfg

def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    raw = load_simple_yaml(CONFIG_PATH)
    runs_dir = raw.get("runs_dir", "runs")
    return raw, SCRIPT_DIR / runs_dir

# -------- uv project bootstrap --------

def ensure_uv_project():
    if not PYPROJECT.exists():
        print("No pyproject.toml found, initializing uv project in", SCRIPT_DIR)
        subprocess.run(
            ["uv", "init", "."],
            cwd=SCRIPT_DIR,
            check=True,
        )

    print("Ensuring Python deps via uv add (iisignature, numpy, pysiglib, matplotlib)...")
    subprocess.run(
        ["uv", "add", "iisignature", "numpy", "pysiglib", "matplotlib"],
        cwd=SCRIPT_DIR,
        check=True,
    )

# -------- run Julia benchmark --------
def run_julia_benchmark(run_dir: Path, base_env: dict) -> Path:
    print("=== Running Julia benchmark in local project ===")

    env = base_env.copy()
    env["JULIA_PROJECT"] = str(SCRIPT_DIR)
    env["BENCHMARK_OUT_CSV"] = str(run_dir / "run_julia.csv")

    julia_script = str(SCRIPT_DIR / "benchmark.jl")

    result = subprocess.run(
        ["julia", julia_script],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        env=env,
    )

    # Save logs
    (run_dir / "julia_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "julia_stderr.log").write_text(result.stderr, encoding="utf-8")

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Julia benchmark failed with code {result.returncode}")

    m = re.search(r"Benchmark grid written to:\s*(.*)", result.stdout)
    if not m:
        raise RuntimeError("Could not find output path in Julia benchmark output.")
    path_str = m.group(1).strip()
    julia_csv = Path(path_str)
    if not julia_csv.is_absolute():
        julia_csv = (SCRIPT_DIR / julia_csv).resolve()
    print(f"Julia CSV: {julia_csv}")
    return julia_csv


# -------- run Python benchmark --------
def run_python_benchmark(run_dir: Path, base_env: dict) -> Path:
    print("=== Running Python benchmark suite via uv ===")

    env = base_env.copy()
    env["BENCHMARK_RUN_DIR"] = str(run_dir)

    result = subprocess.run(
        ["uv", "run", "benchmark.py"],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        env=env,
    )

    # Save logs
    (run_dir / "python_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "python_stderr.log").write_text(result.stderr, encoding="utf-8")

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Python benchmark failed with code {result.returncode}")

    m = re.search(r"Benchmark grid written to:\s*(.*)", result.stdout)
    if not m:
        raise RuntimeError("Could not find output path in Python benchmark output.")
    path_str = m.group(1).strip()
    py_csv = Path(path_str)
    if not py_csv.is_absolute():
        py_csv = (SCRIPT_DIR / py_csv).resolve()
    print(f"Python CSV: {py_csv}")
    return py_csv

# -------- comparison logic --------

def read_julia_csv(path: Path):
    """Read Julia CSV (no library column)"""
    rows_by_key = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            d = int(row["d"])
            m = int(row["m"])
            path_kind = row.get("path_kind", row.get("kind", "")).strip()
            operation = row.get("operation", "signature").strip()
            
            key = (N, d, m, path_kind, operation)
            rows_by_key[key] = row
    return rows_by_key

def read_python_csv(path: Path):
    """Read Python CSV (with library column)"""
    rows_by_key = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            d = int(row["d"])
            m = int(row["m"])
            path_kind = row.get("path_kind", row.get("kind", "")).strip()
            operation = row.get("operation", "signature").strip()
            library = row.get("library", "python").strip()
            
            key = (N, d, m, path_kind, operation, library)
            rows_by_key[key] = row
    return rows_by_key

def compare_runs(julia_csv: Path, python_csv: Path, runs_dir: Path) -> Path:
    julia_rows = read_julia_csv(julia_csv)
    python_rows = read_python_csv(python_csv)

    # Extract unique libraries from Python results
    libraries = set()
    for key in python_rows.keys():
        libraries.add(key[5])  # library is the 6th element
    
    print(f"Found Python libraries: {', '.join(sorted(libraries))}")
    
    # Find common benchmark configurations (without library)
    julia_keys = set((k[0], k[1], k[2], k[3], k[4]) for k in julia_rows.keys())
    python_keys = set((k[0], k[1], k[2], k[3], k[4]) for k in python_rows.keys())
    common_keys = sorted(julia_keys & python_keys)
    
    if not common_keys:
        print("Julia keys sample:", list(julia_keys)[:5])
        print("Python keys sample:", list(python_keys)[:5])
        raise RuntimeError("No overlapping benchmark keys between Julia and Python CSVs.")

    out_path = runs_dir / "comparison.csv"

    fieldnames = [
        "N",
        "d",
        "m",
        "path_kind",
        "operation",
        "python_library",
        "t_ms_julia",
        "t_ms_python",
        "speed_ratio_python_over_julia",
        "alloc_KiB_julia",
        "alloc_KiB_python",
    ]

    runs_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (N, d, m, path_kind, operation) in common_keys:
            jr = julia_rows[(N, d, m, path_kind, operation)]
            t_jl = float(jr["t_ms"])
            alloc_jl = float(jr["alloc_KiB"])

            # Compare against each available Python library
            for lib in libraries:
                py_key = (N, d, m, path_kind, operation, lib)
                if py_key not in python_rows:
                    continue
                    
                pr = python_rows[py_key]
                t_py = float(pr["t_ms"])
                alloc_py = float(pr["alloc_KiB"])

                speed_ratio = t_py / t_jl if t_jl > 0 else float("nan")

                writer.writerow(
                    {
                        "N": N,
                        "d": d,
                        "m": m,
                        "path_kind": path_kind,
                        "operation": operation,
                        "python_library": lib,
                        "t_ms_julia": t_jl,
                        "t_ms_python": t_py,
                        "speed_ratio_python_over_julia": speed_ratio,
                        "alloc_KiB_julia": alloc_jl,
                        "alloc_KiB_python": alloc_py,
                    }
                )

    print("===" * 20)
    print(f"Comparison CSV written to: {out_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        by_library = {}
        for row in reader:
            lib = row["python_library"]
            ratio = float(row["speed_ratio_python_over_julia"])
            if lib not in by_library:
                by_library[lib] = []
            by_library[lib].append(ratio)
    
    for lib in sorted(by_library.keys()):
        ratios = by_library[lib]
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        print(f"{lib:>15s}: avg speedup = {avg_ratio:.2f}x, range = [{min_ratio:.2f}x, {max_ratio:.2f}x]")
    
    return out_path

# -------- plotting logic --------

def load_comparison_rows(comparison_csv: Path):
    rows = []
    with comparison_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "N": int(row["N"]),
                    "d": int(row["d"]),
                    "m": int(row["m"]),
                    "path_kind": row["path_kind"].strip(),
                    "operation": row["operation"].strip(),
                    "python_library": row["python_library"].strip(),
                    "t_ms_julia": float(row["t_ms_julia"]),
                    "t_ms_python": float(row["t_ms_python"]),
                }
            )
    return rows

def get_julia_time(rows, N, d, m, path_kind, operation):
    for r in rows:
        if (
            r["N"] == N
            and r["d"] == d
            and r["m"] == m
            and r["path_kind"] == path_kind
            and r["operation"] == operation
        ):
            return r["t_ms_julia"]
    return None

def get_python_time(rows, lib, N, d, m, path_kind, operation):
    for r in rows:
        if (
            r["N"] == N
            and r["d"] == d
            and r["m"] == m
            and r["path_kind"] == path_kind
            and r["operation"] == operation
            and r["python_library"] == lib
        ):
            return r["t_ms_python"]
    return None

def make_plots(comparison_csv: Path, runs_dir: Path, cfg: dict):
    print("=== Making comparison plots ===")
    rows = load_comparison_rows(comparison_csv)

    if not rows:
        print("No rows found in comparison CSV; skipping plots.")
        return

    # config-derived grids
    Ns = sorted(cfg.get("Ns", []))
    Ds = sorted(cfg.get("Ds", []))
    Ms = sorted(cfg.get("Ms", []))

    if not Ns:
        Ns = sorted({r["N"] for r in rows})
    if not Ds:
        Ds = sorted({r["d"] for r in rows})
    if not Ms:
        Ms = sorted({r["m"] for r in rows})

    # pick "max" for fixed params (worst-case scaling)
    N_fixed_for_d = max(Ns)
    N_fixed_for_m = max(Ns)

    d_fixed_for_N = max(Ds)
    d_fixed_for_m = max(Ds)

    m_fixed_for_N = max(Ms)
    m_fixed_for_d = max(Ms)

    path_kind = cfg.get("path_kind", "sin")
    operations = cfg.get("operations", ["signature", "logsignature"])

    libs_python = sorted({r["python_library"] for r in rows})
    # We'll use these labels in the legend
    all_lib_labels = ["ChenSignatures.jl"] + libs_python

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey="col")

    op_order = ["signature", "logsignature"]
    for row_idx, vary in enumerate(["N", "d", "m"]):
        for col_idx, op in enumerate(op_order):
            ax = axes[row_idx, col_idx]

            if op not in operations:
                ax.set_visible(False)
                continue

            # Determine x-grid and fixed params for this subplot
            if vary == "N":
                xs = Ns
                d_fix = d_fixed_for_N
                m_fix = m_fixed_for_N
                xlabel = "N (number of paths)"
            elif vary == "d":
                xs = Ds
                d_fix = None
                m_fix = m_fixed_for_d
                xlabel = "d (dimension)"
            else:  # vary m
                xs = Ms
                d_fix = d_fixed_for_m
                m_fix = None
                xlabel = "m (signature level)"

            # Build series for each library
            for lib_label in all_lib_labels:
                ys = []
                xs_effective = []

                for x in xs:
                    if vary == "N":
                        N = x
                        d = d_fix
                        m = m_fix
                    elif vary == "d":
                        N = N_fixed_for_d
                        d = x
                        m = m_fix
                    else:  # vary m
                        N = N_fixed_for_m
                        d = d_fix
                        m = x

                    # ChenSignatures.jl times come from t_ms_julia once per (N,d,m,op)
                    if lib_label == "ChenSignatures.jl":
                        t = get_julia_time(rows, N, d, m, path_kind, op)
                    else:
                        t = get_python_time(rows, lib_label, N, d, m, path_kind, op)

                    if t is not None and t > 0.0:
                        xs_effective.append(x)
                        ys.append(t)

                if len(xs_effective) >= 2:
                    ax.plot(xs_effective, ys, marker="o", label=lib_label)

            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("time (ms)")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            title = f"{op}, vary {vary}"
            if vary == "N":
                title += f" (d={d_fixed_for_N}, m={m_fixed_for_N})"
            elif vary == "d":
                title += f" (N={N_fixed_for_d}, m={m_fixed_for_d})"
            else:
                title += f" (N={N_fixed_for_m}, d={d_fixed_for_m})"
            ax.set_title(title)

            if row_idx == 0:
                ax.legend()

    fig.tight_layout()
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_plot = runs_dir / "comparison_3x2.png"
    fig.savefig(out_plot, dpi=300)
    print(f"Plots written to: {out_plot}")

# -------- main --------

def main():
    cfg, runs_root = load_config()
    print(f"Using runs root from config: {runs_root}")

    # One directory per run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Current run directory: {run_dir}")

    # Dump config snapshot
    cfg_copy_path = run_dir / "benchmark_config.yaml"
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, cfg_copy_path)
    else:
        cfg_copy_path.write_text("# benchmark_config.yaml not found at run time\n", encoding="utf-8")

    # Dump "effective" config (after defaults) for debugging
    cfg_effective_path = run_dir / "config_effective.txt"
    with cfg_effective_path.open("w", encoding="utf-8") as f:
        f.write("Effective benchmark config (Python view)\n")
        f.write("=======================================\n")
        for k, v in sorted(cfg.items()):
            f.write(f"{k}: {v!r}\n")

    base_env = os.environ.copy()

    ensure_uv_project()
    julia_csv = run_julia_benchmark(run_dir, base_env)
    python_csv = run_python_benchmark(run_dir, base_env)
    comparison_csv = compare_runs(julia_csv, python_csv, run_dir)
    make_plots(comparison_csv, run_dir, cfg)

if __name__ == "__main__":
    main()
