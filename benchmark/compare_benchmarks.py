# compare_benchmarks.py
# Orchestrator script that runs both Julia and Python benchmarks,
# then generates comparison CSV and performance plots.

import csv
import re
import subprocess
import os
from pathlib import Path
from datetime import datetime
import shutil
import sys

import matplotlib.pyplot as plt

# Import shared utilities
from common import load_config, SCRIPT_DIR, CONFIG_PATH

PYPROJECT = SCRIPT_DIR / "pyproject.toml"

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
    print("\n" + "=" * 60)
    print("Running Julia Benchmark")
    print("=" * 60)

    env = base_env.copy()
    env["JULIA_PROJECT"] = str(SCRIPT_DIR)
    env["BENCHMARK_OUT_CSV"] = str(run_dir / "julia_results.csv")

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
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Julia benchmark failed with code {result.returncode}")

    julia_csv = run_dir / "julia_results.csv"
    if not julia_csv.exists():
        raise RuntimeError(f"Expected Julia output not found: {julia_csv}")
    
    print(f"✓ Julia results: {julia_csv.name}")
    return julia_csv


# -------- run Python benchmark --------
def run_python_benchmark(run_dir: Path, base_env: dict) -> Path:
    print("\n" + "=" * 60)
    print("Running Python Benchmark")
    print("=" * 60)

    env = base_env.copy()
    env["BENCHMARK_OUT_CSV"] = str(run_dir / "python_results.csv")

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
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Python benchmark failed with code {result.returncode}")

    python_csv = run_dir / "python_results.csv"
    if not python_csv.exists():
        raise RuntimeError(f"Expected Python output not found: {python_csv}")
    
    print(f"✓ Python results: {python_csv.name}")
    return python_csv

# -------- comparison logic --------

def read_csv_unified(path: Path):
    """Read CSV with unified schema: N, d, m, path_kind, operation, language, library, method, path_type, t_ms, alloc_KiB"""
    rows_by_key = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            d = int(row["d"])
            m = int(row["m"])
            path_kind = row["path_kind"].strip()
            operation = row["operation"].strip()
            language = row["language"].strip()
            library = row["library"].strip()
            method = row.get("method", "").strip()
            path_type = row.get("path_type", "").strip()
            
            key = (N, d, m, path_kind, operation, language, library, method, path_type)
            rows_by_key[key] = row
    return rows_by_key

def compare_runs(julia_csv: Path, python_csv: Path, runs_dir: Path) -> Path:
    julia_rows = read_csv_unified(julia_csv)
    python_rows = read_csv_unified(python_csv)

    # Extract Julia and Python libraries
    julia_libs = set((k[5], k[6]) for k in julia_rows.keys())  # (language, library)
    python_libs = set((k[5], k[6]) for k in python_rows.keys())
    
    print(f"Found Julia: {', '.join(lib for _, lib in julia_libs)}")
    print(f"Found Python: {', '.join(lib for _, lib in python_libs)}")
    
    # For comparison, we need matching (N, d, m, path_kind, operation)
    # Then compare across language/library/method/path_type
    julia_configs = set((k[0], k[1], k[2], k[3], k[4]) for k in julia_rows.keys())
    python_configs = set((k[0], k[1], k[2], k[3], k[4]) for k in python_rows.keys())
    common_configs = sorted(julia_configs & python_configs)
    
    if not common_configs:
        print("Julia configs sample:", list(julia_configs)[:5])
        print("Python configs sample:", list(python_configs)[:5])
        raise RuntimeError("No overlapping benchmark configs between Julia and Python CSVs.")

    out_path = runs_dir / "comparison.csv"

    fieldnames = [
        "N", "d", "m", "path_kind", "operation",
        "julia_library", "julia_method", "julia_path_type",
        "python_library", "python_method", "python_path_type",
        "t_ms_julia", "t_ms_python", "speed_ratio_python_over_julia",
        "alloc_KiB_julia", "alloc_KiB_python",
    ]

    runs_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (N, d, m, path_kind, operation) in common_configs:
            # Get all Julia variants for this config
            julia_variants = [(k, v) for k, v in julia_rows.items() 
                            if k[0:5] == (N, d, m, path_kind, operation) and k[5] == "julia"]
            
            # Get all Python variants for this config
            python_variants = [(k, v) for k, v in python_rows.items() 
                             if k[0:5] == (N, d, m, path_kind, operation) and k[5] == "python"]
            
            # Compare each Julia variant with each Python variant
            for (jk, jr) in julia_variants:
                t_jl = float(jr["t_ms"])
                alloc_jl = float(jr["alloc_KiB"])
                
                for (pk, pr) in python_variants:
                    t_py = float(pr["t_ms"])
                    alloc_py = float(pr["alloc_KiB"])
                    
                    speed_ratio = t_py / t_jl if t_jl > 0 else float("nan")
                    
                    writer.writerow({
                        "N": N,
                        "d": d,
                        "m": m,
                        "path_kind": path_kind,
                        "operation": operation,
                        "julia_library": jk[6],
                        "julia_method": jk[7],
                        "julia_path_type": jk[8],
                        "python_library": pk[6],
                        "python_method": pk[7],
                        "python_path_type": pk[8],
                        "t_ms_julia": t_jl,
                        "t_ms_python": t_py,
                        "speed_ratio_python_over_julia": speed_ratio,
                        "alloc_KiB_julia": alloc_jl,
                        "alloc_KiB_python": alloc_py,
                    })

    print("===" * 20)
    print(f"Comparison CSV written to: {out_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        by_comparison = {}
        for row in reader:
            key = f"{row['julia_library']}({row['julia_path_type']}) vs {row['python_library']}"
            ratio = float(row["speed_ratio_python_over_julia"])
            if key not in by_comparison:
                by_comparison[key] = []
            by_comparison[key].append(ratio)
    
    for comp in sorted(by_comparison.keys()):
        ratios = by_comparison[comp]
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        print(f"{comp:>50s}: avg speedup = {avg_ratio:.2f}x, range = [{min_ratio:.2f}x, {max_ratio:.2f}x]")
    
    return out_path

# -------- plotting logic --------

def load_comparison_rows(comparison_csv: Path):
    rows = []
    with comparison_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "N": int(row["N"]),
                "d": int(row["d"]),
                "m": int(row["m"]),
                "path_kind": row["path_kind"].strip(),
                "operation": row["operation"].strip(),
                "julia_library": row["julia_library"].strip(),
                "julia_path_type": row["julia_path_type"].strip(),
                "python_library": row["python_library"].strip(),
                "t_ms_julia": float(row["t_ms_julia"]),
                "t_ms_python": float(row["t_ms_python"]),
            })
    return rows

def get_julia_time(rows, N, d, m, path_kind, operation, julia_path_type):
    for r in rows:
        if (r["N"] == N and r["d"] == d and r["m"] == m 
            and r["path_kind"] == path_kind and r["operation"] == operation
            and r["julia_path_type"] == julia_path_type):
            return r["t_ms_julia"]
    return None

def get_python_time(rows, lib, N, d, m, path_kind, operation):
    for r in rows:
        if (r["N"] == N and r["d"] == d and r["m"] == m 
            and r["path_kind"] == path_kind and r["operation"] == operation
            and r["python_library"] == lib):
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

    # Get unique Julia path types and Python libraries
    julia_path_types = sorted({r["julia_path_type"] for r in rows})
    python_libs = sorted({r["python_library"] for r in rows})
    
    # Build labels for legend
    all_labels = []
    for pt in julia_path_types:
        all_labels.append(f"ChenSignatures.jl({pt})")
    all_labels.extend(python_libs)

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
                xlabel = "N (number of points)"
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

            # Plot each Julia path type
            for pt in julia_path_types:
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

                    t = get_julia_time(rows, N, d, m, path_kind, op, pt)
                    if t is not None and t > 0.0:
                        xs_effective.append(x)
                        ys.append(t)

                if len(xs_effective) >= 2:
                    ax.plot(xs_effective, ys, marker="o", label=f"ChenSignatures.jl({pt})")

            # Plot each Python library
            for lib in python_libs:
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

                    t = get_python_time(rows, lib, N, d, m, path_kind, op)
                    if t is not None and t > 0.0:
                        xs_effective.append(x)
                        ys.append(t)

                if len(xs_effective) >= 2:
                    ax.plot(xs_effective, ys, marker="o", label=lib)

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
    from common import setup_run_folder, finalize_run_folder
    
    cfg = load_config(CONFIG_PATH)
    
    print("=" * 60)
    print("Benchmark Comparison (Orchestrated)")
    print("=" * 60)
    print(f"Configuration:")
    for k, v in sorted(cfg.items()):
        print(f"  {k:15s} = {v}")
    print()

    # Setup comparison run folder
    run_dir = setup_run_folder("benchmark_comparison", cfg)

    base_env = os.environ.copy()

    ensure_uv_project()
    julia_csv = run_julia_benchmark(run_dir, base_env)
    python_csv = run_python_benchmark(run_dir, base_env)
    comparison_csv = compare_runs(julia_csv, python_csv, run_dir)
    make_plots(comparison_csv, run_dir, cfg)
    
    # Prepare summary
    summary = {
        "julia_csv": julia_csv.name,
        "python_csv": python_csv.name,
        "comparison_csv": comparison_csv.name,
        "plots": "comparison_3x2.png",
    }
    
    # Add performance summary from comparison CSV
    with comparison_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        by_comparison = {}
        for row in reader:
            key = f"{row['julia_library']}({row['julia_path_type']}) vs {row['python_library']}"
            ratio = float(row["speed_ratio_python_over_julia"])
            if key not in by_comparison:
                by_comparison[key] = []
            by_comparison[key].append(ratio)
    
    summary["performance"] = {}
    for comp in sorted(by_comparison.keys()):
        ratios = by_comparison[comp]
        avg_ratio = sum(ratios) / len(ratios)
        summary["performance"][comp] = f"{avg_ratio:.2f}x"
    
    finalize_run_folder(run_dir, summary)
    print(f"\n✓ Orchestrated benchmark complete: {run_dir}")


if __name__ == "__main__":
    main()