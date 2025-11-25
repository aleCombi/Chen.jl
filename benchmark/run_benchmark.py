# run_benchmarks.py

import csv
import re
import subprocess
import ast
import os
from pathlib import Path
from datetime import datetime

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

    print("Ensuring Python deps via uv add (iisignature, numpy)...")
    subprocess.run(
        ["uv", "add", "iisignature", "numpy"],
        cwd=SCRIPT_DIR,
        check=True,
    )

# -------- run Julia benchmark --------

def run_julia_benchmark() -> Path:
    print("=== Running Julia benchmark in local project ===")

    env = os.environ.copy()
    env["JULIA_PROJECT"] = str(SCRIPT_DIR)

    julia_script = str(SCRIPT_DIR / "benchmark.jl")

    result = subprocess.run(
        ["julia", julia_script],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        env=env,
    )

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

def run_python_benchmark() -> Path:
    print("=== Running Python/iisignature benchmark via uv ===")
    result = subprocess.run(
        ["uv", "run", "benchmark.py"],
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
    )

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

def read_csv_by_key(path: Path):
    rows_by_key = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            d = int(row["d"])
            m = int(row["m"])
            path_kind = row.get("path_kind", row.get("kind", "")).strip()
            # New key: operation
            operation = row.get("operation", "signature").strip()
            
            key = (N, d, m, path_kind, operation)
            rows_by_key[key] = row
    return rows_by_key

def compare_runs(julia_csv: Path, python_csv: Path, runs_dir: Path) -> Path:
    julia_rows = read_csv_by_key(julia_csv)
    python_rows = read_csv_by_key(python_csv)

    common_keys = sorted(set(julia_rows.keys()) & set(python_rows.keys()))
    if not common_keys:
        print("Julia keys sample:", list(julia_rows.keys())[:5])
        print("Python keys sample:", list(python_rows.keys())[:5])
        raise RuntimeError("No overlapping benchmark keys between Julia and Python CSVs.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = runs_dir / f"comparison_{ts}.csv"

    fieldnames = [
        "N",
        "d",
        "m",
        "path_kind",
        "operation",
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
            pr = python_rows[(N, d, m, path_kind, operation)]

            t_jl = float(jr["t_ms"])
            t_py = float(pr["t_ms"])
            alloc_jl = float(jr["alloc_KiB"])
            alloc_py = float(pr["alloc_KiB"])

            speed_ratio = t_py / t_jl if t_jl > 0 else float("nan")

            writer.writerow(
                {
                    "N": N,
                    "d": d,
                    "m": m,
                    "path_kind": path_kind,
                    "operation": operation,
                    "t_ms_julia": t_jl,
                    "t_ms_python": t_py,
                    "speed_ratio_python_over_julia": speed_ratio,
                    "alloc_KiB_julia": alloc_jl,
                    "alloc_KiB_python": alloc_py,
                }
            )

    print("===" * 20)
    print(f"Comparison CSV written to: {out_path}")
    return out_path

def main():
    cfg, runs_dir = load_config()
    print(f"Using runs_dir from config: {runs_dir}")

    ensure_uv_project()
    julia_csv = run_julia_benchmark()
    python_csv = run_python_benchmark()
    compare_runs(julia_csv, python_csv, runs_dir)

if __name__ == "__main__":
    main()