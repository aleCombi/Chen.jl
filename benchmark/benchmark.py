# benchmark.py
import os
import csv
import math
import time
import tracemalloc
import ast
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Try importing both libraries
try:
    import iisignature
    HAS_IISIG = True
except ImportError:
    HAS_IISIG = False
    print("Warning: iisignature not available", file=sys.stderr)

try:
    import pysiglib
    HAS_PYSIGLIB = True
except ImportError:
    HAS_PYSIGLIB = False
    print("Warning: pysiglib not available", file=sys.stderr)

# -------- simple YAML-ish config loader --------

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
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "benchmark_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = load_simple_yaml(config_path)

    Ns = raw.get("Ns", [150, 1000, 2000])
    Ds = raw.get("Ds", [2, 6, 7, 8])
    Ms = raw.get("Ms", [4, 6])
    path_kind = raw.get("path_kind", "linear")
    runs_dir = raw.get("runs_dir", "runs")
    repeats = int(raw.get("repeats", 5))
    logsig_method = raw.get("logsig_method", "O")
    operations = raw.get("operations", ["signature", "logsignature"])

    path_kind = path_kind.lower()
    if path_kind not in ("linear", "sin"):
        raise ValueError(f"Unknown path_kind '{path_kind}', expected 'linear' or 'sin'.")

    return {
        "Ns": Ns,
        "Ds": Ds,
        "Ms": Ms,
        "path_kind": path_kind,
        "runs_dir": runs_dir,
        "repeats": repeats,
        "logsig_method": logsig_method,
        "operations": operations,
    }

# -------- path generators --------

def make_path_linear(d: int, N: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, N)
    path = np.empty((N, d), dtype=float)
    path[:, 0] = ts
    if d > 1:
        path[:, 1:] = 2.0 * ts[:, None]
    return path

def make_path_sin(d: int, N: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, N)
    omega = 2.0 * math.pi
    ks = np.arange(1, d + 1, dtype=float)
    path = np.sin(omega * ts[:, None] * ks[None, :])
    return path

def make_path(d: int, N: int, kind: str) -> np.ndarray:
    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")

# -------- benchmarking helpers --------

def time_and_peak_memory(func, repeats: int = 5):
    best_time = float("inf")
    best_peak = 0

    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        dur = end - start
        if dur < best_time:
            best_time = dur
            best_peak = peak

    return best_time, best_peak

# -------- benchmark implementations --------

def bench_iisignature(d: int, m: int, N: int, path_kind: str, operation: str, method: str, repeats: int):
    """Benchmark using iisignature library"""
    if not HAS_IISIG:
        return None
        
    # Force clear iisignature cache to prevent 'prepare' conflicts
    if hasattr(iisignature, "_basis_cache"):
        iisignature._basis_cache.clear()
        
    path = make_path(d, N, path_kind)
    
    if operation == "signature":
        arg = m
        func = iisignature.sig
    elif operation == "logsignature":
        if d < 2:
            return None
        try:
            arg = iisignature.prepare(d, m, method)
            func = lambda p, basis: iisignature.logsig(p, basis, method)
        except Exception as e:
            print(f"Error preparing iisignature for d={d}, m={m}, method={method}: {e}", file=sys.stderr)
            return None
    else:
        raise ValueError(f"Unknown operation: {operation}")

    try:
        # warmup
        _ = func(path, arg)

        def run_op():
            func(path, arg)

        t_sec, peak_bytes = time_and_peak_memory(run_op, repeats=repeats)
    except Exception as e:
        print(f"iisignature benchmark failed for {operation} d={d} m={m}: {e}", file=sys.stderr)
        return None

    t_ms = t_sec * 1000.0
    alloc_kib = peak_bytes / 1024.0

    return {
        "N": N,
        "d": d,
        "m": m,
        "path_kind": path_kind,
        "operation": operation,
        "library": "iisignature",
        "t_ms": t_ms,
        "alloc_KiB": alloc_kib,
    }

def bench_pysiglib(d: int, m: int, N: int, path_kind: str, operation: str, repeats: int):
    """Benchmark using pysiglib library (signature only - no logsig support)"""
    if not HAS_PYSIGLIB:
        return None
    
    # pysiglib only supports signature
    if operation != "signature":
        return None
    
    path = make_path(d, N, path_kind)
    
    try:
        # pysiglib API: pysiglib.signature(path, degree)
        func = lambda: pysiglib.signature(path, degree=m)
    except Exception as e:
        print(f"Error setting up pysiglib sig for d={d}, m={m}: {e}", file=sys.stderr)
        return None

    try:
        # warmup
        _ = func()

        t_sec, peak_bytes = time_and_peak_memory(func, repeats=repeats)
    except Exception as e:
        print(f"pysiglib benchmark failed for {operation} d={d} m={m}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None

    t_ms = t_sec * 1000.0
    alloc_kib = peak_bytes / 1024.0

    return {
        "N": N,
        "d": d,
        "m": m,
        "path_kind": path_kind,
        "operation": operation,
        "library": "pysiglib",
        "t_ms": t_ms,
        "alloc_KiB": alloc_kib,
    }

# -------- sweep + write grid to file --------

def run_bench():
    cfg = load_config()
    Ns = cfg["Ns"]
    Ds = cfg["Ds"]
    Ms = cfg["Ms"]
    path_kind = cfg["path_kind"]
    runs_dir = cfg["runs_dir"]
    repeats = cfg["repeats"]
    logsig_method = cfg["logsig_method"]
    operations = cfg["operations"]

    print("Running Python benchmark suite with config:")
    print(f"  path_kind     = {path_kind}")
    print(f"  Ns            = {Ns}")
    print(f"  Ds            = {Ds}")
    print(f"  Ms            = {Ms}")
    print(f"  operations    = {operations}")
    print(f"  runs_dir      = \"{runs_dir}\"")
    print(f"  repeats       = {repeats}")
    print(f"  logsig_method = \"{logsig_method}\" (iisignature only)")
    print(f"  iisignature   = {'available' if HAS_IISIG else 'NOT AVAILABLE'}")
    print(f"  pysiglib      = {'available' if HAS_PYSIGLIB else 'NOT AVAILABLE'}")

    script_dir = Path(__file__).resolve().parent

    # If BENCHMARK_RUN_DIR is set, we write everything there.
    env_run_dir = os.environ.get("BENCHMARK_RUN_DIR")
    if env_run_dir:
        runs_path = Path(env_run_dir)
    else:
        runs_path = script_dir / runs_dir

    runs_path.mkdir(parents=True, exist_ok=True)

    results = []

    for N in Ns:
        for d in Ds:
            for m in Ms:
                for op in operations:
                    # Benchmark iisignature
                    if HAS_IISIG:
                        res = bench_iisignature(d, m, N, path_kind, op,
                                                logsig_method, repeats=repeats)
                        if res is not None:
                            results.append(res)
                    
                    # Benchmark pysiglib (only for signature)
                    if HAS_PYSIGLIB and op == "signature":
                        res = bench_pysiglib(d, m, N, path_kind, op, repeats=repeats)
                        if res is not None:
                            results.append(res)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = runs_path / f"run_python_{ts}.csv"

    fieldnames = ["N", "d", "m", "path_kind", "operation", "library", "t_ms", "alloc_KiB"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("=" * 60)
    print(f"Benchmark grid written to: {csv_path}")
    return csv_path

if __name__ == "__main__":
    run_bench()