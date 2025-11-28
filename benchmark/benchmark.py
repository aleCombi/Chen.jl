# benchmark.py
import os
import csv
import time
import tracemalloc
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Import shared utilities
from common import load_config, make_path, SCRIPT_DIR

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
        method_name = "sig"
    elif operation == "logsignature":
        if d < 2:
            return None
        try:
            arg = iisignature.prepare(d, m, method)
            func = lambda p, basis: iisignature.logsig(p, basis, method)
            method_name = "logsig"
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
        "language": "python",
        "library": "iisignature",
        "method": method_name,
        "path_type": "ndarray",
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
        method_name = "signature"
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
        "language": "python",
        "library": "pysiglib",
        "method": method_name,
        "path_type": "ndarray",
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
    repeats = cfg["repeats"]
    logsig_method = cfg["logsig_method"]
    operations = cfg["operations"]

    print("=" * 60)
    print("Python Benchmark Suite")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  path_kind     = {path_kind}")
    print(f"  Ns            = {Ns}")
    print(f"  Ds            = {Ds}")
    print(f"  Ms            = {Ms}")
    print(f"  operations    = {operations}")
    print(f"  repeats       = {repeats}")
    print(f"  logsig_method = {logsig_method} (iisignature only)")
    print(f"  iisignature   = {'available' if HAS_IISIG else 'NOT AVAILABLE'}")
    print(f"  pysiglib      = {'available' if HAS_PYSIGLIB else 'NOT AVAILABLE'}")
    print()

    # Check if orchestrator is overriding output location
    env_out_csv = os.environ.get("BENCHMARK_OUT_CSV", "")
    if env_out_csv:
        # Orchestrator mode: write to specified location
        csv_path = Path(env_out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Orchestrator mode: writing to {csv_path}")
        run_dir = None
    else:
        # Standalone mode: create own run folder
        from common import setup_run_folder, finalize_run_folder
        run_dir = setup_run_folder("benchmark_python", cfg)
        csv_path = run_dir / "results.csv"

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

    fieldnames = ["N", "d", "m", "path_kind", "operation", "language", "library", "method", "path_type", "t_ms", "alloc_KiB"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("=" * 60)
    print(f"Results written to: {csv_path}")
    
    # Only write summary in standalone mode
    if run_dir is not None:
        summary = {
            "total_benchmarks": len(results),
            "libraries": list(set(r["library"] for r in results)),
            "operations": list(set(r["operation"] for r in results)),
            "output_csv": str(csv_path.name),
        }
        
        finalize_run_folder(run_dir, summary)
    
    return csv_path

if __name__ == "__main__":
    run_bench()