# check_signatures.py

import ast
import math
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import iisignature

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.yaml"

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
    Ns = raw.get("Ns", [5, 10, 20])
    Ds = raw.get("Ds", [1, 2, 7])
    Ms = raw.get("Ms", [3, 5, 7])
    path_kind = raw.get("path_kind", "linear").lower()
    runs_dir = raw.get("runs_dir", "runs")
    logsig_method = raw.get("logsig_method", "O")
    return Ns, Ds, Ms, path_kind, SCRIPT_DIR / runs_dir, logsig_method

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

# -------- call Julia helper --------

def julia_signature(N: int, d: int, m: int, path_kind: str, operation: str) -> np.ndarray:
    cmd = [
        "julia",
        "--project=.",
        str(SCRIPT_DIR / "sigcheck.jl"),
        str(N),
        str(d),
        str(m),
        path_kind,
        operation,
    ]

    result = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("Julia stdout:\n", result.stdout, file=sys.stderr)
        print("Julia stderr:\n", result.stderr, file=sys.stderr)
        raise RuntimeError(f"Julia sigcheck failed with code {result.returncode}")

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        print("Julia stderr:\n", result.stderr, file=sys.stderr)
        raise RuntimeError("Julia sigcheck produced no numeric output")
    line = lines[-1]
    values = np.fromstring(line, sep=" ", dtype=float)

    if values.size == 0:
        print("Julia stdout lines:", lines, file=sys.stderr)
        raise RuntimeError(f"Could not parse any floats from Julia output line: {line!r}")

    return values

# -------- main comparison loop --------

def compare_signatures():
    Ns, Ds, Ms, path_kind, runs_dir, logsig_method = load_config()
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = runs_dir / f"signature_compare_{ts}.csv"

    fieldnames = [
        "N",
        "d",
        "m",
        "path_kind",
        "operation",
        "len_sig",
        "max_abs_diff",
        "l2_diff",
        "rel_l2_diff",
    ]

    operations = ["signature", "logsignature"]

    print(f"Comparing signatures using logsig_method='{logsig_method}' for Python...")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in Ns:
            for d in Ds:
                for m in Ms:
                    path = make_path(d, N, path_kind)

                    for op in operations:
                        if op == "logsignature" and d < 2:
                            continue

                        # Force clear iisignature cache
                        if hasattr(iisignature, "_basis_cache"):
                            iisignature._basis_cache.clear()

                        print(f"Comparing {op} for N={N}, d={d}, m={m}, kind={path_kind}...")

                        # 1. Python calculation
                        if op == "signature":
                            sig_py = iisignature.sig(path, m)
                        else:
                            basis = iisignature.prepare(d, m, logsig_method)
                            sig_py = iisignature.logsig(path, basis, logsig_method)
                        
                        sig_py = np.asarray(sig_py, dtype=float).ravel()

                        # 2. Julia calculation
                        sig_jl = julia_signature(N, d, m, path_kind, op)

                        # 3. Validation
                        if sig_jl.shape != sig_py.shape:
                            print("=== DEBUG SHAPE MISMATCH ===")
                            print(f"N={N}, d={d}, m={m}, kind={path_kind}, op={op}")
                            print("Julia sig shape:", sig_jl.shape)
                            print("Python sig shape:", sig_py.shape)
                            print("============================")
                            raise ValueError(
                                f"Shape mismatch for N={N}, d={d}, m={m}, op={op}: "
                                f"Julia {sig_jl.shape}, Python {sig_py.shape}"
                            )

                        diff = sig_jl - sig_py
                        max_abs = float(np.max(np.abs(diff)))
                        l2 = float(np.linalg.norm(diff))
                        norm_ref = float(np.linalg.norm(sig_jl))
                        rel_l2 = l2 / norm_ref if norm_ref > 0 else float("nan")

                        print(f"  len={len(sig_jl)}, max_abs={max_abs:.3e}, rel_l2={rel_l2:.3e}")

                        writer.writerow(
                            {
                                "N": N,
                                "d": d,
                                "m": m,
                                "path_kind": path_kind,
                                "operation": op,
                                "len_sig": len(sig_jl),
                                "max_abs_diff": max_abs,
                                "l2_diff": l2,
                                "rel_l2_diff": rel_l2,
                            }
                        )

    print(f"\nSignature comparison CSV written to: {out_csv}")
    return out_csv

if __name__ == "__main__":
    compare_signatures()