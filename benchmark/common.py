# common.py
"""Shared utilities for benchmark suite"""

import ast
import math
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np

# -------- Constants --------

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.yaml"

# -------- Config Loading --------

def load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Simple YAML parser for benchmark config files"""
    cfg = {}
    if not path.is_file():
        return cfg
    
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
                try:
                    cfg[key] = ast.literal_eval(value)
                except:
                    cfg[key] = value
            else:
                try:
                    cfg[key] = int(value)
                except ValueError:
                    cfg[key] = value
    return cfg

def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and parse benchmark configuration"""
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

# -------- Path Generators --------

def make_path_linear(d: int, N: int) -> np.ndarray:
    """Generate linear path: [t, 2t, 2t, ...]"""
    ts = np.linspace(0.0, 1.0, N)
    path = np.empty((N, d), dtype=float)
    path[:, 0] = ts
    if d > 1:
        path[:, 1:] = 2.0 * ts[:, None]
    return path

def make_path_sin(d: int, N: int) -> np.ndarray:
    """Generate sinusoidal path: [sin(2π·1·t), sin(2π·2·t), ...]"""
    ts = np.linspace(0.0, 1.0, N)
    omega = 2.0 * math.pi
    # Matches Julia: path[i, k] = sin(2pi * t * k)
    # Python array is 0-indexed, so k=1..d maps to cols 0..d-1
    ks = np.arange(1, d + 1, dtype=float)
    path = np.sin(omega * ts[:, None] * ks[None, :])
    return path

def make_path(d: int, N: int, kind: str) -> np.ndarray:
    """Generate path of specified kind"""
    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")

# -------- Run Folder Management --------

def setup_run_folder(run_type: str, cfg: Dict[str, Any]) -> Path:
    """
    Create a timestamped run folder with standardized structure.
    
    Args:
        run_type: One of 'benchmark_julia', 'benchmark_python', 
                  'benchmark_comparison', 'signature_check'
        cfg: Configuration dictionary
    
    Returns:
        Path to the created run folder
    """
    runs_root = SCRIPT_DIR / cfg.get("runs_dir", "runs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{run_type}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config snapshot
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, run_dir / "benchmark_config.yaml")
    
    # Write resolved config
    config_path = run_dir / "config_resolved.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    
    # Create metadata
    metadata = {
        "run_type": run_type,
        "timestamp": ts,
        "start_time": datetime.now().isoformat(),
    }
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created run folder: {run_dir}")
    return run_dir

def finalize_run_folder(run_dir: Path, summary: Dict[str, Any]):
    """
    Write summary and completion metadata to run folder.
    
    Args:
        run_dir: Path to run folder
        summary: Dictionary with summary information
    """
    # Update metadata with end time
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata["end_time"] = datetime.now().isoformat()
    metadata["summary"] = summary
    
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Write human-readable summary
    summary_path = run_dir / "SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Benchmark Run Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Run Type: {metadata.get('run_type', 'unknown')}\n")
        f.write(f"Start:    {metadata.get('start_time', 'unknown')}\n")
        f.write(f"End:      {metadata.get('end_time', 'unknown')}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"{'-' * 60}\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Run completed. Summary written to: {summary_path}")