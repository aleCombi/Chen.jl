# python/chen/_version.py
"""Version handling for chen-signatures"""
from pathlib import Path
import re

def get_version():
    """
    Get package version from the most appropriate source.
    
    Priority:
    1. Package metadata (when pip installed)
    2. Project.toml (when in development)
    3. Fallback hardcoded version (last resort)
    """
    
    # CASE 1: Try package metadata (when pip installed)
    try:
        from importlib.metadata import version
        return version("chen-signatures")
    except Exception:
        pass  # Not installed yet, continue
    
    # CASE 2: Try reading from Project.toml (development mode)
    try:
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]  # python/chen/_version.py -> repo root
        project_toml = repo_root / "Project.toml"
        
        if project_toml.exists():
            content = project_toml.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', 
                            content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass  # Project.toml not accessible
    
    # CASE 3: Fallback (should never happen in production)
    # This will be replaced at build time
    return "0.0.0+unknown"

__version__ = get_version()