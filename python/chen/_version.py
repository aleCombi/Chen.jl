# python/chen/_version.py
"""Version handling for chen-signatures"""
from pathlib import Path
import os
import re
import warnings

def get_version():
    """
    Get package version from the most appropriate source.

    Priority:
    1. Environment variable CHEN_VERSION (when building for release)
    2. Package metadata (when pip installed)
    3. Project.toml (when in development)
    4. Fallback hardcoded version (last resort)
    """

    # CASE 1: Environment variable (used during build)
    env_version = os.environ.get("CHEN_VERSION")
    if env_version:
        return env_version

    # CASE 2: Try package metadata (when pip installed)
    try:
        from importlib.metadata import version
        return version("chen-signatures")
    except Exception:
        pass  # Not installed yet, continue
    
    # CASE 3: Try reading from Project.toml (development mode)
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
    
    # CASE 4: Fallback (should never happen in production)
    warnings.warn(
        "Failed to determine chen-signatures version. "
        "Using fallback '0.0.0+unknown'. "
        "This may cause issues with PyPI uploads.",
        RuntimeWarning,
        stacklevel=2
    )
    return "0.0.0+unknown"

__version__ = get_version()