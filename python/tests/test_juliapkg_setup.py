#!/usr/bin/env python3
"""
Test script to verify juliapkg setup works correctly.
Run this after installing the package to ensure everything is configured properly.
"""

import sys
from pathlib import Path
import pytest

def test_juliapkg_setup():
    """Test that juliapkg is properly configured"""
    print("="*70)
    print("JULIAPKG SETUP TEST")
    print("="*70)
    print()
    
    # Test 1: Check juliapkg is installed
    print("Test 1: Checking juliapkg installation...")
    try:
        import juliapkg
        print(f"  ✓ juliapkg installed")
    except ImportError as e:
        print("  ✗ juliapkg not installed - run: pip install juliapkg")
        pytest.fail(f"juliapkg not installed: {e}")
    print()
    
    # Test 2: Check juliapkg.json exists
    print("Test 2: Checking juliapkg.json...")
    try:
        # Get chen module path (import it here)
        import chen
        chen_path = Path(chen.__file__).parent
        juliapkg_json = chen_path / "juliapkg.json"
        
        assert juliapkg_json.exists(), f"juliapkg.json not found at {juliapkg_json}"
        print(f"  ✓ Found at {juliapkg_json}")
        import json
        with open(juliapkg_json) as f:
            config = json.load(f)
        print(f"  ✓ Configuration: {list(config.get('packages', {}).keys())}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to check juliapkg.json: {e}")
    print()
    
    # Test 3: Check Julia environment status
    print("Test 3: Checking Julia environment...")
    try:
        # Import juliacall which will trigger juliapkg setup
        from juliacall import Main as jl
        
        # Check if ChenSignatures can be loaded
        jl.seval("using ChenSignatures")
        print("  ✓ ChenSignatures loaded in Julia")
        
        # Get version info
        version = str(jl.seval("string(pkgversion(ChenSignatures))"))
        print(f"  ✓ ChenSignatures version: {version}")
        print("  ✓ Julia environment ready")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        pytest.fail(f"Failed to load Julia environment: {e}")
    print()
    
    # Test 4: Test chen functionality (chen already imported in Test 2)
    print("Test 4: Testing chen functionality...")
    try:
        import numpy as np
        
        # Simple signature computation
        path = np.random.randn(10, 3)
        sig = chen.sig(path, 2)
        
        print(f"  ✓ Successfully computed signature")
        print(f"    Input shape: {path.shape}")
        print(f"    Output shape: {sig.shape}")
        print(f"    Expected output size: {3 + 9} = 12")

        assert len(sig) == 12, f"Output size incorrect: {len(sig)} != 12"
        print("  ✓ Output size correct")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to test chen functionality: {e}")
    print()
    
    # Test 5: Check development mode detection
    print("Test 5: Checking development mode...")
    # chen_path already defined from Test 2: .../python/chen/
    # Structure: repo_root/python/chen/__init__.py
    # chen_path.parents[1] = repo_root
    repo_root = chen_path.parents[1]
    
    if (repo_root / "Project.toml").exists():
        print(f"  ✓ Development mode detected")
        print(f"    Repo root: {repo_root}")
        print(f"    Using local ChenSignatures.jl")
    else:
        print(f"  ✓ Installed mode (using GitHub/registry)")
        print(f"    Checked path: {repo_root}")
    print()
    
    print("="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)

if __name__ == "__main__":
    # When run as a script, call pytest on this file
    sys.exit(pytest.main([__file__, "-v"]))