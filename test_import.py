#!/usr/bin/env python3
"""Diagnostic script to test fkl_ffi installation."""

import sys
import os
from pathlib import Path

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

print("\n" + "="*60)
print("Testing fkl_ffi import...")
print("="*60)

try:
    import fkl_ffi
    print(f"✓ fkl_ffi imported successfully")
    print(f"  Location: {fkl_ffi.__file__}")
    print(f"  Package directory: {Path(fkl_ffi.__file__).parent}")
    
    # Check for library file
    lib_path = Path(fkl_ffi.__file__).parent / "_fkl_ffi.so"
    print(f"\nLibrary file check:")
    print(f"  Expected: {lib_path}")
    print(f"  Exists: {lib_path.exists()}")
    
    if not lib_path.exists():
        # Try alternative names
        alt_names = ["_fkl_ffi.so", "lib_fkl_ffi.so", "_fkl_ffi.dylib", "lib_fkl_ffi.dylib"]
        print(f"\nTrying alternative library names:")
        for name in alt_names:
            alt_path = Path(fkl_ffi.__file__).parent / name
            if alt_path.exists():
                print(f"  ✓ Found: {alt_path}")
                break
            else:
                print(f"  ✗ Not found: {alt_path}")
    
    # Try importing Stream
    print(f"\n" + "="*60)
    print("Testing Stream import...")
    print("="*60)
    try:
        from fkl_ffi import Stream
        print("✓ Stream imported successfully")
        
        # Try creating a stream
        print("\nTesting Stream creation...")
        stream = Stream()
        print("✓ Stream created successfully")
        print(f"  Stream handle: {stream.handle}")
        
    except Exception as e:
        print(f"✗ Failed to import/create Stream: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ Failed to import fkl_ffi: {e}")
    import traceback
    traceback.print_exc()
    
    # Check if package is installed
    print(f"\nChecking if package is installed...")
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution("fused-kernel-library")
        print(f"  Package found: {dist}")
        print(f"  Location: {dist.location}")
    except:
        print("  Package not found in pkg_resources")
        # Try pip show
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "fused-kernel-library"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("  Package info from pip:")
                print(result.stdout)
        except:
            pass

