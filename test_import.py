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
    
    # Get package location
    package_dir = None
    if hasattr(fkl_ffi, '__file__') and fkl_ffi.__file__:
        package_dir = Path(fkl_ffi.__file__).parent
        print(f"  Location: {fkl_ffi.__file__}")
        print(f"  Package directory: {package_dir}")
    else:
        # Try to find the package directory
        print(f"  __file__ is None, searching for package...")
        import importlib.util
        spec = importlib.util.find_spec("fkl_ffi")
        if spec:
            print(f"  Spec found: {spec}")
            if spec.origin and spec.origin != "namespace":
                package_dir = Path(spec.origin).parent
                print(f"  Found via spec.origin: {spec.origin}")
                print(f"  Package directory: {package_dir}")
            elif spec.submodule_search_locations:
                # Namespace package - check all locations
                print(f"  Namespace package, checking locations:")
                for loc in spec.submodule_search_locations:
                    print(f"    - {loc}")
                    potential_dir = Path(loc)
                    if potential_dir.exists():
                        package_dir = potential_dir
                        print(f"  Using: {package_dir}")
                        break
        
        # Also search in site-packages and sys.path
        if not package_dir:
            print(f"  Searching sys.path...")
            for site_path in sys.path:
                potential_dir = Path(site_path) / "fkl_ffi"
                if potential_dir.exists() and potential_dir.is_dir():
                    package_dir = potential_dir
                    print(f"  Found in sys.path: {package_dir}")
                    break
            
            # Also check if we're in the source directory (editable install)
            source_dir = Path(__file__).parent / "python" / "fkl_ffi"
            if source_dir.exists():
                package_dir = source_dir
                print(f"  Found in source directory: {package_dir}")
        
        if not package_dir:
            print(f"  ✗ Could not determine package directory")
    
    # Check for library file
    if package_dir:
        lib_path = package_dir / "_fkl_ffi.so"
        print(f"\nLibrary file check:")
        print(f"  Expected: {lib_path}")
        print(f"  Exists: {lib_path.exists()}")
        
        if not lib_path.exists():
            # Try alternative names and locations
            alt_names = ["_fkl_ffi.so", "lib_fkl_ffi.so"]
            print(f"\nTrying alternative library names:")
            found = False
            for name in alt_names:
                alt_path = package_dir / name
                if alt_path.exists():
                    print(f"  ✓ Found: {alt_path}")
                    found = True
                    break
                else:
                    print(f"  ✗ Not found: {alt_path}")
            
            # Also check build directory
            build_path = Path(__file__).parent / "build" / "python" / "_fkl_ffi.so"
            if build_path.exists():
                print(f"  ✓ Found in build directory: {build_path}")
                found = True
    
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

