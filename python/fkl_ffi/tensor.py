"""
Tensor module for FKL Python bindings.

Provides tensor operations compatible with DLPack.
"""

import tvm_ffi
import numpy as np
import ctypes
from typing import Tuple, Optional

# DLManagedTensor might not be directly available from dlpack package
# It's typically part of the C API, not the Python package
try:
    from dlpack import DLManagedTensor
except ImportError:
    # DLManagedTensor is not available, we'll define a minimal structure if needed
    # For now, we'll work with DLTensor which is available via tvm_ffi
    DLManagedTensor = None

class Tensor:
    """
    Tensor wrapper for FKL operations.
    
    Provides a DLPack-compatible tensor interface for GPU operations.
    """
    
    def __init__(self, data: np.ndarray, device: str = "cuda"):
        """
        Create a tensor from a numpy array.
        
        Args:
            data: NumPy array containing the data
            device: Device type ("cuda" or "cpu")
        """
        self._handle = None
        self._lib = None
        self._data = data
        self._device = device
        self._init_lib()
        self._create_tensor()
    
    def _init_lib(self):
        """Initialize the FKL FFI library."""
        import ctypes
        import os
        from pathlib import Path
        
        # The library should be in the same directory as this Python file
        # (fkl_ffi/_fkl_ffi.so in site-packages)
        lib_name = "_fkl_ffi"
        if os.name == 'nt':  # Windows
            lib_name = f"{lib_name}.dll"
        elif os.name == 'posix':  # Linux/macOS
            lib_name = f"{lib_name}.so" if os.uname().sysname != 'Darwin' else f"{lib_name}.dylib"
        
        # Try multiple paths in order of preference
        possible_paths = []
        
        # First, try to find package directory
        package_dir = None
        if __file__:
            package_dir = Path(__file__).parent
        else:
            # Try to find package directory using importlib
            try:
                import importlib.util
                spec = importlib.util.find_spec("fkl_ffi")
                if spec and spec.origin and spec.origin != "namespace":
                    package_dir = Path(spec.origin).parent
                elif spec and spec.submodule_search_locations:
                    for loc in spec.submodule_search_locations:
                        if Path(loc).exists():
                            package_dir = Path(loc)
                            break
            except:
                pass
        
        # Check build directory (where scikit-build-core puts it during development)
        # Project root is likely in sys.path for editable installs
        import sys
        for path_str in sys.path:
            path = Path(path_str)
            if path.exists() and path.is_absolute():
                # Check if this looks like project root (has python/ and build/ directories)
                build_path = path / "build" / "python" / lib_name
                if build_path.exists():
                    possible_paths.append(build_path)
                # Also check python/fkl_ffi/ in project root
                python_pkg_path = path / "python" / "fkl_ffi" / lib_name
                if python_pkg_path.exists():
                    possible_paths.append(python_pkg_path)
        
        # Check package directory (for installed packages)
        if package_dir:
            possible_paths.append(package_dir / lib_name)
        
        # Check current working directory
        cwd_build = Path.cwd() / "build" / "python" / lib_name
        if cwd_build.exists():
            possible_paths.append(cwd_build)
        possible_paths.append(Path.cwd() / lib_name)
        
        lib_path = None
        for path in possible_paths:
            if path.exists():
                lib_path = path
                break
        
        if lib_path is None:
            # Try system library path (e.g., LD_LIBRARY_PATH)
            try:
                self._lib = ctypes.CDLL(lib_name)
            except OSError:
                raise RuntimeError(
                    f"Failed to find FKL FFI library '{lib_name}'. "
                    f"Tried paths: {[str(p) for p in possible_paths]}. "
                    f"Make sure the library is built and installed."
                )
        else:
            self._lib = ctypes.CDLL(str(lib_path))
        
        # Set up function signatures
        self._lib.FKLTensorCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # DLTensor*
            ctypes.POINTER(ctypes.c_void_p)   # FKLTensorHandle*
        ]
        self._lib.FKLTensorCreate.restype = ctypes.c_int
        
        self._lib.FKLTensorDestroy.argtypes = [ctypes.c_void_p]
        self._lib.FKLTensorDestroy.restype = ctypes.c_int
        
        self._lib.FKLTensorGetDLTensor.argtypes = [
            ctypes.c_void_p,  # FKLTensorHandle
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))  # DLTensor**
        ]
        self._lib.FKLTensorGetDLTensor.restype = ctypes.c_int
    
    def _create_tensor(self):
        """Create the internal tensor handle."""
        # Convert numpy array to DLPack tensor
        dl_tensor = tvm_ffi.from_dlpack(self._data)
        
        # Create FKL tensor
        handle_ptr = ctypes.POINTER(ctypes.c_void_p)()
        ret = self._lib.FKLTensorCreate(
            ctypes.byref(dl_tensor),
            ctypes.byref(handle_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to create FKL tensor")
        self._handle = handle_ptr.contents.value
    
    def to_numpy(self) -> np.ndarray:
        """Convert tensor back to numpy array."""
        if self._handle is None:
            raise RuntimeError("Tensor is not initialized")
        
        # Get DLTensor
        dltensor_ptr = ctypes.POINTER(ctypes.c_void_p)()
        ret = self._lib.FKLTensorGetDLTensor(
            ctypes.c_void_p(self._handle),
            ctypes.byref(dltensor_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to get DLTensor from FKL tensor")
        
        # Convert DLPack to numpy
        # This would need proper DLPack conversion
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the tensor shape."""
        return self._data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get the tensor data type."""
        return self._data.dtype
    
    @property
    def handle(self):
        """Get the internal tensor handle (for advanced usage)."""
        return self._handle
    
    def __del__(self):
        """Destroy the tensor when the object is garbage collected."""
        if self._handle is not None and self._lib is not None:
            try:
                self._lib.FKLTensorDestroy(ctypes.c_void_p(self._handle))
            except:
                pass  # Ignore errors during cleanup

