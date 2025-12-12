"""
Tensor module for FKL Python bindings.

Provides tensor operations compatible with DLPack.
"""

import tvm_ffi
import numpy as np
import ctypes
from typing import Tuple, Optional
from dlpack import DLManagedTensor

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
        possible_paths = [
            Path(__file__).parent / lib_name,  # Same directory as Python package (installed location)
            Path(__file__).parent.parent / "build" / "python" / lib_name,  # Build directory (development)
            Path(__file__).parent.parent / lib_name,  # Parent directory
            Path.cwd() / lib_name,  # Current working directory
        ]
        
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

