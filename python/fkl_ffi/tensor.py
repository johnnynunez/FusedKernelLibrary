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
        try:
            self._lib = tvm_ffi.load_module("_fkl_ffi")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FKL FFI library. Error: {e}"
            )
    
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

