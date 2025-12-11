"""
Stream module for FKL Python bindings.

Provides GPU stream management for asynchronous execution.
"""

import ctypes
from typing import Optional

class Stream:
    """
    GPU stream for asynchronous kernel execution.
    
    A stream represents a sequence of operations that execute in order
    on the GPU. Multiple streams can execute concurrently.
    """
    
    def __init__(self, cuda_stream: Optional[int] = None):
        """
        Create a new GPU stream.
        
        Args:
            cuda_stream: Optional CUDA stream handle. If None, creates a new stream.
        """
        self._handle = None
        self._lib = None
        self._init_lib()
        
        if cuda_stream is not None:
            # Create from existing CUDA stream
            handle_ptr = ctypes.c_void_p()
            ret = self._lib.FKLStreamFromCUDAStream(
                ctypes.byref(handle_ptr),
                ctypes.c_void_p(cuda_stream)
            )
            if ret != 0:
                raise RuntimeError("Failed to create FKL stream from CUDA stream")
            self._handle = handle_ptr.value
        else:
            # Create new stream
            handle_ptr = ctypes.c_void_p()
            ret = self._lib.FKLStreamCreate(ctypes.byref(handle_ptr))
            if ret != 0:
                raise RuntimeError("Failed to create FKL stream")
            self._handle = handle_ptr.value
    
    def _init_lib(self):
        """Initialize the FKL FFI library."""
        import ctypes
        import os
        from pathlib import Path
        
        # Try to load the shared library
        lib_name = "_fkl_ffi"
        if os.name == 'nt':  # Windows
            lib_name = f"{lib_name}.dll"
        elif os.name == 'posix':  # Linux/macOS
            lib_name = f"lib{lib_name}.so" if os.uname().sysname != 'Darwin' else f"lib{lib_name}.dylib"
        
        # Try multiple paths
        possible_paths = [
            Path(__file__).parent.parent / lib_name,
            Path(__file__).parent.parent.parent / "build" / "python" / lib_name,
            Path.cwd() / lib_name,
        ]
        
        lib_path = None
        for path in possible_paths:
            if path.exists():
                lib_path = path
                break
        
        if lib_path is None:
            # Try system library path
            try:
                self._lib = ctypes.CDLL(lib_name)
            except OSError:
                raise RuntimeError(
                    f"Failed to find FKL FFI library '{lib_name}'. "
                    f"Make sure the library is built and available."
                )
        else:
            self._lib = ctypes.CDLL(str(lib_path))
        
        # Set up function signatures
        self._lib.FKLStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.FKLStreamCreate.restype = ctypes.c_int
        
        self._lib.FKLStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.FKLStreamDestroy.restype = ctypes.c_int
        
        self._lib.FKLStreamSync.argtypes = [ctypes.c_void_p]
        self._lib.FKLStreamSync.restype = ctypes.c_int
        
        self._lib.FKLStreamFromCUDAStream.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p
        ]
        self._lib.FKLStreamFromCUDAStream.restype = ctypes.c_int
    
    def sync(self):
        """Synchronize the stream, waiting for all operations to complete."""
        if self._handle is None:
            raise RuntimeError("Stream is not initialized")
        
        ret = self._lib.FKLStreamSync(ctypes.c_void_p(self._handle))
        if ret != 0:
            raise RuntimeError("Failed to sync FKL stream")
    
    def __del__(self):
        """Destroy the stream when the object is garbage collected."""
        if self._handle is not None and self._lib is not None:
            try:
                self._lib.FKLStreamDestroy(ctypes.c_void_p(self._handle))
            except:
                pass  # Ignore errors during cleanup
    
    @property
    def handle(self):
        """Get the internal stream handle (for advanced usage)."""
        return self._handle

