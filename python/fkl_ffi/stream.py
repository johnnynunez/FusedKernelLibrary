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
    
    def __init__(self, cuda_stream: Optional[int] = None, hip_stream: Optional[int] = None):
        """
        Create a new GPU stream.
        
        Args:
            cuda_stream: Optional CUDA stream handle (for NVIDIA GPUs). 
                        If None, creates a new stream.
            hip_stream: Optional HIP stream handle (for AMD GPUs).
                       If None, creates a new stream.
        
        Note: Only one of cuda_stream or hip_stream should be provided.
        """
        self._handle = None
        self._lib = None
        self._init_lib()
        
        if hip_stream is not None:
            # Create from existing HIP stream (AMD GPU)
            handle_ptr = ctypes.c_void_p()
            ret = self._lib.FKLStreamFromHIPStream(
                ctypes.byref(handle_ptr),
                ctypes.c_void_p(hip_stream)
            )
            if ret != 0:
                raise RuntimeError("Failed to create FKL stream from HIP stream")
            self._handle = handle_ptr.value
        elif cuda_stream is not None:
            # Create from existing CUDA stream (NVIDIA GPU)
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
        
        self._lib.FKLStreamFromHIPStream.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p
        ]
        self._lib.FKLStreamFromHIPStream.restype = ctypes.c_int
    
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

