"""
JIT compilation module for FKL Python bindings.

Provides Just-In-Time compilation of CUDA kernels.
"""

import tvm_ffi
import ctypes
from typing import Optional, Union

class JITCompiler:
    """
    Just-In-Time compiler for CUDA kernels.
    
    Compiles CUDA kernel code at runtime using NVRTC.
    """
    
    def __init__(self):
        """Initialize the JIT compiler."""
        self._lib = None
        self._init_lib()
    
    def _init_lib(self):
        """Initialize the FKL FFI library."""
        try:
            self._lib = tvm_ffi.load_module("_fkl_ffi")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FKL FFI library. Error: {e}"
            )
    
    def compile(
        self,
        kernel_code: str,
        kernel_name: str,
        options: Optional[Union[str, list]] = None
    ) -> bytes:
        """
        Compile a CUDA kernel using nvcc (not NVRTC).
        
        IMPORTANT: FKL operations have both HOST code (build functions) 
        and DEVICE code (exec functions) in the same struct. Therefore,
        we MUST use nvcc to compile, not NVRTC (which only compiles device code).
        
        Args:
            kernel_code: Complete CUDA source code (host + device)
            kernel_name: Name of the kernel function
            options: Compilation options (string or list of strings)
                    Default: ["-arch=sm_75"]
        
        Returns:
            Compiled kernel as bytes (.so file)
        """
        import ctypes
        
        if options is None:
            options = "-arch=sm_75"
        elif isinstance(options, list):
            options = " ".join(options)
        
        # Call JIT compile function (uses nvcc internally)
        kernel_code_bytes = kernel_code.encode('utf-8')
        kernel_name_bytes = kernel_name.encode('utf-8')
        options_bytes = options.encode('utf-8')
        
        # Allocate output buffers
        cubin_ptr = ctypes.POINTER(ctypes.c_void_p)()
        cubin_size = ctypes.c_size_t()
        
        ret = self._lib.FKLJITCompileKernel(
            kernel_code_bytes,
            kernel_name_bytes,
            options_bytes,
            ctypes.byref(cubin_ptr),
            ctypes.byref(cubin_size)
        )
        
        if ret != 0:
            raise RuntimeError(f"Failed to compile kernel '{kernel_name}' with nvcc")
        
        # Copy compiled code
        result = ctypes.string_at(cubin_ptr, cubin_size.value)
        
        # Free allocated memory
        if cubin_ptr:
            ctypes.CDLL(None).free(cubin_ptr)
        
        return bytes(result)
    
    def load_module(self, module_path: str):
        """
        Load a pre-compiled kernel module.
        
        Args:
            module_path: Path to the compiled module (.so or .cubin file)
        
        Returns:
            TVM-FFI module handle
        """
        ret = self._lib.FKLJITLoadModule(
            module_path.encode('utf-8'),
            ctypes.byref(ctypes.c_void_p())
        )
        if ret != 0:
            raise RuntimeError(f"Failed to load module from {module_path}")
        
        # Return module handle
        return tvm_ffi.load_module(module_path)

