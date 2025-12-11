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
        Compile a CUDA kernel to PTX/CUBIN.
        
        Args:
            kernel_code: CUDA kernel source code
            kernel_name: Name of the kernel function
            options: Compilation options (string or list of strings)
                    Default: ["-arch=sm_75"]
        
        Returns:
            Compiled kernel as bytes (PTX or CUBIN)
        """
        if options is None:
            options = "-arch=sm_75"
        elif isinstance(options, list):
            options = " ".join(options)
        
        # Call JIT compile function
        result = self._lib.get_global_func("fkl.JIT.compile_kernel")(
            kernel_code,
            kernel_name,
            options
        )
        
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

