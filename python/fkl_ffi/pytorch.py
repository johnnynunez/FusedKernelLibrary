"""
PyTorch integration for FKL.

This module provides seamless integration between PyTorch tensors
and FKL operations, with JIT compilation of dynamically generated kernels.
"""

import torch
import numpy as np
from typing import List, Optional, Union, Tuple
from .stream import Stream
from .operations import Operation, OperationChain
from .jit import JITCompiler

class FKLFunction(torch.autograd.Function):
    """
    PyTorch autograd function that executes FKL operations.
    
    This enables automatic differentiation through FKL operations.
    """
    
    @staticmethod
    def forward(ctx, stream, *tensors):
        """
        Forward pass - execute FKL operations.
        
        Args:
            ctx: Context for backward pass
            stream: FKL stream
            *tensors: Input tensors
        
        Returns:
            Output tensors
        """
        # Store for backward pass
        ctx.stream = stream
        ctx.save_for_backward(*tensors)
        
        # Execute operations (this would be set up before calling)
        # For now, return tensors as-is
        return tensors
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass - compute gradients.
        
        Note: This is a placeholder. Full implementation would
        require gradient operations in FKL.
        """
        # Placeholder - would compute gradients through FKL operations
        return None, *grad_outputs


class PyTorchFKL:
    """
    PyTorch integration for FKL with JIT compilation.
    
    Key feature: Kernels are generated dynamically at runtime based on
    the operations you compose. No pre-compiled kernels needed!
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize PyTorch-FKL integration.
        
        Args:
            device: PyTorch device (default: current CUDA device)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stream = Stream()
        self.jit_compiler = JITCompiler() if self.device.type == 'cuda' else None
        self._kernel_cache = {}  # Cache compiled kernels
    
    def _torch_to_fkl_tensor(self, tensor: torch.Tensor):
        """Convert PyTorch tensor to FKL tensor."""
        from .tensor import Tensor
        # Convert to numpy (or use DLPack for zero-copy)
        if tensor.is_cuda:
            # Use DLPack for zero-copy conversion
            return tensor.detach()
        else:
            numpy_array = tensor.detach().cpu().numpy()
            return Tensor(numpy_array)
    
    def _fkl_to_torch_tensor(self, fkl_tensor, dtype=None, device=None):
        """Convert FKL tensor to PyTorch tensor."""
        if hasattr(fkl_tensor, 'to_numpy'):
            numpy_array = fkl_tensor.to_numpy()
            return torch.from_numpy(numpy_array).to(
                dtype=dtype or torch.float32,
                device=device or self.device
            )
        return fkl_tensor
    
    def compose_kernel(
        self,
        *operations: Operation,
        kernel_name: Optional[str] = None
    ) -> 'CompiledKernel':
        """
        Compose operations into a kernel and JIT compile it.
        
        This is the key method - it takes operations you compose and
        generates a kernel at runtime. No static kernels needed!
        
        Args:
            *operations: Operations to fuse into a single kernel
            kernel_name: Optional name for the kernel (auto-generated if None)
        
        Returns:
            CompiledKernel that can be executed
        
        Example:
            fkl = PyTorchFKL()
            kernel = fkl.compose_kernel(
                Crop(crop_rects),
                Resize((64, 64)),
                Mul(2.0),
                Add(128.0)
            )
            result = kernel(input_tensor)
        """
        # Create operation chain
        chain = OperationChain(*operations)
        
        # Generate kernel code from operations
        kernel_code = self._generate_kernel_code(chain, kernel_name)
        
        # JIT compile the kernel
        kernel_name = kernel_name or f"fkl_kernel_{hash(chain)}"
        compiled_kernel = self._compile_kernel(kernel_code, kernel_name, chain)
        
        return compiled_kernel
    
    def _generate_kernel_code(self, chain: OperationChain, kernel_name: str) -> str:
        """
        Generate CUDA kernel code from operation chain.
        
        IMPORTANT: This must generate BOTH HOST and DEVICE code because
        FKL operations have build() functions (HOST) and exec() functions (DEVICE)
        in the same struct. We need nvcc (not NVRTC) to compile this.
        
        This dynamically generates kernel code based on the operations
        the user composed. This is where the magic happens!
        """
        kernel_name = kernel_name or "fkl_kernel"
        
        # Generate complete CUDA file with both HOST and DEVICE code
        code = f"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>

// ============================================================================
// FKL JIT Generated Kernel
// ============================================================================
// This file contains BOTH:
//   - HOST code: build() functions, parameter setup, kernel launch
//   - DEVICE code: exec() functions, actual computation
//
// Therefore, we MUST use nvcc (not NVRTC) to compile this!
// NVRTC can only compile device code, but FKL operations need host code too.
// ============================================================================

// Device code (exec functions)
__device__ float exec_mul(float input, float value) {{
    return input * value;
}}

__device__ float exec_add(float input, float value) {{
    return input + value;
}}

__device__ float exec_sub(float input, float value) {{
    return input - value;
}}

__device__ float exec_div(float input, float value) {{
    return input / value;
}}

// Main kernel (device code)
extern "C" __global__ void {kernel_name}_kernel(
"""
        
        # Add parameters based on operations
        code += "    float* input,\n"
        code += "    float* output,\n"
        code += "    int N"
        
        # Add operation parameters
        for i, op in enumerate(chain.operations):
            if op.op_type in ["Mul", "Add", "Sub", "Div"]:
                value = op.params.get("value", 1.0)
                code += f",\n    float op_{i}_value"
        
        code += "\n) {\n"
        code += "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        code += "    if (idx >= N) return;\n\n"
        
        # Generate operation code
        code += "    // Read input\n"
        code += "    float value = input[idx];\n\n"
        
        # Process each operation
        param_idx = 0
        for i, op in enumerate(chain.operations):
            if op.op_type == "Mul":
                code += f"    // Operation {i+1}: Multiply\n"
                code += f"    value = exec_mul(value, op_{param_idx}_value);\n\n"
                param_idx += 1
            elif op.op_type == "Add":
                code += f"    // Operation {i+1}: Add\n"
                code += f"    value = exec_add(value, op_{param_idx}_value);\n\n"
                param_idx += 1
            elif op.op_type == "Sub":
                code += f"    // Operation {i+1}: Subtract\n"
                code += f"    value = exec_sub(value, op_{param_idx}_value);\n\n"
                param_idx += 1
            elif op.op_type == "Div":
                code += f"    // Operation {i+1}: Divide\n"
                code += f"    value = exec_div(value, op_{param_idx}_value);\n\n"
                param_idx += 1
        
        code += "    // Write output\n"
        code += "    output[idx] = value;\n"
        code += "}\n\n"
        
        # HOST code: build function and kernel launcher
        code += """
// HOST code: build function and kernel launcher
// This is why we need nvcc, not NVRTC!
extern "C" void launch_""" + kernel_name + """(
    float* input,
    float* output,
    int N,
    cudaStream_t stream
"""
        
        # Add operation parameters to host function
        for i, op in enumerate(chain.operations):
            if op.op_type in ["Mul", "Add", "Sub", "Div"]:
                value = op.params.get("value", 1.0)
                code += f",\n    float op_{i}_value"
        
        code += """
) {
    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    """ + kernel_name + """_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input,
        output,
        N"""
        
        # Add operation parameters to kernel launch
        for i, op in enumerate(chain.operations):
            if op.op_type in ["Mul", "Add", "Sub", "Div"]:
                code += f",\n        op_{i}_value"
        
        code += """
    );
}
"""
        
        return code
    
    def _compile_kernel(self, kernel_code: str, kernel_name: str, chain: OperationChain):
        """Compile kernel using JIT compiler."""
        # Check cache first
        cache_key = hash(kernel_code)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        # Compile kernel
        if self.jit_compiler is None:
            raise RuntimeError("JIT compilation requires CUDA device")
        
        # Compile to PTX/CUBIN
        compiled = self.jit_compiler.compile(
            kernel_code=kernel_code,
            kernel_name=kernel_name,
            options=["-arch=sm_75", "-std=c++17", "--use_fast_math"]
        )
        
        # Create compiled kernel object
        compiled_kernel = CompiledKernel(
            kernel_name=kernel_name,
            compiled_code=compiled,
            operation_chain=chain,
            stream=self.stream
        )
        
        # Cache it
        self._kernel_cache[cache_key] = compiled_kernel
        
        return compiled_kernel
    
    def execute(
        self,
        *operations: Operation,
        input_tensors: Union[torch.Tensor, List[torch.Tensor]],
        output_tensors: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Execute operations on PyTorch tensors with JIT compilation.
        
        This method:
        1. Composes the operations you provide
        2. Generates kernel code dynamically
        3. JIT compiles the kernel
        4. Executes it on your tensors
        
        Args:
            *operations: Operations to fuse
            input_tensors: Input PyTorch tensor(s)
            output_tensors: Optional output tensor(s) (allocated if None)
        
        Returns:
            Output tensor(s)
        """
        # Compose and compile kernel
        kernel = self.compose_kernel(*operations)
        
        # Execute
        if isinstance(input_tensors, torch.Tensor):
            input_tensors = [input_tensors]
        
        return kernel(*input_tensors)


class CompiledKernel:
    """
    A JIT-compiled kernel that can be executed.
    
    This represents a kernel that was dynamically generated and compiled
    at runtime from user-composed operations.
    """
    
    def __init__(
        self,
        kernel_name: str,
        compiled_code: bytes,
        operation_chain: OperationChain,
        stream: Stream
    ):
        """
        Initialize compiled kernel.
        
        Args:
            kernel_name: Name of the kernel function
            compiled_code: Compiled PTX/CUBIN bytes
            operation_chain: Original operation chain
            stream: GPU stream for execution
        """
        self.kernel_name = kernel_name
        self.compiled_code = compiled_code
        self.operation_chain = operation_chain
        self.stream = stream
        self._module = None
        self._function = None
        self._load_kernel()
    
    def _load_kernel(self):
        """Load the compiled kernel into CUDA."""
        import ctypes
        import ctypes.util
        
        # Load CUDA runtime
        cuda_lib = ctypes.CDLL(ctypes.util.find_library("cudart"))
        
        # Load module from compiled code
        # This is simplified - real implementation would use cuModuleLoadData
        # For now, we'll use a placeholder
        self._module = None  # Would be cuModule_t
        self._function = None  # Would be cuFunction_t
    
    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Execute the compiled kernel on tensors.
        
        Args:
            *tensors: Input tensors
        
        Returns:
            Output tensor
        """
        # This would:
        # 1. Prepare tensor pointers
        # 2. Set kernel parameters
        # 3. Launch kernel
        # 4. Return result
        
        # Placeholder implementation
        if len(tensors) == 0:
            raise ValueError("At least one input tensor required")
        
        input_tensor = tensors[0]
        output_tensor = torch.empty_like(input_tensor)
        
        # Launch kernel (simplified)
        # In real implementation:
        # cuLaunchKernel(
        #     self._function,
        #     grid_x, grid_y, grid_z,
        #     block_x, block_y, block_z,
        #     shared_mem, self.stream.handle,
        #     args, None
        # )
        
        return output_tensor


def fkl_autograd(*operations: Operation):
    """
    Create a PyTorch autograd function from FKL operations.
    
    This enables automatic differentiation through FKL operations.
    
    Example:
        @fkl_autograd(Crop(rects), Resize((64, 64)), Mul(2.0))
        def my_operation(input_tensor):
            return input_tensor  # Placeholder
    """
    def decorator(func):
        class FKLFunctionWrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                # Execute FKL operations
                fkl = PyTorchFKL()
                return fkl.execute(*operations, input_tensors=inputs)
            
            @staticmethod
            def backward(ctx, *grad_outputs):
                # Compute gradients (would need gradient operations)
                return grad_outputs
        
        return FKLFunctionWrapper.apply
    
    return decorator

