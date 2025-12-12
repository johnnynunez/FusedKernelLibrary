"""
Fused Kernel Library (FKL) Python bindings using TVM-FFI.

This module provides Python bindings for the Fused Kernel Library,
enabling automatic kernel fusion for GPU operations.

Key Features:
- Automatic Vertical Fusion: Combine operations, data stays in registers
- Automatic Horizontal Fusion: Batch operations automatically
- No Manual Kernels: Users compose operations, library generates optimized kernels
- Exponential Composability: Each new operation adds 1000s of possible combinations

Example:
    from fkl_ffi import Stream, Tensor, execute_operations
    from fkl_ffi.operations import Crop, Resize, Mul, Add, TensorRead, TensorWrite
    
    stream = Stream()
    input_tensor = Tensor(input_data)
    output_tensor = Tensor(output_shape)
    
    # Automatically creates a fused kernel - no CUDA knowledge needed!
    execute_operations(
        stream,
        TensorRead(input_tensor),
        Crop(crop_rects),  # Automatically batched for multiple crops!
        Resize((64, 128)),
        Mul(2.0),
        Add(128.0),
        TensorWrite(output_tensor)
    )
    stream.sync()
"""

__version__ = "0.2.0"

from .stream import Stream
from .tensor import Tensor, from_dlpack
from .jit import JITCompiler
from .operations import (
    Operation,
    UnaryOperation,
    BinaryOperation,
    ReadOperation,
    WriteOperation,
    Mul,
    Add,
    Sub,
    Div,
    Crop,
    Resize,
    ColorConvert,
    SaturateCast,
    TensorRead,
    TensorWrite,
    OperationChain,
    execute_operations,
)

# PyTorch integration (optional, requires PyTorch)
try:
    from .pytorch import PyTorchFKL, CompiledKernel, fkl_autograd
    __all__ = [
        "Stream",
        "Tensor",
        "from_dlpack",
        "JITCompiler",
        "Operation",
        "UnaryOperation",
        "BinaryOperation",
        "ReadOperation",
        "WriteOperation",
        "Mul",
        "Add",
        "Sub",
        "Div",
        "Crop",
        "Resize",
        "ColorConvert",
        "SaturateCast",
        "TensorRead",
        "TensorWrite",
        "OperationChain",
        "execute_operations",
        "PyTorchFKL",
        "CompiledKernel",
        "fkl_autograd",
    ]
except ImportError:
    # PyTorch not available
    __all__ = [
        "Stream",
        "Tensor",
        "from_dlpack",
        "JITCompiler",
        "Operation",
        "UnaryOperation",
        "BinaryOperation",
        "ReadOperation",
        "WriteOperation",
        "Mul",
        "Add",
        "Sub",
        "Div",
        "Crop",
        "Resize",
        "ColorConvert",
        "SaturateCast",
        "TensorRead",
        "TensorWrite",
        "OperationChain",
        "execute_operations",
    ]
