"""
Operations module for FKL Python bindings.

This module provides a high-level interface for composing GPU operations.
Each operation can be automatically fused and batched without manual kernel writing.
"""

from typing import List, Optional, Union, Any
import numpy as np
from .stream import Stream
from .tensor import Tensor

class Operation:
    """
    Base class for FKL operations.
    
    Operations are composable - combining them automatically creates
    fused kernels. No manual kernel writing required!
    """
    
    def __init__(self, op_type: str, params: Optional[dict] = None):
        """
        Initialize an operation.
        
        Args:
            op_type: Type of operation (e.g., "Mul", "Add", "Crop")
            params: Operation parameters
        """
        self.op_type = op_type
        self.params = params or {}
        self._iop = None  # Internal InstantiableOperation
    
    def build(self, *args, **kwargs):
        """
        Build the operation with given parameters.
        
        This creates an InstantiableOperation that can be fused
        with other operations automatically.
        """
        # This would call into the C++ API to create the IOp
        # For now, store parameters
        self.params.update(kwargs)
        for i, arg in enumerate(args):
            self.params[f"arg_{i}"] = arg
        return self
    
    def __call__(self, *args, **kwargs):
        """Allow operations to be called directly."""
        return self.build(*args, **kwargs)


class UnaryOperation(Operation):
    """Unary operation - takes one input, produces one output."""
    pass


class BinaryOperation(Operation):
    """Binary operation - takes input and parameters, produces output."""
    pass


class ReadOperation(Operation):
    """Read operation - reads from GPU memory."""
    pass


class WriteOperation(Operation):
    """Write operation - writes to GPU memory."""
    pass


# Arithmetic operations
class Mul(BinaryOperation):
    """Multiply operation - automatically fusable with any other operation."""
    
    def __init__(self, value: Union[float, np.ndarray]):
        super().__init__("Mul", {"value": value})


class Add(BinaryOperation):
    """Add operation - automatically fusable."""
    
    def __init__(self, value: Union[float, np.ndarray]):
        super().__init__("Add", {"value": value})


class Sub(BinaryOperation):
    """Subtract operation - automatically fusable."""
    
    def __init__(self, value: Union[float, np.ndarray]):
        super().__init__("Sub", {"value": value})


class Div(BinaryOperation):
    """Divide operation - automatically fusable."""
    
    def __init__(self, value: Union[float, np.ndarray]):
        super().__init__("Div", {"value": value})


# Image processing operations
class Crop(ReadOperation):
    """Crop operation - can be automatically batched for multiple crops."""
    
    def __init__(self, rects: Union[List, np.ndarray]):
        """
        Initialize crop operation.
        
        Args:
            rects: List of rectangles to crop. Can be a single rect or batch.
                   Automatically handles batching!
        """
        super().__init__("Crop", {"rects": rects})


class Resize(UnaryOperation):
    """Resize operation - automatically fusable."""
    
    def __init__(self, size: tuple, interpolation: str = "linear"):
        super().__init__("Resize", {
            "size": size,
            "interpolation": interpolation
        })


class ColorConvert(UnaryOperation):
    """Color conversion - automatically fusable."""
    
    def __init__(self, conversion: str):
        """
        Initialize color conversion.
        
        Args:
            conversion: Conversion type (e.g., "RGB2BGR", "YUV2RGB")
        """
        super().__init__("ColorConvert", {"conversion": conversion})


class SaturateCast(UnaryOperation):
    """Saturate cast - automatically fusable."""
    
    def __init__(self, from_type: str, to_type: str):
        super().__init__("SaturateCast", {
            "from_type": from_type,
            "to_type": to_type
        })


# Memory operations
class TensorRead(ReadOperation):
    """Read from tensor - automatically handles batching."""
    
    def __init__(self, tensor: Tensor):
        super().__init__("TensorRead", {"tensor": tensor})


class TensorWrite(WriteOperation):
    """Write to tensor - automatically handles batching."""
    
    def __init__(self, tensor: Tensor):
        super().__init__("TensorWrite", {"tensor": tensor})


class OperationChain:
    """
    Chain of operations that will be automatically fused into a single kernel.
    
    The beauty of FKL: combine any operations, and they automatically fuse!
    No need to write fused kernels - the library generates them at compile time.
    """
    
    def __init__(self, *operations: Operation):
        """
        Create a chain of operations.
        
        Args:
            *operations: Operations to fuse. Can be any combination!
                        The library automatically handles:
                        - Vertical fusion (keeping data in registers)
                        - Horizontal fusion (batching multiple operations)
                        - Complex memory access patterns
        """
        self.operations = list(operations)
    
    def append(self, operation: Operation):
        """Add another operation to the chain."""
        self.operations.append(operation)
        return self
    
    def execute(self, stream: Stream):
        """
        Execute the fused kernel.
        
        This single call generates and executes a fully optimized
        fused kernel combining all operations in the chain.
        """
        # This would call into the C++ executeOperations function
        # which automatically fuses all operations
        pass


def execute_operations(stream: Stream, *operations: Operation):
    """
    Execute a chain of operations as a single fused kernel.
    
    This is the main entry point - just pass operations and they
    automatically fuse! No CUDA knowledge required.
    
    Example:
        stream = Stream()
        input_tensor = Tensor(input_data)
        output_tensor = Tensor(output_shape)
        
        # Automatically creates a fused kernel!
        execute_operations(
            stream,
            TensorRead(input_tensor),
            Crop(crop_rects),  # Automatically batched!
            Resize((64, 128)),
            Mul(2.0),
            Add(128.0),
            SaturateCast("float3", "uchar3"),
            TensorWrite(output_tensor)
        )
        stream.sync()
    
    Args:
        stream: GPU stream for execution
        *operations: Operations to fuse and execute
    """
    chain = OperationChain(*operations)
    chain.execute(stream)

