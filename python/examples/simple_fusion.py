"""
Simple example demonstrating automatic kernel fusion.

This example shows how to combine operations that automatically fuse
into a single optimized kernel - no manual CUDA programming required!
"""

import numpy as np
from fkl_ffi import Stream, Tensor, execute_operations
from fkl_ffi.operations import Mul, Add, TensorRead, TensorWrite

def main():
    # Create GPU stream
    stream = Stream()
    
    # Create input and output tensors
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    output_data = np.zeros_like(input_data)
    
    input_tensor = Tensor(input_data)
    output_tensor = Tensor(output_data)
    
    # Automatically creates a single fused kernel that:
    # 1. Reads from input_tensor
    # 2. Multiplies by 2.0
    # 3. Adds 1.0
    # 4. Writes to output_tensor
    # All in one kernel, data stays in registers!
    execute_operations(
        stream,
        TensorRead(input_tensor),
        Mul(2.0),  # Automatically fused!
        Add(1.0),  # Automatically fused!
        TensorWrite(output_tensor)
    )
    
    # Wait for completion
    stream.sync()
    
    # Get results
    result = output_tensor.to_numpy()
    print("Input:", input_data)
    print("Output:", result)
    print("Expected:", input_data * 2.0 + 1.0)
    
    assert np.allclose(result, input_data * 2.0 + 1.0)
    print("✓ Test passed!")


if __name__ == "__main__":
    main()

