"""
Example demonstrating automatic horizontal fusion (batching).

This shows how operations are automatically batched - no need to write
batched versions of operations!
"""

import numpy as np
from fkl_ffi import Stream, Tensor, execute_operations
from fkl_ffi.operations import Crop, Resize, Mul, Add, TensorRead, TensorWrite

def main():
    """
    Process multiple image crops in a single fused kernel.
    
    With traditional libraries, you'd need to:
    1. Write separate kernels for each crop
    2. Write a batched version of each operation
    3. Manually manage memory for intermediate results
    
    With FKL, just pass multiple crops - it's automatically batched!
    """
    stream = Stream()
    
    # Source image (e.g., 4K image)
    source_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    source_tensor = Tensor(source_image)
    
    # Define multiple crops (automatically batched!)
    crop_rects = [
        (300, 125, 60, 40),   # Crop 1
        (400, 125, 60, 40),   # Crop 2
        (530, 270, 130, 140), # Crop 3
        (560, 115, 100, 35),  # Crop 4
        (572, 196, 40, 15),   # Crop 5
    ]
    
    # Output tensor for all crops
    output_size = (60, 60, 3)
    batch_size = len(crop_rects)
    output_tensor = Tensor(np.zeros((batch_size, *output_size), dtype=np.uint8))
    
    # Single fused kernel that:
    # 1. Reads from source (with 5 different crops - automatically batched!)
    # 2. Resizes each crop to 60x60 (automatically batched!)
    # 3. Multiplies by 2.0 (automatically batched!)
    # 4. Adds 128.0 (automatically batched!)
    # 5. Writes all results (automatically batched!)
    #
    # This creates ONE kernel that processes all 5 crops in parallel!
    # No manual batching code needed!
    execute_operations(
        stream,
        TensorRead(source_tensor),
        Crop(crop_rects),  # Automatically handles batching!
        Resize((60, 60)),
        Mul(2.0),
        Add(128.0),
        TensorWrite(output_tensor)
    )
    
    stream.sync()
    
    print(f"✓ Processed {batch_size} crops in a single fused kernel!")
    print(f"  - No manual batching code")
    print(f"  - No intermediate memory allocations")
    print(f"  - Maximum GPU utilization")


if __name__ == "__main__":
    main()

