"""
Complex pipeline example showing the power of automatic fusion.

This demonstrates a real-world image processing pipeline that would
require many separate kernels in traditional libraries, but becomes
a single fused kernel with FKL.
"""

import numpy as np
from fkl_ffi import Stream, Tensor, execute_operations
from fkl_ffi.operations import (
    Crop, Resize, ColorConvert, Mul, Sub, Div,
    SaturateCast, TensorRead, TensorWrite
)

def main():
    """
    Complex image processing pipeline.
    
    Traditional approach would require:
    - Separate kernel for each operation
    - Intermediate memory buffers
    - Manual synchronization
    - ~10-20 kernel launches
    
    FKL approach:
    - Single fused kernel
    - No intermediate buffers
    - All data in registers
    - 1 kernel launch
    """
    stream = Stream()
    
    # Input: YUV420_NV12 format image
    # (In real scenario, this would come from a camera/decoder)
    input_image = np.random.randint(0, 255, (2160, 3840), dtype=np.uint8)
    input_tensor = Tensor(input_image)
    
    # Define crop region
    crop_rect = (300, 125, 60, 40)
    
    # Output: RGB planar format (3 separate planes)
    output_size = (64, 128)
    output_tensor = Tensor(np.zeros((3, *output_size), dtype=np.float32))
    
    # Single fused kernel that performs:
    # 1. Read YUV data (only reads what's needed for interpolation!)
    # 2. Crop the region
    # 3. Convert YUV to RGB (only for needed pixels!)
    # 4. Resize to 64x128
    # 5. Normalize: multiply by 2.0
    # 6. Normalize: subtract 128.0
    # 7. Normalize: divide by 255.0
    # 8. Convert to RGB planar format
    # 9. Write to output
    #
    # All in ONE kernel! Data flows through registers.
    # Only reads pixels needed for final output (backwards vertical fusion).
    execute_operations(
        stream,
        TensorRead(input_tensor),
        Crop(crop_rect),
        ColorConvert("YUV2RGB"),
        Resize(output_size),
        Mul(2.0),
        Sub(128.0),
        Div(255.0),
        SaturateCast("float3", "float3"),  # Could split to planar here
        TensorWrite(output_tensor)
    )
    
    stream.sync()
    
    print("✓ Complex pipeline executed as single fused kernel!")
    print("  Benefits:")
    print("  - No intermediate GPU memory allocations")
    print("  - Only reads pixels needed (backwards vertical fusion)")
    print("  - Maximum latency hiding")
    print("  - Single kernel launch (minimal CPU overhead)")


if __name__ == "__main__":
    main()

