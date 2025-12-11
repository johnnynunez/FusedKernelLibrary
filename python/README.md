# Fused Kernel Library - Python Bindings

Python bindings for the Fused Kernel Library using TVM-FFI, enabling automatic GPU kernel fusion with a high-level Python API.

## Key Features

### 🚀 Automatic Kernel Fusion
- **Vertical Fusion**: Combine operations, data stays in GPU registers
- **Horizontal Fusion**: Automatically batch operations - no manual batching code needed
- **Exponential Composability**: Each new operation adds 1000s of possible kernel combinations

### 💡 No Manual Kernel Writing
- Users compose operations using a high-level API
- Library automatically generates optimized fused kernels at compile time
- No CUDA knowledge required!

### ⚡ Performance
- Up to 20,000× speedup vs traditional libraries
- Eliminates intermediate memory allocations
- Maximum GPU resource utilization

## Installation

```bash
# Install dependencies
pip install numpy tvm-ffi dlpack

# Build and install FKL
pip install -e .
```

## Quick Start

```python
from fkl_ffi import Stream, Tensor, execute_operations
from fkl_ffi.operations import Mul, Add, TensorRead, TensorWrite

# Create stream and tensors
stream = Stream()
input_tensor = Tensor(input_data)
output_tensor = Tensor(output_shape)

# Automatically creates a fused kernel!
execute_operations(
    stream,
    TensorRead(input_tensor),
    Mul(2.0),      # Automatically fused!
    Add(1.0),      # Automatically fused!
    TensorWrite(output_tensor)
)
stream.sync()
```

## Examples

### Simple Fusion
```python
# See examples/simple_fusion.py
# Combines multiple operations into one kernel
```

### Automatic Batching
```python
# See examples/batched_operations.py
# Process multiple crops - automatically batched!
crop_rects = [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
execute_operations(
    stream,
    TensorRead(source),
    Crop(crop_rects),  # Automatically handles batching!
    Resize((64, 64)),
    TensorWrite(output)
)
```

### Complex Pipeline
```python
# See examples/complex_pipeline.py
# Real-world image processing pipeline as single kernel
```

## Architecture

### Operations
Operations are composable building blocks:
- **Unary Operations**: Transform data (Mul, Add, Resize, etc.)
- **Binary Operations**: Operations with parameters (Crop with rects, etc.)
- **Read Operations**: Read from GPU memory
- **Write Operations**: Write to GPU memory

### Automatic Fusion
When you combine operations:
1. **Vertical Fusion**: Operations chain together, data flows through registers
2. **Horizontal Fusion**: Multiple data items processed in parallel
3. **Backwards Vertical Fusion**: Only reads data needed for final output

### No Manual Kernels
- Each operation you add multiplies the possible combinations
- Library generates optimized kernels automatically
- No need to write batched versions - it's automatic!

## Building

```bash
# Configure CMake
cmake -B build -DBUILD_PYTHON_BINDINGS=ON

# Build
cmake --build build

# Install
pip install -e .
```

## JIT Compilation

**Key Feature**: Kernels are generated dynamically at runtime! No pre-compiled kernels needed.

### Basic JIT Usage

```python
from fkl_ffi import JITCompiler

jit = JITCompiler()
compiled_kernel = jit.compile(
    kernel_code="...",
    kernel_name="my_kernel",
    options=["-arch=sm_75"]
)
```

### PyTorch Integration with JIT

```python
import torch
from fkl_ffi.pytorch import PyTorchFKL
from fkl_ffi.operations import Mul, Add

# Initialize
fkl = PyTorchFKL(device=torch.device('cuda'))

# Compose operations - kernel generated at runtime!
kernel = fkl.compose_kernel(
    Mul(2.0),
    Add(1.0)
)

# Execute on PyTorch tensors
input_tensor = torch.randn(1000, device='cuda')
output = kernel(input_tensor)
```

**How it works:**
1. You compose operations in Python
2. FKL generates CUDA kernel code dynamically
3. Kernel is JIT compiled at runtime using NVRTC
4. Compiled kernel is executed on your tensors

**Benefits:**
- No static kernels - infinite combinations possible
- Each operation combo = unique optimized kernel
- Automatic fusion and batching
- No CUDA knowledge required

## Performance

The library achieves significant speedups by:
- Eliminating intermediate memory writes
- Keeping data in GPU registers
- Maximizing latency hiding
- Automatic batching for better GPU utilization

See the research paper for detailed benchmarks.

## License

Apache 2.0

