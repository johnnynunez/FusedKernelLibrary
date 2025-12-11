# Building FKL Python Bindings

## Prerequisites

1. **TVM-FFI**: Install the TVM-FFI library
   ```bash
   pip install apache-tvm-ffi
   ```

2. **CMake 3.22+**: Required for building

3. **CUDA** (optional): For GPU support
   - CUDA Toolkit 11.0+
   - nvcc compiler

4. **Python 3.8+**: With development headers

## Building

### Option 1: Using pip (Recommended)

```bash
# Install build dependencies
pip install scikit-build-core cmake numpy

# Build and install
pip install -e .
```

### Option 2: Using CMake directly

```bash
# Configure
cmake -B build \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DENABLE_JIT=ON \
    -DENABLE_CUDA=ON

# Build
cmake --build build

# The library will be in build/python/_fkl_ffi.so (or .dylib on macOS)
```

### Option 3: Using setup.py

```bash
python setup.py build_ext --inplace
```

## Finding TVM-FFI

The build system will try to find TVM-FFI in this order:

1. CMake package: `find_package(tvm-ffi)`
2. pkg-config: `pkg_check_modules(TVM_FFI tvm-ffi)`
3. tvm-ffi-config: Uses `tvm-ffi-config --cflags --ldflags --libs`

If TVM-FFI is not found, you can specify it manually:

```bash
cmake -B build \
    -DTVM_FFI_INCLUDE_DIR=/path/to/tvm-ffi/include \
    -DTVM_FFI_LIBRARY=/path/to/tvm-ffi/lib/libtvm_ffi.so
```

## Building Python Wheels

```bash
# Install build tools
pip install build wheel

# Build wheel
python -m build

# The wheel will be in dist/
```

## Troubleshooting

### TVM-FFI not found
- Make sure `tvm-ffi` is installed: `pip install apache-tvm-ffi`
- Check if `tvm-ffi-config` is in PATH
- Try setting `TVM_FFI_ROOT` environment variable

### CUDA not found
- Set `CUDA_PATH` environment variable
- Or disable CUDA: `-DENABLE_CUDA=OFF`

### Python headers not found
- Install python-dev (Linux) or python3-dev
- On macOS: Usually included with Python
- On Windows: Install Python from python.org

## Development Build

For development with editable install:

```bash
pip install -e .[dev]
```

This installs in development mode and includes dev dependencies.

