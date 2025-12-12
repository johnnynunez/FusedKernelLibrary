"""
Tensor module for FKL Python bindings.

Provides tensor operations compatible with DLPack.
"""

import numpy as np
import ctypes
from typing import Tuple, Optional, Union

# DLPack device type constants (from dlpack.h)
kDLCPU = 1
kDLCUDA = 2
kDLCUDAHost = 3
kDLOpenCL = 4
kDLVulkan = 5
kDLMetal = 6
kDLVPI = 7
kDLROCM = 10
kDLROCMHost = 11
kDLCUDAManaged = 13
kDLOneAPI = 14
kDLWebGPU = 15
kDLHexagon = 16

# DLPack data type codes
kDLInt = 0
kDLUInt = 1
kDLFloat = 2
kDLOpaqueHandle = 3
kDLBfloat = 4
kDLComplex = 5
kDLBool = 6

# Define DLTensor structure using ctypes
class DLDevice(ctypes.Structure):
    """DLDevice structure from DLPack."""
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int32),
    ]

class DLDataType(ctypes.Structure):
    """DLDataType structure from DLPack."""
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

class DLTensor(ctypes.Structure):
    """DLTensor structure from DLPack."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

# Define deleter function type
# Note: We use c_void_p here to avoid circular dependency, then cast when needed
DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

class DLManagedTensor(ctypes.Structure):
    """DLManagedTensor structure from DLPack (legacy version)."""
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", DLPACK_DELETER),  # Takes void_p, will be cast to DLManagedTensor* in implementation
    ]

def _numpy_dtype_to_dlpack(dtype: np.dtype) -> Tuple[int, int, int]:
    """Convert numpy dtype to DLPack dtype (code, bits, lanes)."""
    if dtype == np.bool_:
        return (kDLBool, 8, 1)
    elif dtype == np.int8:
        return (kDLInt, 8, 1)
    elif dtype == np.int16:
        return (kDLInt, 16, 1)
    elif dtype == np.int32:
        return (kDLInt, 32, 1)
    elif dtype == np.int64:
        return (kDLInt, 64, 1)
    elif dtype == np.uint8:
        return (kDLUInt, 8, 1)
    elif dtype == np.uint16:
        return (kDLUInt, 16, 1)
    elif dtype == np.uint32:
        return (kDLUInt, 32, 1)
    elif dtype == np.uint64:
        return (kDLUInt, 64, 1)
    elif dtype == np.float16:
        return (kDLFloat, 16, 1)
    elif dtype == np.float32:
        return (kDLFloat, 32, 1)
    elif dtype == np.float64:
        return (kDLFloat, 64, 1)
    elif dtype == np.complex64:
        return (kDLComplex, 64, 1)
    elif dtype == np.complex128:
        return (kDLComplex, 128, 1)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def _device_str_to_dlpack(device: str) -> int:
    """Convert device string to DLPack device type."""
    device_lower = device.lower()
    if device_lower == "cpu":
        return kDLCPU
    elif device_lower == "cuda":
        return kDLCUDA
    elif device_lower == "rocm" or device_lower == "hip":
        return kDLROCM
    else:
        return kDLCPU  # Default to CPU

def _extract_dltensor_from_capsule(capsule) -> Optional[DLTensor]:
    """Extract DLTensor from a DLPack PyCapsule."""
    try:
        # Get the pointer from the capsule
        # The capsule name should be "dltensor" or "dltensor_versioned"
        import ctypes.pythonapi as pyapi
        
        # Check if it's a valid capsule
        if not hasattr(capsule, '__class__') or 'capsule' not in str(type(capsule)).lower():
            return None
        
        # Try to get the pointer using ctypes
        # PyCapsule_GetPointer returns void*
        PyCapsule_GetPointer = pyapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        PyCapsule_GetPointer.restype = ctypes.c_void_p
        
        # Get pointer to DLManagedTensor
        try:
            ptr = PyCapsule_GetPointer(capsule, b"dltensor")
        except:
            try:
                ptr = PyCapsule_GetPointer(capsule, b"dltensor_versioned")
            except:
                return None
        
        if ptr is None:
            return None
        
        # Cast to DLManagedTensor pointer
        managed_tensor = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor)).contents
        
        # Return a copy of the DLTensor (we'll need to copy the shape/strides arrays)
        return managed_tensor.dl_tensor
        
    except Exception:
        # If extraction fails, return None
        return None

class Tensor:
    """
    Tensor wrapper for FKL operations.
    
    Provides a DLPack-compatible tensor interface for GPU operations.
    """
    
    def __init__(self, data: Union[np.ndarray, 'Tensor'], device: str = "cuda"):
        """
        Create a tensor from a numpy array or another tensor.
        
        Args:
            data: NumPy array, Tensor, or DLPack-compatible object
            device: Device type ("cuda", "cpu", "rocm", etc.)
        """
        self._handle = None
        self._lib = None
        self._device = device
        self._managed_tensor = None  # Keep reference to prevent deletion
        self._init_lib()
        
        # Handle different input types
        # Check numpy arrays first (they also have __dlpack__, but we want to handle them specially)
        if isinstance(data, Tensor):
            # Copy from another tensor
            self._data = data._data
            self._device = data._device
            self._create_tensor_from_dlpack()
        elif isinstance(data, np.ndarray):
            # NumPy array - store reference and try DLPack, fallback to manual conversion
            self._data = data
            self._create_tensor_from_dlpack()
        elif hasattr(data, '__dlpack__'):
            # DLPack-compatible object (e.g., PyTorch tensor, CuPy array)
            # Try to keep numpy reference if it's convertible
            if isinstance(data, np.ndarray):
                self._data = data
            else:
                self._data = None  # Will try to extract from DLPack
            self._create_tensor_from_dlpack_object(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _init_lib(self):
        """Initialize the FKL FFI library."""
        import os
        from pathlib import Path
        
        # The library should be in the same directory as this Python file
        lib_name = "_fkl_ffi"
        if os.name == 'nt':  # Windows
            lib_name = f"{lib_name}.dll"
        elif os.name == 'posix':  # Linux/macOS
            lib_name = f"{lib_name}.so" if os.uname().sysname != 'Darwin' else f"{lib_name}.dylib"
        
        # Try multiple paths in order of preference
        possible_paths = []
        
        # First, try to find package directory
        package_dir = None
        if __file__:
            package_dir = Path(__file__).parent
        else:
            # Try to find package directory using importlib
            try:
                import importlib.util
                spec = importlib.util.find_spec("fkl_ffi")
                if spec and spec.origin and spec.origin != "namespace":
                    package_dir = Path(spec.origin).parent
                elif spec and spec.submodule_search_locations:
                    for loc in spec.submodule_search_locations:
                        if Path(loc).exists():
                            package_dir = Path(loc)
                            break
            except:
                pass
        
        # Check build directory (where scikit-build-core puts it during development)
        import sys
        for path_str in sys.path:
            path = Path(path_str)
            if path.exists() and path.is_absolute():
                build_path = path / "build" / "python" / lib_name
                if build_path.exists():
                    possible_paths.append(build_path)
                python_pkg_path = path / "python" / "fkl_ffi" / lib_name
                if python_pkg_path.exists():
                    possible_paths.append(python_pkg_path)
        
        # Check package directory (for installed packages)
        if package_dir:
            possible_paths.append(package_dir / lib_name)
        
        # Check current working directory
        cwd_build = Path.cwd() / "build" / "python" / lib_name
        if cwd_build.exists():
            possible_paths.append(cwd_build)
        possible_paths.append(Path.cwd() / lib_name)
        
        lib_path = None
        for path in possible_paths:
            if path.exists():
                lib_path = path
                break
        
        if lib_path is None:
            # Try system library path
            try:
                self._lib = ctypes.CDLL(lib_name)
            except OSError:
                raise RuntimeError(
                    f"Failed to find FKL FFI library '{lib_name}'. "
                    f"Tried paths: {[str(p) for p in possible_paths]}. "
                    f"Make sure the library is built and installed."
                )
        else:
            self._lib = ctypes.CDLL(str(lib_path))
        
        # Set up function signatures
        self._lib.FKLTensorCreate.argtypes = [
            ctypes.POINTER(DLTensor),  # DLTensor*
            ctypes.POINTER(ctypes.c_void_p)   # FKLTensorHandle*
        ]
        self._lib.FKLTensorCreate.restype = ctypes.c_int
        
        self._lib.FKLTensorDestroy.argtypes = [ctypes.c_void_p]
        self._lib.FKLTensorDestroy.restype = ctypes.c_int
        
        self._lib.FKLTensorGetDLTensor.argtypes = [
            ctypes.c_void_p,  # FKLTensorHandle
            ctypes.POINTER(ctypes.POINTER(DLTensor))  # DLTensor**
        ]
        self._lib.FKLTensorGetDLTensor.restype = ctypes.c_int
    
    def _create_tensor_from_dlpack(self):
        """Create tensor from numpy array using DLPack."""
        if self._data is None:
            raise ValueError("No data provided")
        
        # Get DLPack capsule from numpy array
        try:
            capsule = self._data.__dlpack__()
            self._create_tensor_from_capsule(capsule)
        except (AttributeError, Exception):
            # If DLPack fails, fall back to manual conversion from numpy
            # This handles cases where DLPack extraction doesn't work
            dltensor = self._create_dltensor_from_numpy(self._data)
            self._create_tensor_from_dltensor(dltensor)
    
    def _create_tensor_from_dlpack_object(self, obj):
        """Create tensor from any DLPack-compatible object."""
        try:
            capsule = obj.__dlpack__()
        except AttributeError:
            raise ValueError("Object does not support DLPack protocol") from None
        
        self._create_tensor_from_capsule(capsule)
    
    def _create_tensor_from_capsule(self, capsule):
        """Create FKL tensor from DLPack capsule."""
        # Extract DLTensor from capsule
        dltensor = _extract_dltensor_from_capsule(capsule)
        
        if dltensor is None:
            # Fallback: create DLTensor manually from numpy array
            if self._data is None:
                raise ValueError("Cannot create tensor: DLPack extraction failed and no numpy array available")
            dltensor = self._create_dltensor_from_numpy(self._data)
        else:
            # Store reference to prevent deletion
            self._managed_tensor = capsule
        
        # Allocate memory for shape and strides if needed
        if dltensor.ndim > 0:
            # Create copies of shape and strides arrays
            shape_array = (ctypes.c_int64 * dltensor.ndim)(*[dltensor.shape[i] for i in range(dltensor.ndim)])
            strides_array = None
            if dltensor.strides:
                strides_array = (ctypes.c_int64 * dltensor.ndim)(*[dltensor.strides[i] for i in range(dltensor.ndim)])
            
            # Create a new DLTensor with copied arrays
            new_dltensor = DLTensor(
                data=dltensor.data,
                device=dltensor.device,
                ndim=dltensor.ndim,
                dtype=dltensor.dtype,
                shape=ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64)),
                strides=ctypes.cast(strides_array, ctypes.POINTER(ctypes.c_int64)) if strides_array else None,
                byte_offset=dltensor.byte_offset
            )
        else:
            new_dltensor = dltensor
        
        # Create FKL tensor
        handle_ptr = ctypes.c_void_p()
        ret = self._lib.FKLTensorCreate(
            ctypes.byref(new_dltensor),
            ctypes.byref(handle_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to create FKL tensor")
        self._handle = handle_ptr.value
    
    def _create_tensor_from_dltensor(self, dltensor: DLTensor):
        """Create FKL tensor directly from a DLTensor structure."""
        # Allocate memory for shape and strides if needed
        if dltensor.ndim > 0:
            # Create copies of shape and strides arrays
            shape_array = (ctypes.c_int64 * dltensor.ndim)(*[dltensor.shape[i] for i in range(dltensor.ndim)])
            strides_array = None
            if dltensor.strides:
                strides_array = (ctypes.c_int64 * dltensor.ndim)(*[dltensor.strides[i] for i in range(dltensor.ndim)])
            
            # Create a new DLTensor with copied arrays
            new_dltensor = DLTensor(
                data=dltensor.data,
                device=dltensor.device,
                ndim=dltensor.ndim,
                dtype=dltensor.dtype,
                shape=ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64)),
                strides=ctypes.cast(strides_array, ctypes.POINTER(ctypes.c_int64)) if strides_array else None,
                byte_offset=dltensor.byte_offset
            )
        else:
            new_dltensor = dltensor
        
        # Create FKL tensor
        handle_ptr = ctypes.c_void_p()
        ret = self._lib.FKLTensorCreate(
            ctypes.byref(new_dltensor),
            ctypes.byref(handle_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to create FKL tensor")
        self._handle = handle_ptr.value
    
    def _create_dltensor_from_numpy(self, arr: np.ndarray) -> DLTensor:
        """Create DLTensor structure from numpy array."""
        # Convert dtype
        dtype_code, dtype_bits, dtype_lanes = _numpy_dtype_to_dlpack(arr.dtype)
        dtype = DLDataType(code=dtype_code, bits=dtype_bits, lanes=dtype_lanes)
        
        # Create device
        device_type = _device_str_to_dlpack(self._device)
        device = DLDevice(device_type=device_type, device_id=0)
        
        # Get shape and strides
        ndim = arr.ndim
        shape_array = (ctypes.c_int64 * ndim)(*arr.shape)
        strides_array = (ctypes.c_int64 * ndim)(*[s // arr.itemsize for s in arr.strides])
        
        # Get data pointer
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        
        # Create DLTensor
        dltensor = DLTensor(
            data=data_ptr,
            device=device,
            ndim=ndim,
            dtype=dtype,
            shape=ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64)),
            strides=ctypes.cast(strides_array, ctypes.POINTER(ctypes.c_int64)),
            byte_offset=0
        )
        
        return dltensor
    
    def __dlpack__(self, stream=None):  # pylint: disable=unused-argument
        """
        Export tensor as DLPack capsule.
        
        Args:
            stream: Optional stream for synchronization (not used for CPU)
        
        Returns:
            PyCapsule containing DLManagedTensor
        """
        if self._handle is None:
            raise RuntimeError("Tensor is not initialized")
        
        # Get DLTensor from FKL tensor
        dltensor_ptr = ctypes.POINTER(DLTensor)()
        ret = self._lib.FKLTensorGetDLTensor(
            ctypes.c_void_p(self._handle),
            ctypes.byref(dltensor_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to get DLTensor from FKL tensor")
        
        dltensor = dltensor_ptr.contents
        
        # Create DLManagedTensor
        # We need to keep a reference to prevent deletion
        managed_tensor = DLManagedTensor()
        managed_tensor.dl_tensor = dltensor
        managed_tensor.manager_ctx = ctypes.cast(ctypes.py_object(self), ctypes.c_void_p)
        
        # Define deleter function
        def deleter(_managed_ptr):
            # The tensor will be cleaned up when Python object is deleted
            # We don't need to do anything here as Python's GC will handle it
            pass
        
        managed_tensor.deleter = DLPACK_DELETER(deleter)
        
        # Create capsule
        import ctypes.pythonapi as pyapi
        PyCapsule_New = pyapi.PyCapsule_New
        PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        PyCapsule_New.restype = ctypes.py_object
        
        capsule = PyCapsule_New(
            ctypes.addressof(managed_tensor),
            b"dltensor",
            None
        )
        
        # Increment reference to self to keep it alive
        ctypes.pythonapi.Py_INCREF(ctypes.py_object(self))
        
        return capsule
    
    def __dlpack_device__(self):
        """
        Return device information for DLPack.
        
        Returns:
            Tuple of (device_type, device_id)
        """
        if self._handle is None:
            # Return device from string
            return (_device_str_to_dlpack(self._device), 0)
        
        # Get DLTensor to read device info
        dltensor_ptr = ctypes.POINTER(DLTensor)()
        ret = self._lib.FKLTensorGetDLTensor(
            ctypes.c_void_p(self._handle),
            ctypes.byref(dltensor_ptr)
        )
        if ret != 0:
            return (_device_str_to_dlpack(self._device), 0)
        
        dltensor = dltensor_ptr.contents
        return (dltensor.device.device_type, dltensor.device.device_id)
    
    def to_numpy(self) -> np.ndarray:
        """Convert tensor back to numpy array."""
        if self._data is not None:
            # If we have the original numpy array, return it
            return self._data
        
        if self._handle is None:
            raise RuntimeError("Tensor is not initialized")
        
        # Get DLTensor
        dltensor_ptr = ctypes.POINTER(DLTensor)()
        ret = self._lib.FKLTensorGetDLTensor(
            ctypes.c_void_p(self._handle),
            ctypes.byref(dltensor_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to get DLTensor from FKL tensor")
        
        dltensor = dltensor_ptr.contents
        
        # Convert DLTensor to numpy array
        # This is a simplified version - full implementation would handle
        # different devices, dtypes, and memory layouts
        if dltensor.device.device_type == kDLCPU:
            # For CPU, we can directly create numpy array from data pointer
            # Note: This assumes the data is still valid and accessible
            # For GPU tensors, this won't work directly
            raise NotImplementedError("to_numpy() for GPU tensors requires memory transfer - not yet implemented")
        else:
            raise NotImplementedError(f"to_numpy() for device type {dltensor.device.device_type} not yet implemented")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the tensor shape."""
        if self._data is not None:
            return self._data.shape
        
        if self._handle is None:
            raise RuntimeError("Tensor is not initialized")
        
        # Get DLTensor
        dltensor_ptr = ctypes.POINTER(DLTensor)()
        ret = self._lib.FKLTensorGetDLTensor(
            ctypes.c_void_p(self._handle),
            ctypes.byref(dltensor_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to get DLTensor from FKL tensor")
        
        dltensor = dltensor_ptr.contents
        return tuple(dltensor.shape[i] for i in range(dltensor.ndim))
    
    @property
    def dtype(self) -> np.dtype:
        """Get the tensor data type."""
        if self._data is not None:
            return self._data.dtype
        
        if self._handle is None:
            raise RuntimeError("Tensor is not initialized")
        
        # Get DLTensor
        dltensor_ptr = ctypes.POINTER(DLTensor)()
        ret = self._lib.FKLTensorGetDLTensor(
            ctypes.c_void_p(self._handle),
            ctypes.byref(dltensor_ptr)
        )
        if ret != 0:
            raise RuntimeError("Failed to get DLTensor from FKL tensor")
        
        dltensor = dltensor_ptr.contents
        
        # Map DLPack dtype to numpy dtype
        dtype_map = {
            (kDLFloat, 32): np.float32,
            (kDLFloat, 64): np.float64,
            (kDLInt, 32): np.int32,
            (kDLInt, 64): np.int64,
            (kDLUInt, 32): np.uint32,
            (kDLUInt, 64): np.uint64,
            (kDLBool, 8): np.bool_,
        }
        return dtype_map.get((dltensor.dtype.code, dltensor.dtype.bits), np.float32)
    
    @property
    def handle(self):
        """Get the internal tensor handle (for advanced usage)."""
        return self._handle
    
    def __del__(self):
        """Destroy the tensor when the object is garbage collected."""
        if self._handle is not None and self._lib is not None:
            try:
                self._lib.FKLTensorDestroy(ctypes.c_void_p(self._handle))
            except Exception:
                pass  # Ignore errors during cleanup


def from_dlpack(obj) -> Tensor:
    """
    Create a Tensor from any DLPack-compatible object.
    
    This function implements the Python Array API standard's from_dlpack().
    
    Args:
        obj: Any object that implements __dlpack__() method
    
    Returns:
        Tensor instance
    """
    return Tensor(obj)
