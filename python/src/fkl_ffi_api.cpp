/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "fkl_ffi/fkl_ffi_api.h"
#include "fkl_stream.h"
#include "fkl_executor.h"
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/module.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/container/tensor.h>
#include <memory>
#include <stdexcept>

using namespace tvm::ffi;

// Helper to convert FKLStreamHandle to TVM object
static ObjectPtr<Object> StreamToObject(FKLStreamHandle stream) {
    // Store stream handle in an opaque object
    return ObjectPtr<Object>(reinterpret_cast<Object*>(stream));
}

static FKLStreamHandle ObjectToStream(ObjectPtr<Object> obj) {
    return reinterpret_cast<FKLStreamHandle>(obj.get());
}

// Register global functions for TVM-FFI using stable C ABI
int FKLFFIRegisterGlobalFunctions() {
    try {
        // Note: The actual registration would use TVM-FFI's C API
        // For now, we provide the C functions that can be called directly
        // Python bindings will use ctypes to call these functions
        
        // The registration happens when the module is loaded
        // This function can be called from Python to ensure functions are registered
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// Module initialization - called when library is loaded
extern "C" {
    // Export functions for Python ctypes
    #ifdef _WIN32
    #define FKL_FFI_EXPORT __declspec(dllexport)
    #else
    #define FKL_FFI_EXPORT __attribute__((visibility("default")))
    #endif
    
    FKL_FFI_EXPORT int FKLStreamCreate(FKLStreamHandle* out) {
        return ::FKLStreamCreate(out);
    }
    
    FKL_FFI_EXPORT int FKLStreamDestroy(FKLStreamHandle stream) {
        return ::FKLStreamDestroy(stream);
    }
    
    FKL_FFI_EXPORT int FKLStreamSync(FKLStreamHandle stream) {
        return ::FKLStreamSync(stream);
    }
    
    FKL_FFI_EXPORT int FKLStreamFromCUDAStream(FKLStreamHandle* out, void* cuda_stream) {
        return ::FKLStreamFromCUDAStream(out, cuda_stream);
    }
    
    FKL_FFI_EXPORT int FKLTensorCreate(DLTensor* tensor, FKLTensorHandle* out) {
        return ::FKLTensorCreate(tensor, out);
    }
    
    FKL_FFI_EXPORT int FKLTensorDestroy(FKLTensorHandle tensor) {
        return ::FKLTensorDestroy(tensor);
    }
    
    FKL_FFI_EXPORT int FKLTensorGetDLTensor(FKLTensorHandle tensor, DLTensor** out) {
        return ::FKLTensorGetDLTensor(tensor, out);
    }
    
    #ifdef FKL_ENABLE_JIT
    FKL_FFI_EXPORT int FKLJITCompileKernel(
        const char* kernel_code,
        const char* kernel_name,
        const char* options,
        void** cubin_out,
        size_t* cubin_size_out
    ) {
        return ::FKLJITCompileKernel(kernel_code, kernel_name, options, cubin_out, cubin_size_out);
    }
    
    FKL_FFI_EXPORT int FKLJITLoadModule(
        const char* module_path,
        TVMFFIObjectHandle* module_out
    ) {
        return ::FKLJITLoadModule(module_path, module_out);
    }
    #endif
}

