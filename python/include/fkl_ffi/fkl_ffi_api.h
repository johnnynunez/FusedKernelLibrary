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

#ifndef FKL_FFI_API_H
#define FKL_FFI_API_H

#include <tvm/ffi/c_api.h>
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct FKLStream* FKLStreamHandle;
typedef struct FKLTensor* FKLTensorHandle;

// Stream operations
TVM_DLL int FKLStreamCreate(FKLStreamHandle* out);
TVM_DLL int FKLStreamDestroy(FKLStreamHandle stream);
TVM_DLL int FKLStreamSync(FKLStreamHandle stream);
TVM_DLL int FKLStreamFromCUDAStream(FKLStreamHandle* out, void* cuda_stream);
TVM_DLL int FKLStreamFromHIPStream(FKLStreamHandle* out, void* hip_stream);

// Tensor operations
TVM_DLL int FKLTensorCreate(DLTensor* tensor, FKLTensorHandle* out);
TVM_DLL int FKLTensorDestroy(FKLTensorHandle tensor);
TVM_DLL int FKLTensorGetDLTensor(FKLTensorHandle tensor, DLTensor** out);

// Execute operations using FKL's existing system
// This uses the build() functions (HOST) and exec() functions (DEVICE) 
// that are already compiled. No code generation needed!
// 
// Note: This is a placeholder - full implementation would need to handle
// variadic template operations from Python, which is complex.
// The real solution is to use FKL's executeOperations directly.
TVM_DLL int FKLExecuteOperations(
    FKLStreamHandle stream
    // In real implementation: variadic operations
    // This is complex because executeOperations is a template variadic function
);

// JIT compilation (if enabled) - ONLY if you really need dynamic code generation
// Otherwise, use FKLExecuteOperations which uses the existing compiled system
#ifdef FKL_ENABLE_JIT
TVM_DLL int FKLJITCompileKernel(
    const char* kernel_code,
    const char* kernel_name,
    const char* options,
    void** cubin_out,
    size_t* cubin_size_out
);
TVM_DLL int FKLJITLoadModule(
    const char* module_path,
    TVMFFIObjectHandle* module_out
);
#endif

// Register global functions for TVM-FFI
TVM_DLL int FKLFFIRegisterGlobalFunctions();

#ifdef __cplusplus
}
#endif

#endif // FKL_FFI_API_H

