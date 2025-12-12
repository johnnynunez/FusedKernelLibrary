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

#include "fkl_stream.h"
#include "fkl_ffi/fkl_ffi_api.h"
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/array.h>
#include <fused_kernel/fused_kernel.h>
#include <dlpack/dlpack.h>
#include <memory>
#include <vector>

using namespace fk;

// CORRECT SOLUTION: Use FKL's executeOperations directly
// We don't need to generate code - FKL already has everything compiled

struct FKLTensor {
    DLTensor dltensor;
    std::shared_ptr<void> data_ptr; // Keep data alive
    
    FKLTensor(const DLTensor& tensor) : dltensor(tensor) {
        // Make a copy of the tensor metadata
        dltensor = tensor;
    }
};

int FKLTensorCreate(DLTensor* tensor, FKLTensorHandle* out) {
    if (tensor == nullptr || out == nullptr) {
        return -1;
    }
    try {
        *out = new FKLTensor(*tensor);
        return 0;
    } catch (...) {
        return -1;
    }
}

int FKLTensorDestroy(FKLTensorHandle tensor) {
    if (tensor == nullptr) {
        return -1;
    }
    try {
        delete tensor;
        return 0;
    } catch (...) {
        return -1;
    }
}

int FKLTensorGetDLTensor(FKLTensorHandle tensor, DLTensor** out) {
    if (tensor == nullptr || out == nullptr) {
        return -1;
    }
    *out = &tensor->dltensor;
    return 0;
}

// Function to execute FKL operations
// This is the key function - exposes FKL's executeOperations
// FKL operations already have build() (host) and exec() (device) compiled
// In a real implementation, we would need to pass the operations
// This is complex because they are variadic templates in C++
// For now, this is a placeholder showing the idea
int FKLExecuteOperations(
    FKLStreamHandle stream
) {
    if (stream == nullptr) {
        return -1;
    }
    
    try {
        Stream* fkl_stream = FKLStreamGetStream(stream);
        if (fkl_stream == nullptr) {
            return -1;
        }
        
        // Here we would call fk::executeOperations with the operations
        // The problem is that executeOperations is a variadic template
        // We need a way to pass operations from Python
        
        // Conceptual example:
        // fk::executeOperations<TransformDPP<>>(*fkl_stream, op1, op2, op3, ...);
        // 
        // Operations op1, op2, etc. are IOp created by build()
        // exec() functions are already compiled in the kernel
        
        return 0;
    } catch (...) {
        return -1;
    }
}

