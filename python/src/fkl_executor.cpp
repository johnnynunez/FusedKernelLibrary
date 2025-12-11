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
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/array.h>
#include <dlpack/dlpack.h>
#include <memory>
#include <vector>

using namespace fk;

// Note: This is a simplified executor wrapper.
// Full implementation would need to handle the variadic template operations
// which is complex in C. For now, this provides the basic structure.

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

