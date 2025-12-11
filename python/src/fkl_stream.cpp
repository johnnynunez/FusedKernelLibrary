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
#include <fused_kernel/core/execution_model/stream.h>
#include <memory>
#include <cstring>

#if defined(__HIP__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

using namespace fk;

struct FKLStream {
    std::shared_ptr<fk::Stream> stream;
    
    FKLStream() : stream(std::make_shared<fk::Stream>()) {}
    
    explicit FKLStream(void* cuda_stream) {
        #if defined(__NVCC__) || defined(__CUDACC__)
        stream = std::make_shared<fk::Stream>(reinterpret_cast<cudaStream_t>(cuda_stream));
        #elif defined(__HIP__) || defined(__HIPCC__)
        // HIP streams are compatible with CUDA streams (same type)
        stream = std::make_shared<fk::Stream>(reinterpret_cast<hipStream_t>(cuda_stream));
        #else
        stream = std::make_shared<fk::Stream>();
        #endif
    }
    
    explicit FKLStream(void* hip_stream, bool is_hip) {
        #if defined(__HIP__) || defined(__HIPCC__)
        if (is_hip) {
            stream = std::make_shared<fk::Stream>(reinterpret_cast<hipStream_t>(hip_stream));
        } else {
            stream = std::make_shared<fk::Stream>();
        }
        #else
        stream = std::make_shared<fk::Stream>();
        #endif
    }
};

int FKLStreamCreate(FKLStreamHandle* out) {
    if (out == nullptr) {
        return -1;
    }
    try {
        *out = new FKLStream();
        return 0;
    } catch (...) {
        return -1;
    }
}

int FKLStreamDestroy(FKLStreamHandle stream) {
    if (stream == nullptr) {
        return -1;
    }
    try {
        delete stream;
        return 0;
    } catch (...) {
        return -1;
    }
}

int FKLStreamSync(FKLStreamHandle stream) {
    if (stream == nullptr || stream->stream == nullptr) {
        return -1;
    }
    try {
        stream->stream->sync();
        return 0;
    } catch (...) {
        return -1;
    }
}

int FKLStreamFromCUDAStream(FKLStreamHandle* out, void* cuda_stream) {
    if (out == nullptr) {
        return -1;
    }
    try {
        *out = new FKLStream(cuda_stream);
        return 0;
    } catch (...) {
        return -1;
    }
}

int FKLStreamFromHIPStream(FKLStreamHandle* out, void* hip_stream) {
    if (out == nullptr) {
        return -1;
    }
    try {
        #if defined(__HIP__) || defined(__HIPCC__)
        *out = new FKLStream(hip_stream, true);
        #else
        // HIP not available, create default stream
        *out = new FKLStream();
        #endif
        return 0;
    } catch (...) {
        return -1;
    }
}

// Helper function to get the underlying stream (for internal use)
Stream* FKLStreamGetStream(FKLStreamHandle handle) {
    if (handle == nullptr || handle->stream == nullptr) {
        return nullptr;
    }
    return handle->stream.get();
}

