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

#ifdef FKL_ENABLE_JIT

#include "fkl_ffi/fkl_ffi_api.h"
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      return -1;                                                  \
    }                                                             \
  } while(0)

int FKLJITCompileKernel(
    const char* kernel_code,
    const char* kernel_name,
    const char* options,
    void** cubin_out,
    size_t* cubin_size_out
) {
    if (kernel_code == nullptr || kernel_name == nullptr || 
        cubin_out == nullptr || cubin_size_out == nullptr) {
        return -1;
    }
    
    nvrtcProgram prog;
    
    // Create the program
    std::vector<const char*> opts;
    std::string opt_str;
    
    if (options != nullptr && strlen(options) > 0) {
        opt_str = options;
    } else {
        // Default options
        opt_str = "-arch=sm_75"; // Default to compute capability 7.5
    }
    
    // Parse options (simple space-separated)
    std::istringstream iss(opt_str);
    std::string opt;
    while (iss >> opt) {
        opts.push_back(opt.c_str());
    }
    
    NVRTC_SAFE_CALL(nvrtcCreateProgram(
        &prog,
        kernel_code,
        kernel_name,
        0,
        nullptr,
        nullptr
    ));
    
    // Compile the program
    nvrtcResult compileResult = nvrtcCompileProgram(prog, opts.size(), opts.data());
    
    // Get compilation log
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        // Log could be used for error reporting
    }
    
    if (compileResult != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        return -1;
    }
    
    // Get PTX
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
    
    // For now, we return PTX. In a full implementation, you might want to
    // compile PTX to CUBIN using cuModuleLoadDataEx
    *cubin_out = malloc(ptxSize);
    if (*cubin_out == nullptr) {
        nvrtcDestroyProgram(&prog);
        return -1;
    }
    memcpy(*cubin_out, ptx.data(), ptxSize);
    *cubin_size_out = ptxSize;
    
    nvrtcDestroyProgram(&prog);
    return 0;
}

int FKLJITLoadModule(
    const char* module_path,
    TVMFFIObjectHandle* module_out
) {
    // This would load a pre-compiled module
    // Implementation depends on TVM-FFI module loading API
    if (module_path == nullptr || module_out == nullptr) {
        return -1;
    }
    
    // Placeholder - would use TVM-FFI Module::LoadFromFile
    return -1;
}

#endif // FKL_ENABLE_JIT

