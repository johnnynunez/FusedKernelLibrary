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

// JIT compilation is CUDA-specific (uses nvcc)
// Skip CUDA-specific code when building for HIP or when CUDA is not available

extern "C" {

#if defined(__HIP__) || defined(__HIPCC__) || defined(FKL_ENABLE_HIP)
// HIP build - JIT is not supported (JIT uses nvcc which is CUDA-specific)
// Provide stub implementations that return errors
int FKLJITCompileKernel(
    const char* kernel_code,
    const char* kernel_name,
    const char* options,
    void** cubin_out,
    size_t* cubin_size_out
) {
    (void)kernel_code; (void)kernel_name; (void)options;
    (void)cubin_out; (void)cubin_size_out;
    return -1; // JIT not supported for HIP
}

int FKLJITLoadModule(
    const char* module_path,
    TVMFFIObjectHandle* module_out
) {
    (void)module_path; (void)module_out;
    return -1; // JIT not supported for HIP
}
#elif defined(__CUDACC__) || defined(__NVCC__) || defined(__CUDA__)
// CUDA compiler detected - include CUDA headers and provide full implementation
// Only include when actually using a CUDA compiler to avoid errors when CUDA is not available
#include <cuda_runtime.h>

#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

// JIT compilation using nvcc (not NVRTC) because FKL operations
// have both HOST code (build functions) and DEVICE code (exec functions)
// in the same struct. NVRTC can only compile device code.

static std::string get_temp_directory() {
#ifdef _WIN32
    char temp_path[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    return std::string(temp_path);
#else
    const char* tmpdir = getenv("TMPDIR");
    if (tmpdir) return std::string(tmpdir);
    return "/tmp";
#endif
}

static std::string create_temp_file(const std::string& prefix, const std::string& suffix) {
    std::string tmpdir = get_temp_directory();
    std::string filename = tmpdir + "/" + prefix;
    
#ifdef _WIN32
    char unique_name[MAX_PATH];
    GetTempFileNameA(tmpdir.c_str(), prefix.c_str(), 0, unique_name);
    return std::string(unique_name) + suffix;
#else
    char template_str[256];
    snprintf(template_str, sizeof(template_str), "%s/%s_XXXXXX%s", 
             tmpdir.c_str(), prefix.c_str(), suffix.c_str());
    int fd = mkstemps(template_str, suffix.length());
    if (fd == -1) return "";
    close(fd);
    return std::string(template_str);
#endif
}

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
    
    // Generate temporary file names
    std::string cu_file = create_temp_file("fkl_jit_", ".cu");
    std::string so_file = create_temp_file("fkl_jit_", ".so");
    
    if (cu_file.empty() || so_file.empty()) {
        return -1;
    }
    
    // Write kernel code to .cu file
    // The kernel code must include BOTH host and device code
    // because FKL operations have build() (host) and exec() (device) in same struct
    std::ofstream out(cu_file);
    if (!out.is_open()) {
        return -1;
    }
    
    // Write includes needed for FKL
    out << "#include <cuda_runtime.h>\n";
    out << "#include <device_launch_parameters.h>\n";
    out << "// FKL JIT compiled kernel\n";
    out << "// This includes both HOST code (build functions) and DEVICE code (exec functions)\n\n";
    
    // Write the kernel code
    out << kernel_code;
    out.close();
    
    // Build command: use nvcc (not NVRTC) because we need to compile host+device code
    std::ostringstream cmd;
    cmd << "nvcc ";
    
    // Add options
    if (options != nullptr && strlen(options) > 0) {
        cmd << options << " ";
    } else {
        // Default options
        cmd << "-arch=sm_75 ";  // Default compute capability
    }
    
    // Compile to shared library
    cmd << "--shared ";
    cmd << "-Xcompiler -fPIC ";
    cmd << "-o " << so_file << " ";
    cmd << cu_file;
    
    // Execute nvcc
    int ret = system(cmd.str().c_str());
    if (ret != 0) {
        // Cleanup
        remove(cu_file.c_str());
        return -1;
    }
    
    // Read compiled .so file
    std::ifstream in(so_file, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        remove(cu_file.c_str());
        remove(so_file.c_str());
        return -1;
    }
    
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    *cubin_out = malloc(file_size);
    if (*cubin_out == nullptr) {
        in.close();
        remove(cu_file.c_str());
        remove(so_file.c_str());
        return -1;
    }
    
    in.read(static_cast<char*>(*cubin_out), file_size);
    in.close();
    *cubin_size_out = file_size;
    
    // Cleanup temp files (optional - could keep for debugging)
    remove(cu_file.c_str());
    // Keep .so file for now - might be needed for loading
    
    return 0;
}

int FKLJITLoadModule(
    const char* module_path,
    TVMFFIObjectHandle* module_out
) {
    // Load a pre-compiled module (.so file compiled with nvcc)
    if (module_path == nullptr || module_out == nullptr) {
        return -1;
    }
    
    // Use TVM-FFI Module::LoadFromFile or dlopen
    // For now, placeholder
    return -1;
}

#else
// CUDA not available (neither HIP nor CUDA compiler detected)
// Provide stub implementations that return errors
int FKLJITCompileKernel(
    const char* kernel_code,
    const char* kernel_name,
    const char* options,
    void** cubin_out,
    size_t* cubin_size_out
) {
    (void)kernel_code; (void)kernel_name; (void)options;
    (void)cubin_out; (void)cubin_size_out;
    return -1; // JIT requires CUDA
}

int FKLJITLoadModule(
    const char* module_path,
    TVMFFIObjectHandle* module_out
) {
    (void)module_path; (void)module_out;
    return -1; // JIT requires CUDA
}

#endif // CUDA compiler check

} // extern "C"

#endif // FKL_ENABLE_JIT

