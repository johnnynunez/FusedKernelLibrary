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
#include <tvm/ffi/c_api.h>
#include <memory>
#include <stdexcept>

// Note: TVM-FFI C++ headers (function.h, module.h, etc.) are not currently used
// and may not be available in all TVM-FFI installations. Using only the C API.

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
// Note: Function implementations are in fkl_stream.cpp, fkl_executor.cpp, and fkl_jit.cpp
// This file only contains the registration function

