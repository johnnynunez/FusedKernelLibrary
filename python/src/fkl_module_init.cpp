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
#include <tvm/ffi/c_api.h>

// Module initialization function
// Called when Python loads the module
extern "C" {

#ifdef _WIN32
#define FKL_FFI_EXPORT __declspec(dllexport)
#else
#define FKL_FFI_EXPORT __attribute__((visibility("default")))
#endif

FKL_FFI_EXPORT int FKLFFIInit() {
    // Initialize TVM-FFI if needed
    // Register global functions
    return FKLFFIRegisterGlobalFunctions();
}

} // extern "C"

