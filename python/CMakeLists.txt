# Python bindings and TVM-FFI integration for FusedKernelLibrary
# Note: This is included as a subdirectory, so cmake_minimum_required is not needed
# Note: scikit-build-core handles Python integration, so we don't need to find Python here

option(BUILD_PYTHON_BINDINGS "Build Python bindings using TVM-FFI" ON)
option(ENABLE_JIT "Enable JIT compilation support" ON)

# Find TVM-FFI
# Method 1: Try CMake CONFIG mode (recommended - uses pre-built package)
find_package(tvm_ffi CONFIG QUIET)
if(tvm_ffi_FOUND)
    message(STATUS "Found TVM-FFI via CMake CONFIG (pre-built package)")
    set(TVM_FFI_FOUND TRUE)
endif()

# Method 2: Try to build from source if CONFIG mode fails
if(NOT TVM_FFI_FOUND)
    find_program(PYTHON_EXECUTABLE python3 python)
    if(PYTHON_EXECUTABLE)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -m tvm_ffi.config --sourcedir
            OUTPUT_VARIABLE TVM_FFI_SOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TVM_FFI_SOURCE_DIR AND EXISTS ${TVM_FFI_SOURCE_DIR})
            message(STATUS "Found TVM-FFI source at ${TVM_FFI_SOURCE_DIR}, building from source")
            add_subdirectory(${TVM_FFI_SOURCE_DIR} tvm_ffi)
            set(TVM_FFI_FOUND TRUE)
        endif()
    endif()
endif()

# Method 3: Fallback to pkg-config
if(NOT TVM_FFI_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TVM_FFI QUIET tvm-ffi)
        if(TVM_FFI_FOUND)
            message(STATUS "Found TVM-FFI via pkg-config")
        endif()
    endif()
endif()

if(BUILD_PYTHON_BINDINGS AND NOT TVM_FFI_FOUND)
    message(WARNING "TVM-FFI not found. Python bindings will not be built.")
    message(STATUS "Install TVM-FFI: pip install apache-tvm-ffi")
    message(STATUS "Note: TVM-FFI may still work at runtime if installed as a Python package")
    # Don't disable BUILD_PYTHON_BINDINGS - TVM-FFI might be available at runtime
    # set(BUILD_PYTHON_BINDINGS OFF)
endif()

if(BUILD_PYTHON_BINDINGS)
    # Add TVM-FFI C API wrapper library
    add_library(fkl_ffi SHARED
        src/fkl_ffi_api.cpp
        src/fkl_stream.cpp
        src/fkl_executor.cpp
        src/fkl_module_init.cpp
    )

    target_include_directories(fkl_ffi
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # Link against FKL
    target_link_libraries(fkl_ffi
        PUBLIC
            FKL::FKL
    )

    # Link TVM-FFI
    if(tvm_ffi_FOUND)
        # Found via CMake CONFIG mode (pre-built package)
        target_link_libraries(fkl_ffi PUBLIC tvm_ffi_shared)
        message(STATUS "Linking against tvm_ffi_shared (pre-built)")
    elseif(TVM_FFI_FOUND)
        # Found via source build or pkg-config
        if(TARGET tvm_ffi_shared)
            # Built from source
            target_link_libraries(fkl_ffi PUBLIC tvm_ffi_shared)
            message(STATUS "Linking against tvm_ffi_shared (from source)")
        elseif(TVM_FFI_LIBRARIES)
            # Found via pkg-config
            target_include_directories(fkl_ffi PUBLIC ${TVM_FFI_INCLUDE_DIRS})
            target_link_libraries(fkl_ffi PUBLIC ${TVM_FFI_LIBRARIES})
            target_compile_options(fkl_ffi PUBLIC ${TVM_FFI_CFLAGS_OTHER})
            message(STATUS "Linking against TVM-FFI (via pkg-config)")
        endif()
    else()
        # TVM-FFI not found at build time, but might be available at runtime
        # The Python bindings will try to load it dynamically
        message(STATUS "TVM-FFI not found at build time - will be loaded at runtime")
    endif()

    # CUDA support
    if(ENABLE_CUDA)
        target_compile_options(fkl_ffi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)
        if(ENABLE_JIT)
            target_compile_definitions(fkl_ffi PUBLIC FKL_ENABLE_JIT)
            target_sources(fkl_ffi PRIVATE src/fkl_jit.cpp)
            # Find NVRTC for JIT
            find_library(CUDART_LIBRARY cudart)
            find_library(NVRTC_LIBRARY nvrtc)
            if(CUDART_LIBRARY AND NVRTC_LIBRARY)
                target_link_libraries(fkl_ffi PRIVATE ${CUDART_LIBRARY} ${NVRTC_LIBRARY})
            endif()
        endif()
    endif()

    # HIP/ROCm support
    if(DEFINED ENABLE_HIP AND ENABLE_HIP)
        target_compile_options(fkl_ffi PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fPIC>)
        target_sources(fkl_ffi PRIVATE src/fkl_stream_hip.cpp)
        # Link HIP libraries
        if(DEFINED ROCM_HIP_LIBRARY AND ROCM_HIP_LIBRARY)
            target_link_libraries(fkl_ffi PRIVATE ${ROCM_HIP_LIBRARY})
        endif()
        # Add HIP include directories
        if(DEFINED ROCM_INCLUDE_DIR AND ROCM_INCLUDE_DIR)
            target_include_directories(fkl_ffi PRIVATE ${ROCM_INCLUDE_DIR})
        endif()
        target_compile_definitions(fkl_ffi PUBLIC FKL_ENABLE_HIP)
    endif()

    # Set output name
    set_target_properties(fkl_ffi PROPERTIES
        OUTPUT_NAME "_fkl_ffi"
        PREFIX ""
    )

    # Install the library
    install(TARGETS fkl_ffi
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Install headers
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fkl_ffi
        FILES_MATCHING PATTERN "*.h"
    )
endif()
