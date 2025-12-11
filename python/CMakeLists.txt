# Python bindings and TVM-FFI integration for FusedKernelLibrary

cmake_minimum_required(VERSION 3.22)

# Find TVM-FFI
find_package(tvm-ffi QUIET)
if(NOT tvm-ffi_FOUND)
    # Try to find via pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TVM_FFI QUIET tvm-ffi)
    endif()
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

option(BUILD_PYTHON_BINDINGS "Build Python bindings using TVM-FFI" ON)
option(ENABLE_JIT "Enable JIT compilation support" ON)

if(BUILD_PYTHON_BINDINGS)
    if(NOT tvm-ffi_FOUND AND NOT TVM_FFI_FOUND)
        message(WARNING "TVM-FFI not found. Python bindings will not be built.")
        message(STATUS "Install TVM-FFI: pip install tvm-ffi")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
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
    if(tvm-ffi_FOUND)
        target_link_libraries(fkl_ffi PUBLIC tvm::ffi::c_api)
        target_include_directories(fkl_ffi PUBLIC ${tvm-ffi_INCLUDE_DIRS})
    elseif(TVM_FFI_FOUND)
        target_link_libraries(fkl_ffi PUBLIC ${TVM_FFI_LIBRARIES})
        target_include_directories(fkl_ffi PUBLIC ${TVM_FFI_INCLUDE_DIRS})
        target_compile_options(fkl_ffi PUBLIC ${TVM_FFI_CFLAGS_OTHER})
    else
        # Try to use tvm-ffi-config if available
        find_program(TVM_FFI_CONFIG tvm-ffi-config)
        if(TVM_FFI_CONFIG)
            execute_process(
                COMMAND ${TVM_FFI_CONFIG} --cflags
                OUTPUT_VARIABLE TVM_FFI_CFLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            execute_process(
                COMMAND ${TVM_FFI_CONFIG} --ldflags
                OUTPUT_VARIABLE TVM_FFI_LDFLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            execute_process(
                COMMAND ${TVM_FFI_CONFIG} --libs
                OUTPUT_VARIABLE TVM_FFI_LIBS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            target_compile_options(fkl_ffi PUBLIC ${TVM_FFI_CFLAGS})
            target_link_options(fkl_ffi PUBLIC ${TVM_FFI_LDFLAGS})
            target_link_libraries(fkl_ffi PUBLIC ${TVM_FFI_LIBS})
        endif()
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
    if(ENABLE_HIP)
        target_compile_options(fkl_ffi PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fPIC>)
        target_sources(fkl_ffi PRIVATE src/fkl_stream_hip.cpp)
        # Link HIP libraries
        if(ROCM_HIP_LIBRARY)
            target_link_libraries(fkl_ffi PRIVATE ${ROCM_HIP_LIBRARY})
        endif()
        # Add HIP include directories
        if(ROCM_INCLUDE_DIR)
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

