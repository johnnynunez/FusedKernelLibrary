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

# Method 2: Try to find via Python package (for headers and libraries)
if(NOT TVM_FFI_FOUND)
    # For uv environments, we MUST use the Python from the virtual environment
    # where apache-tvm-ffi is installed, not the build environment Python
    # Priority order:
    # 1. Check for uv virtual environment Python (fused/bin/python3)
    # 2. Use Python3_EXECUTABLE from find_package if available
    # 3. Fallback to system Python
    
    # First, try to find uv virtual environment Python
    if(EXISTS "${CMAKE_SOURCE_DIR}/fused/bin/python3")
        set(PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/fused/bin/python3")
        message(STATUS "Using Python from uv environment: ${PYTHON_EXECUTABLE}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../fused/bin/python3")
        set(PYTHON_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../fused/bin/python3")
        message(STATUS "Using Python from uv environment: ${PYTHON_EXECUTABLE}")
    else()
        # Fallback to find_package Python3 (but this might be the build environment)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_FOUND AND Python3_EXECUTABLE)
            set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
            message(STATUS "Using Python from find_package: ${PYTHON_EXECUTABLE}")
            message(WARNING "This might be the build environment Python. TVM-FFI should be in the uv virtual environment.")
        elseif(DEFINED Python3_EXECUTABLE)
            set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
            message(STATUS "Using Python3_EXECUTABLE: ${PYTHON_EXECUTABLE}")
            message(WARNING "This might be the build environment Python. TVM-FFI should be in the uv virtual environment.")
        else()
            find_program(PYTHON_EXECUTABLE python3 python)
            if(PYTHON_EXECUTABLE)
                message(STATUS "Using system Python: ${PYTHON_EXECUTABLE}")
            endif()
        endif()
    endif()
    
    if(PYTHON_EXECUTABLE)
        # Method 2a: Try to get source directory for building from source (recommended)
        # This is the most reliable method when apache-tvm-ffi is installed
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -m tvm_ffi.config --sourcedir
            OUTPUT_VARIABLE TVM_FFI_SOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE TVM_FFI_CONFIG_RESULT
        )
        if(TVM_FFI_SOURCE_DIR AND EXISTS ${TVM_FFI_SOURCE_DIR})
            message(STATUS "Found TVM-FFI source at ${TVM_FFI_SOURCE_DIR}, building from source")
            add_subdirectory(${TVM_FFI_SOURCE_DIR} tvm_ffi)
            set(TVM_FFI_FOUND TRUE)
        else()
            # Method 2b: Try to find installed package location and look for headers/libs
            # Try to find installed package location
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))"
                OUTPUT_VARIABLE TVM_FFI_PYTHON_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE TVM_FFI_IMPORT_RESULT
            )
            if(TVM_FFI_PYTHON_DIR)
                message(STATUS "TVM-FFI Python package found at: ${TVM_FFI_PYTHON_DIR}")
                # Look for CMake config files in typical locations
                find_file(TVM_FFI_CONFIG_FILE
                    NAMES tvm_ffiConfig.cmake tvm-ffi-config.cmake
                    PATHS
                        ${TVM_FFI_PYTHON_DIR}/../share/cmake/tvm_ffi
                        ${TVM_FFI_PYTHON_DIR}/../../share/cmake/tvm_ffi
                        ${TVM_FFI_PYTHON_DIR}/../lib/cmake/tvm_ffi
                        ${TVM_FFI_PYTHON_DIR}/../../lib/cmake/tvm_ffi
                    NO_DEFAULT_PATH
                )
                if(TVM_FFI_CONFIG_FILE)
                    get_filename_component(TVM_FFI_ROOT ${TVM_FFI_CONFIG_FILE} DIRECTORY)
                    get_filename_component(TVM_FFI_ROOT ${TVM_FFI_ROOT} DIRECTORY)
                    set(tvm_ffi_DIR ${TVM_FFI_ROOT}/lib/cmake/tvm_ffi)
                    find_package(tvm_ffi CONFIG QUIET)
                    if(tvm_ffi_FOUND)
                        message(STATUS "Found TVM-FFI via Python package CMake config at ${TVM_FFI_PYTHON_DIR}")
                        set(TVM_FFI_FOUND TRUE)
                    endif()
                else()
                    # Try to find headers and libraries directly
                    find_path(TVM_FFI_INCLUDE_DIR
                        NAMES tvm/ffi/c_api.h
                        PATHS
                            ${TVM_FFI_PYTHON_DIR}/../include
                            ${TVM_FFI_PYTHON_DIR}/../../include
                            ${TVM_FFI_PYTHON_DIR}/include
                        NO_DEFAULT_PATH
                    )
                    find_library(TVM_FFI_LIBRARY
                        NAMES tvm_ffi_shared tvm_ffi
                        PATHS
                            ${TVM_FFI_PYTHON_DIR}
                            ${TVM_FFI_PYTHON_DIR}/../lib
                            ${TVM_FFI_PYTHON_DIR}/../../lib
                        NO_DEFAULT_PATH
                    )
                    if(TVM_FFI_INCLUDE_DIR OR TVM_FFI_LIBRARY)
                        set(TVM_FFI_FOUND TRUE)
                        if(TVM_FFI_INCLUDE_DIR)
                            set(TVM_FFI_INCLUDE_DIRS ${TVM_FFI_INCLUDE_DIR})
                        endif()
                        if(TVM_FFI_LIBRARY)
                            set(TVM_FFI_LIBRARIES ${TVM_FFI_LIBRARY})
                        endif()
                        message(STATUS "Found TVM-FFI headers/libs via Python package at ${TVM_FFI_PYTHON_DIR}")
                    else()
                        message(STATUS "TVM-FFI Python package found but headers/libs not found. Try: python -m tvm_ffi.config --sourcedir")
                    endif()
                endif()
            else()
                message(STATUS "TVM-FFI Python package not found. Make sure apache-tvm-ffi is installed.")
            endif()
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
    message(STATUS "Note: Make sure apache-tvm-ffi is installed in the same Python environment")
    set(BUILD_PYTHON_BINDINGS OFF)
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
