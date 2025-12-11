# HIP/ROCm initialization for Fused Kernel Library
# Similar to cuda_init.cmake but for AMD GPUs

cmake_minimum_required(VERSION 3.22)

# Find HIP
find_package(HIP QUIET)

if(NOT HIP_FOUND)
    # Try to find HIP via environment variables
    if(DEFINED ENV{ROCM_PATH})
        set(HIP_ROOT_DIR $ENV{ROCM_PATH}/hip)
    elseif(DEFINED ENV{HIP_PATH})
        set(HIP_ROOT_DIR $ENV{HIP_PATH})
    else
        # Default ROCm installation path
        set(HIP_ROOT_DIR "/opt/rocm/hip")
    endif()
    
    if(EXISTS ${HIP_ROOT_DIR})
        set(HIP_FOUND TRUE)
        message(STATUS "Found HIP at: ${HIP_ROOT_DIR}")
    else()
        message(WARNING "HIP not found. Set ROCM_PATH or HIP_PATH environment variable.")
        message(WARNING "Expected HIP at: ${HIP_ROOT_DIR}")
        return()
    endif()
endif()

# Enable HIP language
enable_language(HIP)

# Set HIP compiler flags
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Set HIP compiler (clang++)
if(NOT CMAKE_HIP_COMPILER)
    find_program(CMAKE_HIP_COMPILER
        NAMES clang++ hipcc
        PATHS
            ${HIP_ROOT_DIR}/bin
            /opt/rocm/llvm/bin
            $ENV{ROCM_PATH}/llvm/bin
    )
    if(CMAKE_HIP_COMPILER)
        message(STATUS "Found HIP compiler: ${CMAKE_HIP_COMPILER}")
    else()
        message(WARNING "HIP compiler not found. Trying default clang++")
        set(CMAKE_HIP_COMPILER "clang++")
    endif()
endif()

# Find ROCm libraries
find_path(ROCM_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    PATHS
        ${HIP_ROOT_DIR}/include
        /opt/rocm/include
        $ENV{ROCM_PATH}/include
)

find_library(ROCM_HIP_LIBRARY
    NAMES hip_hcc hip_amd
    PATHS
        ${HIP_ROOT_DIR}/lib
        /opt/rocm/lib
        $ENV{ROCM_PATH}/lib
)

if(ROCM_INCLUDE_DIR AND ROCM_HIP_LIBRARY)
    message(STATUS "Found ROCm HIP: ${ROCM_HIP_LIBRARY}")
    message(STATUS "ROCm include dir: ${ROCM_INCLUDE_DIR}")
else()
    message(WARNING "ROCm libraries not found completely")
endif()

# Set HIP architecture (similar to CUDA)
# Default to gfx906 (Vega 20) if not specified
if(NOT CMAKE_HIP_ARCHITECTURES)
    # Try to detect GPU architecture
    find_program(ROCMINFO rocminfo PATHS /opt/rocm/bin $ENV{ROCM_PATH}/bin)
    if(ROCMINFO)
        execute_process(
            COMMAND ${ROCMINFO} | grep -m1 "Name:" | awk "{print $2}"
            OUTPUT_VARIABLE DETECTED_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(DETECTED_ARCH)
            message(STATUS "Detected GPU architecture: ${DETECTED_ARCH}")
            set(CMAKE_HIP_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "HIP architectures")
        else()
            set(CMAKE_HIP_ARCHITECTURES "gfx906" CACHE STRING "HIP architectures")
            message(STATUS "Setting default HIP architecture to gfx906")
            message(STATUS "  (Override with -DCMAKE_HIP_ARCHITECTURES=<your_gpu>)")
        endif()
    else()
        set(CMAKE_HIP_ARCHITECTURES "gfx906" CACHE STRING "HIP architectures")
        message(STATUS "Setting default HIP architecture to gfx906")
        message(STATUS "  (Override with -DCMAKE_HIP_ARCHITECTURES=<your_gpu>)")
    endif()
endif()

# Add compile definitions for HIP
add_compile_definitions(
    CLANG_HOST_DEVICE=1
    __HIP_PLATFORM_AMD__=1
)

# Add include directories
include_directories(${ROCM_INCLUDE_DIR})

# Create HIP target properties similar to CUDA
set(FKL_HIP_ENABLED TRUE)
set(FKL_HIP_COMPILER ${CMAKE_HIP_COMPILER})
set(FKL_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})

message(STATUS "HIP support enabled")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

