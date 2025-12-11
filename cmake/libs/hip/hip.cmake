# HIP/ROCm configuration for FKL
# Similar to cmake/libs/cuda/cuda.cmake

# Set HIP compiler flags
if(ENABLE_HIP)
    # Add HIP-specific compile definitions
    add_compile_definitions(
        __HIP_PLATFORM_AMD__=1
        CLANG_HOST_DEVICE=1
    )
    
    # Set HIP language standard
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    
    # HIP compiler flags
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++17")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC")
    
    # Architecture-specific flags
    if(CMAKE_HIP_ARCHITECTURES)
        foreach(arch ${CMAKE_HIP_ARCHITECTURES})
            set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=${arch}")
        endforeach()
    endif()
    
    message(STATUS "HIP compiler flags: ${CMAKE_HIP_FLAGS}")
    message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()

