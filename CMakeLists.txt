cmake_minimum_required(VERSION 3.10)
project(object_tracker)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ARM / Raspberry Pi 4B optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(BASE_OPT_FLAGS -O3 -pipe -ffast-math -fomit-frame-pointer)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        list(APPEND BASE_OPT_FLAGS -mcpu=cortex-a72 -mtune=cortex-a72)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7" OR CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
        list(APPEND BASE_OPT_FLAGS -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mtune=cortex-a72)
    endif()
    add_compile_options(${BASE_OPT_FLAGS})
    # Linker tweaks (compatible with CMake 3.10)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
endif()

# Try enabling LTO / IPO if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(tracker tracker.cc)

# Link libraries
target_link_libraries(tracker ${OpenCV_LIBS})

# Installation rules (optional)
install(TARGETS tracker DESTINATION bin)