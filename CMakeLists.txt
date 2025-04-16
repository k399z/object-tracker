cmake_minimum_required(VERSION 3.10)
project(object_tracker)

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