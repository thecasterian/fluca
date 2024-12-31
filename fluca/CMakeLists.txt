cmake_minimum_required(VERSION 3.20)

# C flags
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-cast-function-type")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")

# Find packages
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(PETSc REQUIRED)
find_package(HDF5 REQUIRED)
find_package(CGNS REQUIRED)

add_subdirectory(app)
add_subdirectory(src)
