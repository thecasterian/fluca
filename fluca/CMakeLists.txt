cmake_minimum_required(VERSION 3.20)

# Find packages
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(PETSc REQUIRED)
find_package(HDF5 REQUIRED)
find_package(CGNS REQUIRED)

if (${PETSC_VERSION} VERSION_LESS 3.23)
    message(FATAL_ERROR "PETSc version 3.23 or higher is required.")
endif()
if (NOT CGNS_ENABLE_PARALLEL)
    message(FATAL_ERROR "CGNS parallel I/O support is required.")
endif()

# C flags
option(FLUCA_USE_PETSC_CC "Use C compiler used to build PETSc" ON)
if (FLUCA_USE_PETSC_CC)
    set(CMAKE_C_COMPILER ${PETSC_CC})
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")

message(STATUS "C compiler: ${CMAKE_C_COMPILER}")

add_subdirectory(app)
add_subdirectory(src)
add_subdirectory(tests)
