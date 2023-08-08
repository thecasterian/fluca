cmake_minimum_required(VERSION 3.20)

# PkgConfig
find_package(PkgConfig REQUIRED)

# Find PETSc.
pkg_check_modules(PETSC PETSc REQUIRED)
message(STATUS "PETSc include dirs: ${PETSC_INCLUDE_DIRS}")
message(STATUS "PETSc library dirs: ${PETSC_LIBRARY_DIRS}")
message(STATUS "PETSc libraries: ${PETSC_LIBRARIES}")
message(STATUS "PETSc compile flags: ${PETSC_CFLAGS_OTHER}")

# C++ flags.
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")

# Add libraries found.
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})
link_libraries(${PETSC_LIBRARIES})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PETSC_CFLAGS_OTHER}")

add_subdirectory(app)
add_subdirectory(src)
