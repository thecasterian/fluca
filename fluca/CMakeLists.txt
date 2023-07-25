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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add libraries found.
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})
link_libraries(${PETSC_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PETSC_CFLAGS_OTHER}")

add_subdirectory(app)
