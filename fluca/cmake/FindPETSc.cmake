if (NOT DEFINED PETSC_DIR)
    find_path(PETSC_DIR NAMES include/petsc.h
        HINTS ENV PETSC_DIR
        DOC "PETSc installation directory"
    )
endif()

find_path(PETSC_INCLUDE_DIR NAMES petsc.h
    PATHS ${PETSC_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
mark_as_advanced(PETSC_INCLUDE_DIR)

macro (find_petsc_library var lib)
    find_library(${var} NAMES ${lib}
        PATHS ${PETSC_DIR}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
    )
    mark_as_advanced(${var})
endmacro()
find_petsc_library(PETSC_LIBRARY_VEC petscvec)
if (PETSC_LIBRARY_VEC)
    # TODO: find other libraries
else()
    find_petsc_library(PETSC_LIBRARY_SINGLE petsc)
    set(PETSC_LIBRARY ${PETSC_LIBRARY_SINGLE})
endif()

function (find_petsc_version)
    file(STRINGS ${PETSC_INCLUDE_DIR}/petscversion.h PETSC_VERSION_H REGEX "#define PETSC_VERSION_(MAJOR|MINOR|SUBMINOR) ")
    foreach (line ${PETSC_VERSION_H})
        string(REGEX MATCH "PETSC_VERSION_(MAJOR|MINOR|SUBMINOR)[ \t]*([0-9]+)" _ ${line})
        set(PETSC_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
    endforeach()
    set(PETSC_VERSION ${PETSC_VERSION_MAJOR}.${PETSC_VERSION_MINOR}.${PETSC_VERSION_SUBMINOR} PARENT_SCOPE)
endfunction()
find_petsc_version()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
    FOUND_VAR PETSc_FOUND
    REQUIRED_VARS PETSC_INCLUDE_DIR PETSC_LIBRARY
    VERSION_VAR PETSC_VERSION
)

if (PETSc_FOUND)
    set(PETSc_INCLUDE_DIRS ${PETSC_INCLUDE_DIR})
    set(PETSc_LIBRARIES ${PETSC_LIBRARY})
    set(PETSc_VERSION ${PETSC_VERSION})
endif()
