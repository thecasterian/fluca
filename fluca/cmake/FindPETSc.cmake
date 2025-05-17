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
    # --with-single-library=0
    find_petsc_library(PETSC_LIBRARY_SYS petscsys)
    find_petsc_library(PETSC_LIBRARY_MAT petscmat)
    find_petsc_library(PETSC_LIBRARY_DM petscdm)
    find_petsc_library(PETSC_LIBRARY_KSP petscksp)
    find_petsc_library(PETSC_LIBRARY_SNES petscsnes)
    find_petsc_library(PETSC_LIBRARY_TS petscts)
    find_petsc_library(PETSC_LIBRARY_TAO petsctao)
    set(PETSC_LIBRARY
        ${PETSC_LIBRARY_VEC}
        ${PETSC_LIBRARY_SYS}
        ${PETSC_LIBRARY_MAT}
        ${PETSC_LIBRARY_DM}
        ${PETSC_LIBRARY_KSP}
        ${PETSC_LIBRARY_SNES}
        ${PETSC_LIBRARY_TS}
        ${PETSC_LIBRARY_TAO}
    )
else()
    # --with-single-library=1
    find_petsc_library(PETSC_LIBRARY_SINGLE petsc)
    set(PETSC_LIBRARY ${PETSC_LIBRARY_SINGLE})
endif()

if (PETSC_INCLUDE_DIR)
    if (EXISTS ${PETSC_INCLUDE_DIR}/petscversion.h)
        file(STRINGS ${PETSC_INCLUDE_DIR}/petscversion.h PETSC_VERSION_H REGEX "#define PETSC_VERSION_(MAJOR|MINOR|SUBMINOR) ")
        foreach (line ${PETSC_VERSION_H})
            string(REGEX MATCH "PETSC_VERSION_(MAJOR|MINOR|SUBMINOR)[ \t]*([0-9]+)" _ ${line})
            set(PETSC_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
        endforeach()
        set(PETSC_VERSION ${PETSC_VERSION_MAJOR}.${PETSC_VERSION_MINOR}.${PETSC_VERSION_SUBMINOR} CACHE INTERNAL "PETSc version")
    else()
        message(SEND_ERROR "Cannot find ${PETSC_INCLUDE_DIR}/petscversion.h")
    endif()
endif()

if (PETSC_DIR)
    set(PETSC_CONF_VARIABLES ${PETSC_DIR}/lib/petsc/conf/variables)
    if (EXISTS ${PETSC_CONF_VARIABLES})
        find_program(MAKE_PROGRAM NAMES make)
        set(PETSC_CONF_VARIABLES_MAKEFILE ${PROJECT_BINARY_DIR}/petsc_conf_variables.make)
        file(WRITE ${PETSC_CONF_VARIABLES_MAKEFILE}
"include ${PETSC_CONF_VARIABLES}
print_conf_variable:
\t@echo -n \${\${CONF_VAR_NAME}}
")
        macro (find_petsc_conf_variable var conf_var_name)
            execute_process(COMMAND ${MAKE_PROGRAM} -f ${PETSC_CONF_VARIABLES_MAKEFILE} --no-print-directory print_conf_variable CONF_VAR_NAME=${conf_var_name}
                OUTPUT_VARIABLE OUTPUT
            )
            set(${var} ${OUTPUT} CACHE INTERNAL "PETSc configuration variable ${conf_var_name}")
            mark_as_advanced(${var})
        endmacro()
        find_petsc_conf_variable(PETSC_CC PCC)
        find_petsc_conf_variable(PETSC_MPIEXEC MPIEXEC)
        find_petsc_conf_variable(PETSC_SCALAR PETSC_SCALAR)
        find_petsc_conf_variable(PETSC_PRECISION PETSC_PRECISION)
        find_petsc_conf_variable(PETSC_SCALAR_SIZE PETSC_SCALAR_SIZE)
        find_petsc_conf_variable(PETSC_INDEX_SIZE PETSC_INDEX_SIZE)
        file(REMOVE ${PETSC_CONF_VARIABLES_MAKEFILE})
    else()
        message(WARNING "Cannot find PETSc configuration variables; skip checking configurations. Please check your PETSc installation or use the latest version.")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
    FOUND_VAR PETSC_FOUND
    REQUIRED_VARS PETSC_INCLUDE_DIR PETSC_LIBRARY
    VERSION_VAR PETSC_VERSION
)

if (PETSC_FOUND)
    set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIR})
    set(PETSC_LIBRARIES ${PETSC_LIBRARY})
endif()
