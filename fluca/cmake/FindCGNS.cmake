if (NOT DEFINED CGNS_DIR)
    find_path(CGNS_DIR NAMES include/cgnslib.h
        HINTS ENV CGNS_DIR
        DOC "CGNS installation directory"
    )
endif()

find_path(CGNS_INCLUDE_DIR NAMES cgnslib.h
    HINTS ${CGNS_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
mark_as_advanced(CGNS_INCLUDE_DIR)

if (NOT EXISTS ${CGNS_INCLUDE_DIR}/pcgnslib.h)
    message(FATAL_ERROR "CGNS does not support parallel I/O")
endif()

find_library(CGNS_LIBRARY NAMES cgns
    HINTS ${CGNS_DIR}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)
mark_as_advanced(CGNS_LIBRARY)

function (find_cgns_version)
    file(STRINGS ${CGNS_INCLUDE_DIR}/cgnslib.h CGNS_VERSION_H REGEX "#define CGNS_DOTVERS ")
    string(REGEX REPLACE ".*CGNS_DOTVERS[ \t]*([0-9.]+).*" "\\1" CGNS_VERSION ${CGNS_VERSION_H})
    set(CGNS_VERSION ${CGNS_VERSION} PARENT_SCOPE)
endfunction()
find_cgns_version()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGNS
    FOUND_VAR CGNS_FOUND
    REQUIRED_VARS CGNS_INCLUDE_DIR CGNS_LIBRARY
    VERSION_VAR CGNS_VERSION
)

if (CGNS_FOUND)
    set(CGNS_INCLUDE_DIRS ${CGNS_INCLUDE_DIR})
    set(CGNS_LIBRARIES ${CGNS_LIBRARY})
    set(CGNS_VERSION ${CGNS_VERSION})
endif()
