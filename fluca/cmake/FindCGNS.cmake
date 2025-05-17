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

find_library(CGNS_LIBRARY NAMES cgns
    HINTS ${CGNS_DIR}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)
mark_as_advanced(CGNS_LIBRARY)

if (CGNS_INCLUDE_DIR)
    if (EXISTS ${CGNS_INCLUDE_DIR}/cgnslib.h)
        file(STRINGS ${CGNS_INCLUDE_DIR}/cgnslib.h CGNS_VERSION_H REGEX "#define CGNS_DOTVERS ")
        string(REGEX REPLACE ".*CGNS_DOTVERS[ \t]*([0-9.]+).*" "\\1" CGNS_VERSION ${CGNS_VERSION_H})
        set(CGNS_VERSION ${CGNS_VERSION} CACHE INTERNAL "CGNS version")
    else()
        message(SEND_ERROR "Cannot find ${CGNS_INCLUDE_DIR}/cgnslib.h")
    endif()

    if (EXISTS ${CGNS_INCLUDE_DIR}/pcgnslib.h)
        set(CGNS_ENABLE_PARALLEL ON CACHE INTERNAL "CGNS parallel I/O support")
    else()
        set(CGNS_ENABLE_PARALLEL OFF CACHE INTERNAL "CGNS parallel I/O support")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGNS
    FOUND_VAR CGNS_FOUND
    REQUIRED_VARS CGNS_INCLUDE_DIR CGNS_LIBRARY
    VERSION_VAR CGNS_VERSION
)

if (CGNS_FOUND)
    set(CGNS_INCLUDE_DIRS ${CGNS_INCLUDE_DIR})
    set(CGNS_LIBRARIES ${CGNS_LIBRARY})
endif()
