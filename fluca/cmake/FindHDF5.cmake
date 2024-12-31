if (NOT DEFINED HDF5_DIR)
    find_path(HDF5_DIR NAMES include/hdf5.h
        HINTS ENV HDF5_DIR
        DOC "HDF5 installation directory"
    )
endif()

find_path(HDF5_INCLUDE_DIR NAMES hdf5.h
    HINTS ${HDF5_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
mark_as_advanced(HDF5_INCLUDE_DIR)

find_library(HDF5_LIBRARY NAMES hdf5
    HINTS ${HDF5_DIR}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)
mark_as_advanced(HDF5_LIBRARY)

function (find_hdf5_version)
    file(STRINGS ${HDF5_INCLUDE_DIR}/H5public.h HDF5_VERSION_H REGEX "#define H5_VERS_STR ")
    string(REGEX REPLACE ".*H5_VERS_STR[ \t]*\"([0-9.]+)\".*" "\\1" HDF5_VERSION ${HDF5_VERSION_H})
    set(HDF5_VERSION ${HDF5_VERSION} PARENT_SCOPE)
endfunction()
find_hdf5_version()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5
    FOUND_VAR HDF5_FOUND
    REQUIRED_VARS HDF5_INCLUDE_DIR HDF5_LIBRARY
    VERSION_VAR HDF5_VERSION
)

if (HDF5_FOUND)
    set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
    set(HDF5_LIBRARIES ${HDF5_LIBRARY})
    set(HDF5_VERSION ${HDF5_VERSION})
endif()
