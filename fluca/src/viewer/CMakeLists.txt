cmake_minimum_required(VERSION 3.20)

add_library(fluca_viewer SHARED
    interface/viewerbasic.c
    impl/flucacgns/flucacgns.c
)

target_include_directories(fluca_viewer PUBLIC
    ${PETSC_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${CGNS_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_viewer PUBLIC
    ${PETSC_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${CGNS_LIBRARIES}
)
add_library(fluca::viewer ALIAS fluca_viewer)
