cmake_minimum_required(VERSION 3.20)

add_library(fluca_viewer SHARED
    interface/viewerreg.c
    impl/flucacgns/flucacgns.c
)

target_include_directories(fluca_viewer PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_viewer PUBLIC
    hdf5
    cgns
)
add_library(fluca::viewer ALIAS fluca_viewer)
