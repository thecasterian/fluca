cmake_minimum_required(VERSION 3.20)

add_library(fluca_mesh SHARED
    impl/cartesian/cartesian.c
    impl/cartesian/cartesiancgns.c
    interface/meshbasic.c
    interface/meshopts.c
    interface/meshpkg.c
    interface/meshreg.c
)

target_include_directories(fluca_mesh PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
    ${CMAKE_SOURCE_DIR}/fluca/src
)
target_link_libraries(fluca_mesh PUBLIC
    fluca::sys
    cgns
)
add_library(fluca::mesh ALIAS fluca_mesh)
