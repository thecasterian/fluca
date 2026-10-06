cmake_minimum_required(VERSION 3.20)

add_library(fluca_mesh SHARED
    impls/cartesian/cartesian.c
    impls/cartesian/cartesiancgns.c
    interface/meshbasic.c
    interface/meshopts.c
    interface/meshpkg.c
    interface/meshreg.c
    interface/meshvec.c
)

target_include_directories(fluca_mesh PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_mesh PUBLIC
    fluca::sys
    fluca::viewer
)
add_library(fluca::mesh ALIAS fluca_mesh)
