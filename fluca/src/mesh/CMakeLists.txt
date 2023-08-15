cmake_minimum_required(VERSION 3.20)

add_library(fluca_mesh SHARED
    impl/cartesian/cartesian.c
    interface/meshbasic.c
    interface/meshopts.c
    interface/meshpkg.c
    interface/meshreg.c
)

target_include_directories(fluca_mesh PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
    ${CMAKE_SOURCE_DIR}/fluca/src
)
add_library(fluca::mesh ALIAS fluca_mesh)
