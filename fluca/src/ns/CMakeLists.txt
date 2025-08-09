cmake_minimum_required(VERSION 3.20)

add_library(fluca_ns SHARED
    impl/fsm/fsm.c
    impl/fsm/fsmcart.c
    impl/fsm/fsmcart2d.c
    impl/fsm/fsmcartcgns.c
    impl/fsm/fsmsol.c
    interface/nsbasic.c
    interface/nsbc.c
    interface/nsmon.c
    interface/nsopts.c
    interface/nspkg.c
    interface/nsreg.c
    interface/nssol.c
)

target_include_directories(fluca_ns PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_ns PUBLIC
    fluca::mesh
)
add_library(fluca::ns ALIAS fluca_ns)
