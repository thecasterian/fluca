cmake_minimum_required(VERSION 3.20)

add_library(fluca_ns SHARED
    impl/fsm/fsm.c
    impl/fsm/fsmcart.c
    interface/nsbasic.c
    interface/nscgns.c
    interface/nsmon.c
    interface/nsopts.c
    interface/nspkg.c
    interface/nsreg.c
)

target_include_directories(fluca_ns PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_ns PUBLIC
    fluca::sol
)
add_library(fluca::ns ALIAS fluca_ns)
