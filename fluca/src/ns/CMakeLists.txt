cmake_minimum_required(VERSION 3.20)

add_library(fluca_ns SHARED
    impls/cnlinear/cnlinear.c
    interface/nsbasic.c
    interface/nsmon.c
    interface/nsops.c
    interface/nsopts.c
    interface/nspkg.c
    interface/nsreg.c
    interface/nssol.c
    utils/nspcreg.c
    utils/abfpc/abfpc.c
)

target_include_directories(fluca_ns PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_ns PUBLIC
    fluca::phys
    fluca::fd
)
add_library(fluca::ns ALIAS fluca_ns)
