cmake_minimum_required(VERSION 3.20)

add_library(fluca_ns SHARED
    impl/linearcn/cnlinear.c
    impl/linearcn/cnlinearcart2d.c
    interface/nsbasic.c
    interface/nsbc.c
    interface/nsmon.c
    interface/nsopts.c
    interface/nspkg.c
    interface/nsreg.c
    interface/nssol.c
    interface/nssolver.c
)

target_include_directories(fluca_ns PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_ns PUBLIC
    fluca::mesh
)
add_library(fluca::ns ALIAS fluca_ns)
