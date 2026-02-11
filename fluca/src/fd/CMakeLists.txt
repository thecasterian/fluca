cmake_minimum_required(VERSION 3.20)

add_library(fluca_fd SHARED
    impls/derivative/derivative.c
    impls/composition/composition.c
    impls/scale/scale.c
    impls/sum/sum.c
    impls/secondordertvd/secondordertvd.c
    impls/secondordertvd/secondordertvdlimiter.c
    interface/fdbasic.c
    interface/fdpkg.c
    interface/fdreg.c
    interface/fdopts.c
    interface/fdapply.c
    utils/fdutils.c
)

target_include_directories(fluca_fd PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_fd PUBLIC
    fluca::sys
)
add_library(fluca::fd ALIAS fluca_fd)
