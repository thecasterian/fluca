cmake_minimum_required(VERSION 3.20)

add_library(fluca_ib SHARED
    interface/ibbasic.c
    interface/ibopts.c
    interface/ibpkg.c
    interface/ibreg.c
    impls/none/none.c
)
target_include_directories(fluca_ib PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_ib PUBLIC
    fluca::sys
    fluca::viewer
)
add_library(fluca::ib ALIAS fluca_ib)
