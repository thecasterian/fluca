cmake_minimum_required(VERSION 3.20)

add_library(fluca_seg SHARED
    interface/segbasic.c
    interface/segopts.c
    interface/segreg.c
    interface/segpkg.c
    interface/segsol.c
    impls/srk/srk.c
    impls/srk/srktab.c
    impls/srk/srkstep.c
)
target_include_directories(fluca_seg PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_seg PUBLIC fluca::fd)
add_library(fluca::seg ALIAS fluca_seg)
