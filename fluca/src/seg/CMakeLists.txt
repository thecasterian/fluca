cmake_minimum_required(VERSION 3.20)

add_library(fluca_seg SHARED
    impl/fsm/fsm.c
    interface/segbasic.c
    interface/segmon.c
    interface/segopts.c
    interface/segpkg.c
    interface/segreg.c
)

target_include_directories(fluca_seg PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_seg PUBLIC
    fluca::phys
    fluca::viewer
)
add_library(fluca::seg ALIAS fluca_seg)
