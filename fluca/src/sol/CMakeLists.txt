cmake_minimum_required(VERSION 3.20)

add_library(fluca_sol SHARED
    impl/fsm/fsm.c
    impl/fsm/fsmcgns.c
    interface/solbasic.c
    interface/solpkg.c
    interface/solreg.c
)

target_include_directories(fluca_sol PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_sol PUBLIC
    fluca::mesh
    fluca::sys
)
add_library(fluca::sol ALIAS fluca_sol)
