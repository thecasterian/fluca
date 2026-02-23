cmake_minimum_required(VERSION 3.20)

add_library(fluca_phys SHARED
    interface/physbasic.c
    interface/physopts.c
    interface/physpkg.c
    interface/physreg.c
)

target_include_directories(fluca_phys PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_phys PUBLIC
    fluca::fd
)
add_library(fluca::phys ALIAS fluca_phys)
