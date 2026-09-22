cmake_minimum_required(VERSION 3.20)

add_library(fluca_phys SHARED
    impls/laminar/laminar.c
    impls/laminar/laminarops.c
    impls/laminar/laminarsystem.c
    interface/physbasic.c
    interface/physfield.c
    interface/physopts.c
    interface/physpkg.c
    interface/physprop.c
    interface/physreg.c
)
target_include_directories(fluca_phys PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_phys PUBLIC fluca::fd)
add_library(fluca::phys ALIAS fluca_phys)
