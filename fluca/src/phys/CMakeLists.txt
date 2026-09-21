cmake_minimum_required(VERSION 3.20)

add_library(fluca_phys SHARED
    impls/ins/ins.c
    impls/ins/insops.c
    impls/ins/inssystem.c
    interface/physbasic.c
    interface/physfield.c
    interface/physopts.c
    interface/physpkg.c
    interface/physreg.c
    utils/abfpc/abfpc.c
)
target_include_directories(fluca_phys PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_phys PUBLIC fluca::fd)
add_library(fluca::phys ALIAS fluca_phys)
