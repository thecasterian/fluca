cmake_minimum_required(VERSION 3.20)

add_library(fluca_sys SHARED
    map/map.c
    viewer/cgns.c
    flucainit.c
    flucapkg.c
)

target_include_directories(fluca_sys PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
add_library(fluca::sys ALIAS fluca_sys)
