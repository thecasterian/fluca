cmake_minimum_required(VERSION 3.20)

add_library(fluca_sys SHARED
    map/map.c
    flucainit.c
    flucapkg.c
)

target_include_directories(fluca_sys PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
    ${CMAKE_SOURCE_DIR}/fluca/src
)
add_library(fluca::sys ALIAS fluca_sys)
