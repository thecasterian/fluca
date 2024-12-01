cmake_minimum_required(VERSION 3.20)

add_library(fluca_sys SHARED
    flucainit.c
    flucapkg.c
)

target_include_directories(fluca_sys PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_sys PUBLIC
    fluca::viewer
)
add_library(fluca::sys ALIAS fluca_sys)
