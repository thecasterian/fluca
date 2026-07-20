cmake_minimum_required(VERSION 3.20)

add_executable(taylor_green_vortex taylor_green_vortex.c)
target_link_libraries(taylor_green_vortex PRIVATE
    fluca::ns
    m
)
