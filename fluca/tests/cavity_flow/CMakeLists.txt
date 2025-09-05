cmake_minimum_required(VERSION 3.20)

add_executable(cavity_flow cavity_flow.c)
target_link_libraries(cavity_flow PRIVATE
    fluca::ns
    m
)
