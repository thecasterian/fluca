cmake_minimum_required(VERSION 3.20)

add_executable(cavity_flow_2d cavity_flow_2d.c)
target_link_libraries(cavity_flow_2d PRIVATE
    fluca::ns
    m
)
