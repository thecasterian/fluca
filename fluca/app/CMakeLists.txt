cmake_minimum_required(VERSION 3.20)

add_executable(main main.c)
target_link_libraries(main PRIVATE
    fluca::sys
)
