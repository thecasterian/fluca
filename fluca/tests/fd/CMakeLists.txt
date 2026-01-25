cmake_minimum_required(VERSION 3.20)

include(FlucaTestUtils)

set(TEST_SRCS
    ex1.c
    ex2.c
    ex3.c
    ex4.c
    ex5.c
)

foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE
        fluca::fd
        m
    )
    fluca_parse_test_file(${CMAKE_CURRENT_SOURCE_DIR}/${test_src} ${test_name})
endforeach()
