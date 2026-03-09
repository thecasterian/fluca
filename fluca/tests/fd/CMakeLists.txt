cmake_minimum_required(VERSION 3.20)

include(FlucaTestUtils)

set(TEST_SRCS
    ex1.c
    ex2.c
    ex3.c
    ex4.c
    ex5.c
    ex6.c
    ex7.c
    ex8.c
)

foreach(test_src ${TEST_SRCS})
    get_filename_component(test_base_name ${test_src} NAME_WE)
    set(test_name tests_fd_${test_base_name})
    add_executable(${test_name} ${test_src})
    set_target_properties(${test_name} PROPERTIES OUTPUT_NAME ${test_base_name})
    target_link_libraries(${test_name} PRIVATE
        fluca::fd
        m
    )
    fluca_parse_test_file(${CMAKE_CURRENT_SOURCE_DIR}/${test_src} ${test_name} ${test_base_name})
endforeach()
