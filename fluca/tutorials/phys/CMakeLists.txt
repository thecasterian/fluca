cmake_minimum_required(VERSION 3.20)

include(FlucaTestUtils)

set(TUTORIAL_SRCS
    ex1.c
    ex2.c
)

foreach(tutorial_src ${TUTORIAL_SRCS})
    get_filename_component(tutorial_base_name ${tutorial_src} NAME_WE)
    set(tutorial_name tutorials_phys_${tutorial_base_name})
    add_executable(${tutorial_name} ${tutorial_src})
    set_target_properties(${tutorial_name} PROPERTIES OUTPUT_NAME ${tutorial_base_name})
    target_link_libraries(${tutorial_name} PRIVATE
        fluca::phys
        m
    )
    fluca_parse_tutorial_file(${CMAKE_CURRENT_SOURCE_DIR}/${tutorial_src} ${tutorial_name} ${tutorial_base_name})
endforeach()
