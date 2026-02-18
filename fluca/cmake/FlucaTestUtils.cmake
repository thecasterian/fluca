# FlucaTestUtils.cmake
# Reusable CMake functions for parsing and registering tests from source files
# that contain /*TEST ... TEST*/ blocks.

# Get the directory where this module is located (for finding RunTest.cmake)
set(FLUCA_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# fluca_add_test(name base_name executable args output_file source_dir)
#
# Register a single test with CTest.
# - name: Full test name (e.g., tests_fd_ex1_first_deriv)
# - base_name: Base name for default output file (e.g., ex1_first_deriv)
# - executable: Target name of the test executable
# - args: Command-line arguments for the test (can be empty)
# - output_file: Path to golden output file relative to source_dir (can be empty)
# - source_dir: Directory containing the test source and output files
function(fluca_add_test name base_name executable args output_file source_dir)
    # Use default output file if not specified: output/<base_name>.out
    if(NOT output_file)
        set(output_file "output/${base_name}.out")
    endif()

    set(expected_output "${source_dir}/${output_file}")
    add_test(
        NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:${executable}>
            -DTEST_ARGS=${args}
            -DEXPECTED_OUTPUT=${expected_output}
            -P ${FLUCA_CMAKE_DIR}/RunTest.cmake
    )
endfunction()

# fluca_parse_test_file(source_file test_name base_name)
#
# Parse a source file for /*TEST ... TEST*/ blocks and register tests.
# - source_file: Full path to the source file
# - test_name: Full test name prefix (e.g., tests_fd_ex1)
# - base_name: Base name for output files (e.g., ex1)
function(fluca_parse_test_file source_file test_name base_name)

    # Tell CMake to re-configure when the source file changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_file}")

    # Read the source file
    file(READ "${source_file}" file_content)

    # Extract the TEST block using regex
    string(REGEX MATCH "/\\*TEST(.*)TEST\\*/" test_block "${file_content}")

    if(NOT test_block)
        # No TEST block found, skip this file
        return()
    endif()

    # Get the content between /*TEST and TEST*/
    string(REGEX REPLACE "/\\*TEST(.*)TEST\\*/" "\\1" test_content "${test_block}")

    # Get the source directory for output file paths
    get_filename_component(source_dir "${source_file}" DIRECTORY)

    # Parse each test: block
    # Split by "test:" to get individual test cases
    string(REPLACE "\n" ";" lines "${test_content}")

    set(current_suffix "")
    set(current_args "")
    set(current_output_file "")
    set(in_test_block FALSE)

    foreach(line ${lines})
        # Trim whitespace
        string(STRIP "${line}" line)

        if(line MATCHES "^test:$")
            # Start of a new test block
            # First, register the previous test if we have one
            if(in_test_block AND current_suffix)
                fluca_add_test(
                    "${test_name}_${current_suffix}"
                    "${base_name}_${current_suffix}"
                    "${test_name}"
                    "${current_args}"
                    "${current_output_file}"
                    "${source_dir}"
                )
            endif()

            # Reset for new test
            set(current_suffix "")
            set(current_args "")
            set(current_output_file "")
            set(in_test_block TRUE)

        elseif(in_test_block)
            # Parse test configuration lines
            if(line MATCHES "^suffix:[ \t]*(.*)$")
                set(current_suffix "${CMAKE_MATCH_1}")
                string(STRIP "${current_suffix}" current_suffix)

            elseif(line MATCHES "^args:[ \t]*(.*)$")
                set(current_args "${CMAKE_MATCH_1}")
                string(STRIP "${current_args}" current_args)

            elseif(line MATCHES "^output_file:[ \t]*(.*)$")
                set(current_output_file "${CMAKE_MATCH_1}")
                string(STRIP "${current_output_file}" current_output_file)

            elseif(line MATCHES "^nsize:[ \t]*(.*)$")
                # nsize is currently ignored (always 1)
            endif()
        endif()
    endforeach()

    # Register the last test if we have one
    if(in_test_block AND current_suffix)
        fluca_add_test(
            "${test_name}_${current_suffix}"
            "${base_name}_${current_suffix}"
            "${test_name}"
            "${current_args}"
            "${current_output_file}"
            "${source_dir}"
        )
    endif()
endfunction()

# fluca_add_tutorial_test(name executable args)
#
# Register a single tutorial test with CTest (run-only, no golden output comparison).
# - name: Full test name (e.g., tutorials_fd_ex1_default)
# - executable: Target name of the tutorial executable
# - args: Command-line arguments for the tutorial (can be empty)
function(fluca_add_tutorial_test name executable args)
    add_test(
        NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:${executable}>
            -DTEST_ARGS=${args}
            -P ${FLUCA_CMAKE_DIR}/RunTutorial.cmake
    )
    set_tests_properties(${name} PROPERTIES TIMEOUT 300)
endfunction()

# fluca_parse_tutorial_file(source_file tutorial_name base_name)
#
# Parse a source file for /*TEST ... TEST*/ blocks and register tutorial tests.
# Same parsing logic as fluca_parse_test_file, but calls fluca_add_tutorial_test().
# - source_file: Full path to the source file
# - tutorial_name: Full test name prefix (e.g., tutorials_fd_ex1)
# - base_name: Base name (e.g., ex1)
function(fluca_parse_tutorial_file source_file tutorial_name base_name)

    # Tell CMake to re-configure when the source file changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_file}")

    # Read the source file
    file(READ "${source_file}" file_content)

    # Extract the TEST block using regex
    string(REGEX MATCH "/\\*TEST(.*)TEST\\*/" test_block "${file_content}")

    if(NOT test_block)
        # No TEST block found, skip this file
        return()
    endif()

    # Get the content between /*TEST and TEST*/
    string(REGEX REPLACE "/\\*TEST(.*)TEST\\*/" "\\1" test_content "${test_block}")

    # Parse each test: block
    string(REPLACE "\n" ";" lines "${test_content}")

    set(current_suffix "")
    set(current_args "")
    set(in_test_block FALSE)

    foreach(line ${lines})
        # Trim whitespace
        string(STRIP "${line}" line)

        if(line MATCHES "^test:$")
            # Start of a new test block
            # First, register the previous test if we have one
            if(in_test_block AND current_suffix)
                fluca_add_tutorial_test(
                    "${tutorial_name}_${current_suffix}"
                    "${tutorial_name}"
                    "${current_args}"
                )
            endif()

            # Reset for new test
            set(current_suffix "")
            set(current_args "")
            set(in_test_block TRUE)

        elseif(in_test_block)
            # Parse test configuration lines
            if(line MATCHES "^suffix:[ \t]*(.*)$")
                set(current_suffix "${CMAKE_MATCH_1}")
                string(STRIP "${current_suffix}" current_suffix)

            elseif(line MATCHES "^args:[ \t]*(.*)$")
                set(current_args "${CMAKE_MATCH_1}")
                string(STRIP "${current_args}" current_args)

            elseif(line MATCHES "^nsize:[ \t]*(.*)$")
                # nsize is currently ignored (always 1)
            endif()
        endif()
    endforeach()

    # Register the last test if we have one
    if(in_test_block AND current_suffix)
        fluca_add_tutorial_test(
            "${tutorial_name}_${current_suffix}"
            "${tutorial_name}"
            "${current_args}"
        )
    endif()
endfunction()
