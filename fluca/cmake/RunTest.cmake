# RunTest.cmake
# Helper script for running tests and comparing output with golden files.
#
# Expected variables:
# - TEST_EXECUTABLE: Path to the test executable
# - TEST_ARGS: Command-line arguments for the test (can be empty)
# - EXPECTED_OUTPUT: Path to the golden output file

# Validate required variables
if(NOT TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE not defined")
endif()

if(NOT EXPECTED_OUTPUT)
    message(FATAL_ERROR "EXPECTED_OUTPUT not defined")
endif()

# Create a temporary file for capturing stdout
string(RANDOM LENGTH 8 temp_suffix)
set(temp_output "${CMAKE_CURRENT_BINARY_DIR}/test_output_${temp_suffix}.txt")

# Build the command
if(TEST_ARGS)
    separate_arguments(arg_list UNIX_COMMAND "${TEST_ARGS}")
    set(test_command ${TEST_EXECUTABLE} ${arg_list})
else()
    set(test_command ${TEST_EXECUTABLE})
endif()

# Run the test and capture output
execute_process(
    COMMAND ${test_command}
    OUTPUT_FILE ${temp_output}
    ERROR_VARIABLE test_stderr
    RESULT_VARIABLE test_result
)

# Check if the test executable succeeded
if(NOT test_result EQUAL 0)
    # Clean up temp file
    file(REMOVE ${temp_output})
    message(FATAL_ERROR "Test executable failed with exit code: ${test_result}\nStderr: ${test_stderr}")
endif()

# Compare output with expected
execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files ${temp_output} ${EXPECTED_OUTPUT}
    RESULT_VARIABLE compare_result
)

# Clean up temp file
file(REMOVE ${temp_output})

# Check comparison result
if(NOT compare_result EQUAL 0)
    # Read both files for error message
    file(READ ${EXPECTED_OUTPUT} expected_content)
    message(FATAL_ERROR "Output does not match expected output.\nExpected output file: ${EXPECTED_OUTPUT}")
endif()

# Success
message(STATUS "Test passed: output matches expected")
