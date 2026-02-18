# RunTutorial.cmake
# Helper script for running tutorial examples and checking exit code.
# Unlike RunTest.cmake, this does not compare output against golden files.
#
# Expected variables:
# - TEST_EXECUTABLE: Path to the tutorial executable
# - TEST_ARGS: Command-line arguments for the tutorial (can be empty)

# Validate required variables
if(NOT TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE not defined")
endif()

# Build the command
if(TEST_ARGS)
    separate_arguments(arg_list UNIX_COMMAND "${TEST_ARGS}")
    set(test_command ${TEST_EXECUTABLE} ${arg_list})
else()
    set(test_command ${TEST_EXECUTABLE})
endif()

# Run the tutorial
execute_process(
    COMMAND ${test_command}
    OUTPUT_QUIET
    ERROR_VARIABLE test_stderr
    RESULT_VARIABLE test_result
)

# Check if the tutorial succeeded
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Tutorial failed with exit code: ${test_result}\nStderr: ${test_stderr}")
endif()

# Success
message(STATUS "Tutorial passed: exit code 0")
