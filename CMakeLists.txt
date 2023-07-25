cmake_minimum_required(VERSION 3.20)
project(fluca CXX)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(fluca)
