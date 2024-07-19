cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(MCMCOscFit)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set (CMAKE_CXX_STANDARD 23)
# find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(ROOT REQUIRED)
include("${ROOT_USE_FILE}")
include_directories(
    ${ROOT_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wpedantic -flto")

add_subdirectory(src)
add_subdirectory(external)
