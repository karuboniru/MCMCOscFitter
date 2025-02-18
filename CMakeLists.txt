cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(MCMCOscFit)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set (CMAKE_CXX_STANDARD 23)
# find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(ROOT REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${ROOT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/common
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -Wpedantic -ffast-math")

Option(ENABLE_CUDA "Enable CUDA" OFF)

if (ENABLE_CUDA)
    # find_package(CUDA REQUIRED)
    enable_language(CUDA)
endif()

set(DATA_PATH "${CMAKE_SOURCE_DIR}")
add_definitions(-DDATA_PATH="${DATA_PATH}")

#enable asan
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
# set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined ")

add_subdirectory(src)
add_subdirectory(external)
add_subdirectory(cross_check)
