cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(MCMCOscFit)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set (CMAKE_CXX_STANDARD 23)
# find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(ROOT REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${ROOT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/concepts
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -Wpedantic")
# Force bfd linker for safety (local machine has mold as default ld).
add_link_options(-fuse-ld=bfd)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # required for pybind11 .so

Option(ENABLE_CUDA "Enable CUDA" OFF)

set(OSCILLATION_FP "float" CACHE STRING "Oscillation calculation precision: float or double")
set_property(CACHE OSCILLATION_FP PROPERTY STRINGS float double)
if(NOT OSCILLATION_FP STREQUAL "float" AND NOT OSCILLATION_FP STREQUAL "double")
    message(FATAL_ERROR "OSCILLATION_FP must be 'float' or 'double', got '${OSCILLATION_FP}'")
endif()

if (ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -allow-unsupported-compiler")
    enable_language(CUDA)
    add_definitions(-D__MDSPAN_USE_PAREN_OPERATOR=1)
    add_definitions(-DOSCILLATION_FP=${OSCILLATION_FP})
endif()

set(DATA_PATH "${CMAKE_SOURCE_DIR}")
add_definitions(-DDATA_PATH="${DATA_PATH}")

#enable asan
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
# set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined ")

add_subdirectory(src)
add_subdirectory(external)
add_subdirectory(cross_check)

enable_testing()
add_subdirectory(test)
add_subdirectory(pybind)
