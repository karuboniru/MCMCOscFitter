# external input to fitter

# oscillation
add_subdirectory(Prob3plusplus)

# flux
add_subdirectory(hondaflux)
add_subdirectory(hondaflux2d)

# xsec
add_subdirectory(xsec_genie_tune)


add_subdirectory(toyfit_tools)


#### CUDAProb3 Header Only
add_library(CUDAProb3 INTERFACE)
target_include_directories(CUDAProb3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/CUDAProb3)
##### openmp required for CUDAProb3
find_package(OpenMP REQUIRED)
target_link_libraries(CUDAProb3 INTERFACE OpenMP::OpenMP_CXX)
