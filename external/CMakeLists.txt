# external input to fitter

# oscillation
add_subdirectory(Prob3plusplus)

# flux
add_subdirectory(hondaflux)
add_subdirectory(hondaflux2d)

# xsec
add_subdirectory(xsec_genie_tune)


add_subdirectory(toyfit_tools)


#### CUDAProb3 compiled library (modern branch)
add_subdirectory(CUDAProb3)
