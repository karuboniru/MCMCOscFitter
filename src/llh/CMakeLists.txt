include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(
  ModelDataLLH 
  INTERFACE
)

target_include_directories(
  ModelDataLLH
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  ModelDataLLH
  INTERFACE
  state
  data
)


add_library(
  SimpleInteraction
  STATIC
  SimpleInteraction.cxx
)

target_link_libraries(
  SimpleInteraction
  ModelDataLLH
  state
  HondaFlux
  GENIE_XSEC
  Prob3ppOscillation
)

add_library(
  BinnedInteraction
  STATIC
  BinnedInteraction.cxx
)

target_link_libraries(
  BinnedInteraction
  ModelDataLLH
  state
  HondaFlux2D
  GENIE_XSEC
  Prob3ppOscillation
  ParProb3ppOscillation
)

if (ENABLE_CUDA)
  add_library(
    ParBinned
    STATIC
    ParBinned.cu
    ParBinnedInterface.cu
    ParBinnedKernels.cu
  )

  target_link_directories(
    ParBinned
    PUBLIC
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  )

  target_link_libraries(
    ParBinned
    ParProb3ppOscillation
    ModelDataLLH
    state
    HondaFlux2D
    GENIE_XSEC
  )
endif()