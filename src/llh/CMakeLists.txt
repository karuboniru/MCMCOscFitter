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

# Injectable-only: all BinnedInteraction logic except the production constructor.
# No dependency on HondaFlux2D or GENIE_XSEC — safe to link from the pybind
# module without requiring physics data files at import time.
add_library(
  BinnedInteractionInject
  STATIC
  BinnedInteraction.cxx
)

target_link_libraries(
  BinnedInteractionInject
  ModelDataLLH
  state
  ParProb3ppOscillation
)

target_include_directories(
  BinnedInteractionInject
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Full library: injectable + production constructor (links physics globals).
add_library(
  BinnedInteraction
  STATIC
  BinnedInteractionProd.cxx
)

target_link_libraries(
  BinnedInteraction
  BinnedInteractionInject
  HondaFlux2D
  GENIE_XSEC
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

  target_include_directories(
    ParBinned
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
