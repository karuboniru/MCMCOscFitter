include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(
  state 
  INTERFACE
)

target_include_directories(
  state
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(
  OscillationParameters
  STATIC
  OscillationParameters.cxx
)

target_link_libraries(
  OscillationParameters
  state
)

add_library(
  Prob3ppOscillation
  STATIC
  Prob3ppOscillation.cxx
)

target_link_libraries(
  Prob3ppOscillation
  state
  Prob3plusplus
  OscillationParameters
)


if (ENABLE_CUDA)

  add_library(
    ParProb3ppOscillation
    STATIC
    ParProb3ppOscillation.cxx       # host-compiled: constructor, re_calculate, GetProb_Hists
    ParProb3ppOscillation_devspan.cu  # CUDA-compiled: get_dev_span_*
  )

  target_include_directories(
    ParProb3ppOscillation
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/external/CUDAProb3/src
  )

  target_link_libraries(
    ParProb3ppOscillation
    state
    cudaprob3
    OscillationParameters
  )

endif()

target_include_directories(
  ParProb3ppOscillation
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
