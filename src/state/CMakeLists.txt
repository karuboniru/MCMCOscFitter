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
    ParProb3ppOscillation.cu
  )

  set_target_properties(ParProb3ppOscillation PROPERTIES CUDA_ARCHITECTURES "all")


  # set_target_properties(
  #   ParProb3ppOscillation
  #   PROPERTIES
  #   CUDA_SEPARABLE_COMPILATION ON
  # )

  target_link_directories(
    ParProb3ppOscillation
    PUBLIC
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  )

else()

  add_library(
    ParProb3ppOscillation
    STATIC
    ParProb3ppOscillation.cxx
  )

endif()
find_package(OpenMP REQUIRED)
# target_link_libraries(CUDAProb3 INTERFACE OpenMP::OpenMP_CXX)

target_link_libraries(
  ParProb3ppOscillation
  state
  CUDAProb3
  OscillationParameters
  OpenMP::OpenMP_CXX
)