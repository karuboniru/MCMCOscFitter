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
  OscillationParameters
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
  HondaFlux
  GENIE_XSEC
  OscillationParameters
)