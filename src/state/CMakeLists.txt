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
  SHARED
  OscillationParameters.cxx
)

target_link_libraries(
  OscillationParameters
  state
  Prob3plusplus
)