include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(
  data 
  INTERFACE
)

target_include_directories(
  data
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(
  SimpleDataPoint
  STATIC
  SimpleDataPoint.cxx
)

target_link_libraries(
  SimpleDataPoint
  state
)

add_library(
  SimpleDataHist
  STATIC
  SimpleDataHist.cxx
)

target_link_libraries(
  SimpleDataHist
  state
  GENIE_XSEC
)

target_include_directories(
  SimpleDataHist
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)