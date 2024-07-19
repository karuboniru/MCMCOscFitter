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
  SHARED
  SimpleDataPoint.cxx
)

target_link_libraries(
  SimpleDataPoint
  state
)