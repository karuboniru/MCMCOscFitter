include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(
  walker
  STATIC
  walker.cxx
)

target_link_libraries(
  walker PUBLIC
  state
  ROOT::MathCore
)

target_include_directories(
  walker
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
  walker PROPERTIES
  PUBLIC_HEADER "walker.h"
  EXPORT_NAME walker
)