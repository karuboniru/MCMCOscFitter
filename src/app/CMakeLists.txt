include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(testsample testsample.cxx)

target_link_libraries(testsample 
  Prob3plusplus
  HondaFlux
  GENIE_XSEC
  ROOT::ROOTDataFrame
  state
  walker
)


install (TARGETS testsample DESTINATION bin)

add_executable(testfit testfit.cxx)

target_link_libraries(testfit 
  SimpleDataPoint
  SimpleInteraction
  walker
  ROOT::ROOTDataFrame
)


install (TARGETS testfit DESTINATION bin)