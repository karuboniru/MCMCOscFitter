include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(testsample testsample.cxx)

target_link_libraries(testsample 
  Prob3plusplus
  HondaFlux
  GENIE_XSEC
  ROOT::ROOTDataFrame
  ROOT::MathMore
  state
  walker
  NEUTRINO_STATE
  SimpleInteraction
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

add_executable(testfitbinned testfitbinned.cxx)

target_link_libraries(testfitbinned 
  SimpleDataHist
  BinnedInteraction
  walker
  ROOT::ROOTDataFrame
)


install (TARGETS testfitbinned DESTINATION bin)

add_executable(plotter plotter.cxx)

target_link_libraries(plotter 
  ROOT::ROOTDataFrame
)


install (TARGETS plotter DESTINATION bin)