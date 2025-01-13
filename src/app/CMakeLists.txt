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

add_executable(event_rate event_rate.cxx)

target_link_libraries(event_rate 
  SimpleDataHist
  BinnedInteraction
  walker
  ROOT::ROOTDataFrame
)


install (TARGETS event_rate DESTINATION bin)

add_executable(plotter plotter.cxx)

target_link_libraries(plotter 
  ROOT::ROOTDataFrame
)


install (TARGETS plotter DESTINATION bin)


add_executable(chi2fit chi2fit.cxx)

target_link_libraries(chi2fit 
  ROOT::ROOTDataFrame
  BinnedInteraction
  ROOT::Minuit2
  SimpleDataHist
)


install (TARGETS chi2fit DESTINATION bin)

add_executable(chi2table chi2table.cxx)

target_link_libraries(chi2table 
  ROOT::ROOTDataFrame
  BinnedInteraction
  ROOT::Minuit2
  SimpleDataHist
)


install (TARGETS chi2table DESTINATION bin)