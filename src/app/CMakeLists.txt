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


add_executable(event_rateCU event_rateCU.cxx)

target_link_libraries(event_rateCU 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
)


install (TARGETS event_rateCU DESTINATION bin)


add_executable(testfitbinnedCU testfitbinnedCU.cxx)

target_link_libraries(testfitbinnedCU 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
)


install (TARGETS testfitbinnedCU DESTINATION bin)



add_executable(chi2fitCU chi2fitCU.cxx)

target_link_libraries(chi2fitCU 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
  ROOT::Minuit2
)


install (TARGETS chi2fitCU DESTINATION bin)


add_executable(event_rate_xcheck event_rate_xcheck.cxx)

target_link_libraries(event_rate_xcheck 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
)


install (TARGETS event_rate_xcheck DESTINATION bin)


add_executable(chi2fittestCU chi2fittestCU.cxx)

target_link_libraries(chi2fittestCU 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
  ROOT::Minuit2
)


install (TARGETS chi2fittestCU DESTINATION bin)


add_executable(nofit_check nofit_check.cxx)

target_link_libraries(nofit_check 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
  ROOT::Minuit2
)


install (TARGETS nofit_check DESTINATION bin)

add_executable(single_fit single_fit.cxx)
target_link_libraries(single_fit 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
  ROOT::Minuit2
)
install (TARGETS single_fit DESTINATION bin)

add_executable(triple_fit triple_fit.cxx)
target_link_libraries(triple_fit 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
  ROOT::Minuit2
)
install (TARGETS triple_fit DESTINATION bin)

add_executable(event_rate_xcheck_new event_rate_xcheck_new.cxx)
target_link_libraries(event_rate_xcheck_new 
  SimpleDataHist
  ParBinned
  walker
  ROOT::ROOTDataFrame
  ROOT::Minuit2
)
install (TARGETS event_rate_xcheck_new DESTINATION bin)