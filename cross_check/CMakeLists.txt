include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(cross_check cross_check.cxx)

target_link_libraries(cross_check 
  ROOT::ROOTDataFrame
  ROOT::MathMore
)

add_executable(prob_check prob_check.cxx)

target_link_libraries(prob_check 
  ROOT::ROOTDataFrame
  ROOT::MathMore
)

add_executable(root2hist root2hist.cxx)

target_link_libraries(root2hist 
  ROOT::ROOTDataFrame
  ROOT::MathMore
  SimpleDataHist
)

