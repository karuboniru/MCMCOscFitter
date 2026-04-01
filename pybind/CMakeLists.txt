include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(mcmcoscfitter bindings.cxx)

target_link_libraries(mcmcoscfitter PRIVATE
  BinnedInteractionInject   # injectable ctor + all methods, no physics globals
  ParProb3ppOscillation     # default propagator users can pass in
  OscillationParameters
  SimpleDataHist
  walker
  ROOT::Core
  ROOT::Hist
)

target_include_directories(mcmcoscfitter PRIVATE
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/state
  ${CMAKE_SOURCE_DIR}/src/llh
  ${CMAKE_SOURCE_DIR}/src/data
  ${CMAKE_SOURCE_DIR}/src/walker
)
