include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ── test_pure ──────────────────────────────────────────────────────────────
# Tests with zero external physics dependencies (binning utilities, constants).
add_executable(test_pure
  test_binning.cxx
  test_constants.cxx
)
target_link_libraries(test_pure PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_pure PRIVATE
  ${CMAKE_SOURCE_DIR}/src/common
)
catch_discover_tests(test_pure)

# ── test_physics ───────────────────────────────────────────────────────────
# Tests that need ROOT (walker, OscillationParameters, chi2 statistic).
add_executable(test_physics
  test_walker.cxx
  test_osc_params.cxx
  test_chi2.cxx
  test_parallel_tempering.cxx
)
target_link_libraries(test_physics PRIVATE
  Catch2::Catch2WithMain
  OscillationParameters
  walker
  ROOT::Core
  ROOT::Hist
)
target_include_directories(test_physics PRIVATE
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/state
  ${CMAKE_SOURCE_DIR}/src/walker
  ${CMAKE_SOURCE_DIR}/src/llh
)
catch_discover_tests(test_physics)

# ── test_binned ────────────────────────────────────────────────────────────
# Integration test for BinnedInteraction with a mock propagator.
# Does NOT link against HondaFlux2D or GENIE_XSEC — uses the injectable
# constructor instead.
add_executable(test_binned
  test_binned_interaction.cxx
)
target_link_libraries(test_binned PRIVATE
  Catch2::Catch2WithMain
  BinnedInteraction
  SimpleDataHist
  ROOT::Core
  ROOT::Hist
)
target_include_directories(test_binned PRIVATE
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src/state
  ${CMAKE_SOURCE_DIR}/src/llh
  ${CMAKE_SOURCE_DIR}/src/data
)
catch_discover_tests(test_binned)
