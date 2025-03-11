SET(WingFlux_VERSION 3.10.4)
project(WingFlux VERSION ${WingFlux_VERSION} LANGUAGES CXX C)

#Changes default install path to be a subdirectory of the build dir.
#Can set build dir at configure time with -DCMAKE_INSTALL_PREFIX=/install/path
if(CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX STREQUAL
  "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
elseif(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

SET(HEADERS WingFlux.h)

SET(SOURCE WingFlux.cxx WingFlux.cxx)

add_library(WingFlux SHARED ${SOURCE})

target_link_libraries(
  WingFlux PUBLIC 
  ROOT::Core
  ROOT::Hist
)

set_target_properties(WingFlux PROPERTIES 
	PUBLIC_HEADER "${HEADERS}"
	EXPORT_NAME WingFlux)

target_include_directories(
  WingFlux PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                       $<INSTALL_INTERFACE:include>
)

install(TARGETS WingFlux 
		EXPORT WingFlux-target
		LIBRARY DESTINATION lib/
		PUBLIC_HEADER DESTINATION include/)

install(EXPORT WingFlux-target
  FILE WingFluxTargets.cmake
  NAMESPACE WingFlux::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/WingFlux
)

