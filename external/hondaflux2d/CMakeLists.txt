SET(HondaFlux2D_VERSION 3.10.4)
project(HondaFlux2D VERSION ${HondaFlux2D_VERSION} LANGUAGES CXX C)

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

SET(HEADERS hondaflux2d.h)

SET(SOURCE hondaflux2d.cxx)

add_library(HondaFlux2D SHARED ${SOURCE})

target_link_libraries(
  HondaFlux2D PUBLIC 
  ROOT::Core
  ROOT::Hist
)

set_target_properties(HondaFlux2D PROPERTIES 
	PUBLIC_HEADER "${HEADERS}"
	EXPORT_NAME HondaFlux2D)

target_include_directories(
  HondaFlux2D PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                       $<INSTALL_INTERFACE:include>
)

install(TARGETS HondaFlux2D 
		EXPORT HondaFlux2D-target
		LIBRARY DESTINATION lib/
		PUBLIC_HEADER DESTINATION include/)

install(EXPORT HondaFlux2D-target
  FILE HondaFlux2DTargets.cmake
  NAMESPACE HondaFlux2D::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/HondaFlux2D
)

