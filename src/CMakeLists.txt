include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_subdirectory(data)
add_subdirectory(llh)
add_subdirectory(state)
add_subdirectory(walker)


add_subdirectory(app)