include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(data)
add_subdirectory(llh)
add_subdirectory(state)
add_subdirectory(walker)


add_subdirectory(app)