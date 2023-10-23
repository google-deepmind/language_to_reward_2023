# Copy the model files to the binary directory to make them available to the
# built binary.

# Start by pulling the Barkour vB model from MuJoCo Menagerie
add_custom_target(
  copy_menagerie_barkour_resources ALL
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/google_barkour_vb/barkour_vb.xml
          ${CMAKE_CURRENT_BINARY_DIR}/barkour_vb.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/google_barkour_vb/assets
          ${CMAKE_CURRENT_BINARY_DIR}/assets
  # apply a patch to barkour_vb.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/barkour_vb_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/barkour_vb.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/barkour_vb.xml.patch
  COMMENT "Copying Barkour vB from MuJoCo Menagerie")

# Add further task XML from this repository
add_custom_target(copy_barkour_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/task_barkour.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/world.xml
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying Barkour task XML into binary directory")

add_dependencies(copy_barkour_resources copy_menagerie_barkour_resources)
