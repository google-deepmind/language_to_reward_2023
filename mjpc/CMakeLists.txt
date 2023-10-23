add_executable(
  l2r_headless_server
  headless_server.cc
  barkour/barkour.cc
  barkour/barkour.h
)

target_link_libraries(
  l2r_headless_server
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::status
  absl::strings
  mujoco::mujoco
  libmjpc
  mjpc_agent_service
)

target_include_directories(l2r_headless_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(AGENT_SERVICE_COMPILE_OPTIONS=${AGENT_SERVICE_COMPILE_OPTIONS})
target_compile_options(l2r_headless_server PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(l2r_headless_server PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})

add_executable(
  l2r_ui_server
  ui_server.cc
  barkour/barkour.cc
  barkour/barkour.h
)

target_link_libraries(
  l2r_ui_server
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::status
  absl::strings
  mujoco::mujoco
  libmjpc
  mjpc_ui_agent_service
)

target_include_directories(l2r_ui_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(l2r_ui_server PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(l2r_ui_server PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})

add_subdirectory(barkour)
add_dependencies(l2r_headless_server copy_barkour_resources)
add_dependencies(l2r_ui_server copy_barkour_resources)
