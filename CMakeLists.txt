cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  language_to_reward_2023
  VERSION 0.1.0
  DESCRIPTION "Language to Reward: 2023 paper"
  HOMEPAGE_URL "https://github.com/google-deepmind/language_to_reward_2023"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(FindOrFetch)

find_package(Threads REQUIRED)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
  OFF
  CACHE INTERNAL "Build SHARED libraries"
)

set(L2R_MJPC_GIT_TAG
    9752219638bb0d60db772177c1a1a230560f3f92
    CACHE STRING "Git revision for MuJoCo MPC."
)

set(L2R_MENAGERIE_GIT_TAG
    main
    CACHE STRING "Git revision for MuJoCo Menagerie."
)

set(MJPC_BUILD_TESTS OFF)
set(MJPC_GRPC_BUILD_TESTS OFF)
set(MJPC_BUILD_GRPC_SERVICE ON)
set(PYMJPC_BUILD_TESTS OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mjpc
  LIBRARY_NAME
  mjpc
  GIT_REPO
  https://github.com/google-deepmind/mujoco_mpc.git
  GIT_TAG
  ${L2R_MJPC_GIT_TAG}
  TARGETS
  libmjpc
  mjpc_agent_service
  mjpc_ui_agent_service
)

set(BUILD_SHARED_LIBS
  ${BUILD_SHARED_LIBS_OLD}
  CACHE BOOL "Build SHARED libraries" FORCE
)
unset(BUILD_SHARED_LIBS_OLD)

FetchContent_Declare(
  menagerie
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco_menagerie.git
  GIT_TAG ${L2R_MENAGERIE_GIT_TAG}
)

FetchContent_GetProperties(menagerie)
if(NOT menagerie_POPULATED)
  FetchContent_Populate(menagerie)
endif()

add_subdirectory(mjpc)
