cmake_minimum_required(VERSION 3.16.3)

project(robomaster_gesture_native LANGUAGES C)

find_package(
  LeapSDK 6 REQUIRED
  PATHS "$ENV{ProgramFiles}/Ultraleap"
)

add_library(leap_hand_bridge SHARED
  native/leap_hand_bridge.c
  native/leap_hand_bridge.h
)

target_compile_definitions(leap_hand_bridge PRIVATE LHB_BUILD_DLL)
target_include_directories(leap_hand_bridge PUBLIC native)
target_link_libraries(leap_hand_bridge PRIVATE LeapSDK::LeapC)

get_target_property(
  LEAPC_IMPORTED_CONFIG
  LeapSDK::LeapC
  IMPORTED_CONFIGURATIONS
)
get_target_property(
  LEAPC_SHARED_LIB_PATH
  LeapSDK::LeapC
  IMPORTED_LOCATION_${LEAPC_IMPORTED_CONFIG}
)

add_custom_command(
  TARGET leap_hand_bridge
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${LEAPC_SHARED_LIB_PATH}
          $<TARGET_FILE_DIR:leap_hand_bridge>
)

