# mount_drift/ — the TIC calibrator cluster: recovers each sensor's mount (R_BS)
# and world drift (R_DG) via the trained net, and applies the deploy correction
# R_clean = R_DGᵀ·R_device·R_BSᵀ.
#   core/      netfwd net + the tic_calib state machine
#   ports/     tic_calib.h — the calibrate/apply port
#   adapters/  vkc (the Vulkan compute backend the net runs on) + train
#   thirdparty/ volk + Vulkan-Headers (the vendored GPU loader)
# vkc + volk were the old gpu cluster, folded in here (it's the calibrator's GPU
# adapter); the solve cluster borrows vkc_core too, so it depends on mount_drift.

# Standalone-build entry (cloned side-by-side in ~/); a no-op under the superbuild.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.19)
    project(mount_drift C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    option(SINEW_TESTS "Build the cluster's unit/property tests" OFF)
    if(SINEW_TESTS)
        enable_testing()
    endif()
endif()

# The Vulkan compute adapter (vkc) + the vendored volk loader, as one lib so a
# consumer linking both mount_drift and solve gets a single volk (no dup symbols).
add_library(vkc_core STATIC
    adapters/vkc.cpp
    thirdparty/volk/volk.c)
target_include_directories(vkc_core PUBLIC
    adapters
    thirdparty/volk
    thirdparty/Vulkan-Headers/include)
target_compile_definitions(vkc_core PUBLIC VK_NO_PROTOTYPES)  # volk loads vulkan-1 at runtime
target_link_libraries(vkc_core PUBLIC ${CMAKE_DL_LIBS})       # volk dlopen()s the loader

add_library(mount_drift_core STATIC
    core/netfwd.cpp
    core/tic_calib.c)
target_include_directories(mount_drift_core PUBLIC core ports)
target_link_libraries(mount_drift_core PUBLIC vkc_core)  # owns its own imu_types (no wire dep)

# ── Unit tests (opt-in: -DSINEW_TESTS=ON) ────────────────────────────────────
if(SINEW_TESTS)
    add_executable(mount_drift_test tests/test_tic_calib.cpp)
    target_link_libraries(mount_drift_test PRIVATE mount_drift_core)
    add_test(NAME mount_drift_tic_calib COMMAND mount_drift_test)
endif()
