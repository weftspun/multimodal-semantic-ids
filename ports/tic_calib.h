// SPDX-License-Identifier: MIT
// Copyright (c) 2026-present K. S. Ernest (iFire) Lee
//
// core/calib/tic_calib — the TIC calibrator state machine: rolling-window
// mount/drift recovery via the trained net (netfwd), and the deploy correction
// R_clean = R_DGᵀ·R_device·R_BSᵀ.  Engine-free apart from the Vulkan compute the
// net runs on.  The server feeds the raw /sinew/tracker stream through this
// before the body solve (the slim driver no longer calibrates).
#pragma once
#include <stdint.h>

#include "imu_types.h"  // Quat, Accel (mount_drift owns its own trivial types)

#ifdef __cplusplus
extern "C" {
#endif

// Lazy init from $SINEW_TIC_DIR (or ".") — loads wtrained.bin + the 5 .spv and
// builds the netfwd context.  Called automatically by the first tic_calib_push.
void tic_calib_init(void);

// Feed one per-sensor sample (sensor = device node 0..14).  Once all sensors
// are live it snapshots on each fresh timestamp; every 32 snapshots it runs the
// net and refreshes the per-sensor mount (R_BS) / drift (R_DG).
void tic_calib_push(int sensor, Accel a, Accel m, Quat q, uint64_t ms);

// If a calibration is ready, write R_clean = R_DGᵀ·R_device·R_BSᵀ and return 1;
// else return 0 (the caller passes the device quaternion through uncalibrated).
int tic_calib_apply(int sensor, Quat device, Quat *out);

// Offline accuracy check on a labeled window set (export_selftest.py's
// selftest.bin): runs the same deploy path and prints OME vs ground truth.
// Needs wtrained.bin + the 5 .spv in spv_dir (NULL -> ".") and a GPU.  Returns 0
// on success, nonzero on any setup/read failure.
int tic_selftest(const char *window_bin, const char *spv_dir);

#ifdef __cplusplus
}
#endif
