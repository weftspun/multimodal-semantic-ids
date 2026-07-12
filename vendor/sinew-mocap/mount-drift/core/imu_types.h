// SPDX-License-Identifier: MIT
// Copyright (c) 2026-present K. S. Ernest (iFire) Lee
//
// mount_drift/core — the trivial IMU value types the calibrator operates on.
// Vendor-neutral; mount_drift owns its own copy so it depends on no other cluster.
// Guard against redefinition when the driver's sinew_osc.h is included first —
// both define identical structs, so skipping is safe.
#pragma once

#ifndef SINEW_QUAT_DEFINED
#define SINEW_QUAT_DEFINED
typedef struct {
	float w, x, y, z;
} Quat;
#endif

#ifndef SINEW_ACCEL_DEFINED
#define SINEW_ACCEL_DEFINED
typedef struct {
	float x, y, z;
} Accel;
#endif
