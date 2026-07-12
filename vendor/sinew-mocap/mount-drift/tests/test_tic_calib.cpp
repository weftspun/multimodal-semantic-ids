// SPDX-License-Identifier: MIT
// Unit tests for mount_drift/core (the TIC calibrator's no-GPU API contract):
// with no trained weights / no GPU, the calibrator never reports a calibration
// and the deploy apply is a safe pass-through (returns 0).
#include "tic_calib.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
	// Point at a dir with no wtrained.bin so init fails gracefully (no GPU touched).
#ifdef _WIN32
	_putenv_s("SINEW_TIC_DIR", "/nonexistent-sinew-tic-dir");  // MSVC has no setenv
#else
	setenv("SINEW_TIC_DIR", "/nonexistent-sinew-tic-dir", 1);
#endif

	Quat dev = {1.f, 0.f, 0.f, 0.f};
	Quat out = {0.f, 0.f, 0.f, 0.f};

	// No calibration is ready -> apply returns 0 (caller passes the device quat through).
	assert(tic_calib_apply(0, dev, &out) == 0);
	// Out-of-range sensor indices are rejected.
	assert(tic_calib_apply(-1, dev, &out) == 0);
	assert(tic_calib_apply(99, dev, &out) == 0);

	// Pushing samples without weights is a safe no-op (must not crash).
	Accel a = {0.f, 0.f, 1.f};
	Accel m = {1.f, 0.f, 0.f};
	for (int i = 0; i < 20; i++) {
		tic_calib_push(i % 15, a, m, dev, (unsigned)i);
	}
	// Still no calibration.
	assert(tic_calib_apply(0, dev, &out) == 0);

	std::puts("mount_drift/tic_calib: no-GPU API path passed");
	return 0;
}
