// SPDX-License-Identifier: MIT
// Milestone 1 driver: dispatch probe.slang on the GPU and check the autodiff
// result (df/dx(2)=12, f(2)=8) read back from the storage buffer.
#include <cstdio>
#include <cstdlib>

#include "vkc.h"

int main(int argc, char **argv) {
	const char *spv = argc > 1 ? argv[1] : "probe.spv";
	if (!vkc_init()) {
		return 1;
	}
	VkcBuf out = vkc_buffer(16 * sizeof(float));
	VkcPipeline p{};
	if (!vkc_pipeline(spv, "probe", 1, &p)) {
		return 1;
	}
	if (!vkc_run(&p, &out, 1, 1, 1, 1)) {
		return 1;
	}
	float *r = (float *)out.map;
	printf("df/dx(2) = %.6f  (expect 12)\n", r[0]);
	printf("f(2)     = %.6f  (expect 8)\n", r[1]);
	bool ok = (r[0] > 11.99f && r[0] < 12.01f) && (r[1] > 7.99f && r[1] < 8.01f);
	printf("%s\n", ok ? "PASS: Slang autodiff runs on the GPU" : "FAIL");
	return ok ? 0 : 1;
}
