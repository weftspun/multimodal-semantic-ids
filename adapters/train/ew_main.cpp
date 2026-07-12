// SPDX-License-Identifier: MIT
// Elementwise driver: reads inputs.bin (A[n], B[n]), dispatches the entry over n,
// writes outputs.bin (C[n]).  Used by check_ew.py.
//   ew.exe <ew.spv> <entry> <n>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: ew <ew.spv> <entry> <n>\n");
		return 2;
	}
	int n = atoi(argv[3]);
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(2 * n);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "ew: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);
	VkcBuf A = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf B = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf C = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf dims = vkc_buffer(4 * sizeof(float));
	memcpy(A.map, in.data(), (size_t)n * sizeof(float));
	memcpy(B.map, in.data() + n, (size_t)n * sizeof(float));
	((float *)dims.map)[0] = (float)n;

	VkcPipeline pipe{};
	if (!vkc_pipeline(argv[1], argv[2], 4, &pipe)) {
		return 1;
	}
	VkcBuf bufs[4] = {A, B, C, dims};
	vkc_run(&pipe, bufs, 4, (n + 63) / 64, 1, 1);
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(C.map, sizeof(float), (size_t)n, o);
	fclose(o);
	printf("ew %s n=%d done\n", argv[2], n);
	return 0;
}
