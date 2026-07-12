// SPDX-License-Identifier: MIT
// Tiled GEMM driver: reads inputs.bin (A[M*K], B[K*N]), dispatches the named
// GEMM entry 2D over (M, N), writes outputs.bin (C[M*N]).  Used by check_gemm.py.
//   gemm.exe <spv> <entry> <M> <N> <K>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 6) {
		fprintf(stderr, "usage: gemm <spv> <entry> <M> <N> <K>\n");
		return 2;
	}
	const char *spv = argv[1];
	const char *entry = argv[2];
	int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]);
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(M * K + K * N);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "gemm: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);

	VkcBuf A = vkc_buffer((size_t)M * K * sizeof(float));
	VkcBuf B = vkc_buffer((size_t)K * N * sizeof(float));
	VkcBuf C = vkc_buffer((size_t)M * N * sizeof(float));
	VkcBuf dims = vkc_buffer(4 * sizeof(float));
	memcpy(A.map, in.data(), (size_t)M * K * sizeof(float));
	memcpy(B.map, in.data() + M * K, (size_t)K * N * sizeof(float));
	float *d = (float *)dims.map;
	d[0] = (float)M;
	d[1] = (float)N;
	d[2] = (float)K;

	VkcPipeline pipe{};
	if (!vkc_pipeline(spv, entry, 4, &pipe)) {
		return 1;
	}
	VkcBuf bufs[4] = {A, B, C, dims};
	if (!vkc_run(&pipe, bufs, 4, (N + 15) / 16, (M + 15) / 16, 1)) {
		return 1;
	}
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(C.map, sizeof(float), (size_t)M * N, o);
	fclose(o);
	printf("gemm %s %dx%dx%d done\n", entry, M, N, K);
	return 0;
}
