// SPDX-License-Identifier: MIT
// General Adam driver: init.bin [W0(n), G(n)]; run `steps` adam_n updates with a
// fixed gradient; write the final W to outputs.bin.  Verified by check_adamn.py.
//   adamn.exe <adamn.spv> <n> <steps> <lr>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 5) {
		fprintf(stderr, "usage: adamn <adamn.spv> <n> <steps> <lr>\n");
		return 2;
	}
	int n = atoi(argv[2]), steps = atoi(argv[3]);
	float lr = (float)atof(argv[4]);
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(2 * n);
	FILE *f = fopen("init.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "adamn: cannot read init.bin\n");
		return 1;
	}
	fclose(f);
	VkcBuf W = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf G = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf M = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf V = vkc_buffer((size_t)n * sizeof(float));
	VkcBuf params = vkc_buffer(8 * sizeof(float));
	memcpy(W.map, in.data(), (size_t)n * sizeof(float));
	memcpy(G.map, in.data() + n, (size_t)n * sizeof(float));
	float *pp = (float *)params.map;
	pp[1] = lr;
	pp[2] = 0.9f;
	pp[3] = 0.999f;
	pp[4] = 1e-8f;
	pp[5] = (float)n;

	VkcPipeline pipe{};
	if (!vkc_pipeline(argv[1], "adam_n", 5, &pipe)) {
		return 1;
	}
	VkcBuf bufs[5] = {W, G, M, V, params};
	for (int t = 1; t <= steps; t++) {
		pp[0] = (float)t;
		vkc_run(&pipe, bufs, 5, (n + 63) / 64, 1, 1);
	}
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(W.map, sizeof(float), (size_t)n, o);
	fclose(o);
	printf("adamn n=%d steps=%d done\n", n, steps);
	return 0;
}
