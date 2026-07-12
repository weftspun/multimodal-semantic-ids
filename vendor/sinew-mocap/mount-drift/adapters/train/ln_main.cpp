// SPDX-License-Identifier: MIT
// Tiled layernorm driver: fwd then bwd (dx, dg, db) over persistent buffers,
// reusing the cached mean/inv.  Verified by check_ln.py.
//   ln.exe <ln.spv> <R> <D>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: ln <ln.spv> <R> <D>\n");
		return 2;
	}
	const char *spv = argv[1];
	int R = atoi(argv[2]), D = atoi(argv[3]);
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(R * D + D + D + R * D);  // x, g, b, dy
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "ln: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);

	VkcBuf x = vkc_buffer((size_t)R * D * sizeof(float));
	VkcBuf g = vkc_buffer((size_t)D * sizeof(float));
	VkcBuf b = vkc_buffer((size_t)D * sizeof(float));
	VkcBuf dy = vkc_buffer((size_t)R * D * sizeof(float));
	VkcBuf y = vkc_buffer((size_t)R * D * sizeof(float));
	VkcBuf dx = vkc_buffer((size_t)R * D * sizeof(float));
	VkcBuf dg = vkc_buffer((size_t)D * sizeof(float));
	VkcBuf db = vkc_buffer((size_t)D * sizeof(float));
	VkcBuf mean = vkc_buffer((size_t)R * sizeof(float));
	VkcBuf inv = vkc_buffer((size_t)R * sizeof(float));
	VkcBuf dims = vkc_buffer(4 * sizeof(float));
	float *p = in.data();
	memcpy(x.map, p, (size_t)R * D * sizeof(float));
	p += R * D;
	memcpy(g.map, p, (size_t)D * sizeof(float));
	p += D;
	memcpy(b.map, p, (size_t)D * sizeof(float));
	p += D;
	memcpy(dy.map, p, (size_t)R * D * sizeof(float));
	float *d = (float *)dims.map;
	d[0] = (float)R;
	d[1] = (float)D;

	VkcPipeline fwd{}, bdx{}, bdgdb{};
	if (!vkc_pipeline(spv, "ln_fwd", 7, &fwd) || !vkc_pipeline(spv, "ln_bwd_dx", 7, &bdx) ||
	    !vkc_pipeline(spv, "ln_bwd_dgdb", 7, &bdgdb)) {
		return 1;
	}
	{
		VkcBuf bb[7] = {x, g, b, y, mean, inv, dims};
		vkc_run(&fwd, bb, 7, (R + 63) / 64, 1, 1);
	}
	{
		VkcBuf bb[7] = {x, g, dy, dx, mean, inv, dims};
		vkc_run(&bdx, bb, 7, (R + 63) / 64, 1, 1);
	}
	{
		VkcBuf bb[7] = {x, dy, dg, db, mean, inv, dims};
		vkc_run(&bdgdb, bb, 7, (D + 63) / 64, 1, 1);
	}

	std::vector<float> out;
	auto push = [&](VkcBuf &buf, int n) {
		float *m = (float *)buf.map;
		for (int i = 0; i < n; i++) {
			out.push_back(m[i]);
		}
	};
	push(y, R * D);
	push(dx, R * D);
	push(dg, D);
	push(db, D);
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(out.data(), sizeof(float), out.size(), o);
	fclose(o);
	printf("layernorm %dx%d done\n", R, D);
	return 0;
}
