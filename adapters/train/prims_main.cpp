// SPDX-License-Identifier: MIT
// Generic milestone-2/3 driver: one input buffer, one output buffer.  Reads
// inputs.bin (inN floats) into binding 0, dispatches the named entry, writes
// binding 1 (outN floats) to outputs.bin.  Each primitive's kernel owns its
// own flat layout; check_prims.py supplies inN/outN and the torch oracle.
//   prims.exe <spv> <entry> <inN> <outN>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 5) {
		fprintf(stderr, "usage: prims <spv> <entry> <inN> <outN>\n");
		return 2;
	}
	const char *spv = argv[1];
	const char *entry = argv[2];
	int inN = atoi(argv[3]);
	int outN = atoi(argv[4]);
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(inN);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), inN, f) != (size_t)inN) {
		fprintf(stderr, "prims: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);

	VkcBuf bin = vkc_buffer((size_t)inN * sizeof(float));
	VkcBuf bout = vkc_buffer((size_t)outN * sizeof(float));
	memcpy(bin.map, in.data(), (size_t)inN * sizeof(float));

	VkcPipeline pipe{};
	if (!vkc_pipeline(spv, entry, 2, &pipe)) {
		return 1;
	}
	VkcBuf bufs[2] = {bin, bout};
	if (!vkc_run(&pipe, bufs, 2, 1, 1, 1)) {
		return 1;
	}
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(bout.map, sizeof(float), outN, o);
	fclose(o);
	printf("wrote outputs.bin (%d floats)\n", outN);
	return 0;
}
