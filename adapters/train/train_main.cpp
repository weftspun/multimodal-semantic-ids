// SPDX-License-Identifier: MIT
// Milestone 5 training driver: persistent weight/grad/Adam-moment buffers, loop
// of tgrad (loss + grads) → adam (in-place update).  Reads init.bin (W, x, target),
// writes losses.bin (one loss per step) for check_train.py to compare to torch.
//   train.exe <spv> <NW> <ninX> <ntT> <steps> <lr>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 7) {
		fprintf(stderr, "usage: train <spv> <NW> <ninX> <ntT> <steps> <lr>\n");
		return 2;
	}
	int NW = atoi(argv[2]);
	int ninX = atoi(argv[3]);
	int ntT = atoi(argv[4]);
	int steps = atoi(argv[5]);
	float lr = (float)atof(argv[6]);
	if (!vkc_init()) {
		return 1;
	}

	std::vector<float> init(NW + ninX + ntT);
	FILE *f = fopen("init.bin", "rb");
	if (!f || fread(init.data(), sizeof(float), init.size(), f) != init.size()) {
		fprintf(stderr, "train: cannot read init.bin\n");
		return 1;
	}
	fclose(f);

	VkcBuf W = vkc_buffer((size_t)NW * sizeof(float));
	VkcBuf G = vkc_buffer((size_t)NW * sizeof(float));
	VkcBuf X = vkc_buffer((size_t)ninX * sizeof(float));
	VkcBuf T = vkc_buffer((size_t)ntT * sizeof(float));
	VkcBuf M = vkc_buffer((size_t)NW * sizeof(float));
	VkcBuf V = vkc_buffer((size_t)NW * sizeof(float));
	VkcBuf AUX = vkc_buffer(8 * sizeof(float));
	memcpy(W.map, init.data(), (size_t)NW * sizeof(float));
	memcpy(X.map, init.data() + NW, (size_t)ninX * sizeof(float));
	memcpy(T.map, init.data() + NW + ninX, (size_t)ntT * sizeof(float));

	float *aux = (float *)AUX.map;
	aux[2] = lr;
	aux[3] = 0.9f;
	aux[4] = 0.999f;
	aux[5] = 1e-8f;

	VkcPipeline pg{}, pa{};
	if (!vkc_pipeline(argv[1], "tgrad", 7, &pg)) {
		return 1;
	}
	if (!vkc_pipeline(argv[1], "adam", 7, &pa)) {
		return 1;
	}
	VkcBuf bufs[7] = {W, G, X, T, M, V, AUX};

	std::vector<float> losses(steps);
	for (int step = 1; step <= steps; step++) {
		aux[1] = (float)step;
		vkc_run(&pg, bufs, 7, 1, 1, 1);  // loss + grads at current weights
		losses[step - 1] = aux[0];
		vkc_run(&pa, bufs, 7, (NW + 63) / 64, 1, 1);  // adam update
	}
	FILE *o = fopen("losses.bin", "wb");
	fwrite(losses.data(), sizeof(float), losses.size(), o);
	fclose(o);
	printf("trained %d steps; loss %.6f -> %.6f\n", steps, losses[0], losses[steps - 1]);
	return 0;
}
