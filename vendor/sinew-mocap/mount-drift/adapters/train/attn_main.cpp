// SPDX-License-Identifier: MIT
// Tiled attention driver: fwd then bwd (dsim → dQ, dK, dV) over persistent buffers.
// Verified by check_attn.py.
//   attn.exe <attn.spv> <S> <D> <H>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

int main(int argc, char **argv) {
	if (argc < 5) {
		fprintf(stderr, "usage: attn <attn.spv> <S> <D> <H>\n");
		return 2;
	}
	const char *spv = argv[1];
	int S = atoi(argv[2]), D = atoi(argv[3]), H = atoi(argv[4]);
	int DK = D / H;
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(4 * S * D);  // Q, K, V, dOut
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "attn: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);

	VkcBuf Q = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf K = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf V = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf Out = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf P = vkc_buffer((size_t)H * S * S * sizeof(float));
	VkcBuf dOut = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf dQ = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf dK = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf dV = vkc_buffer((size_t)S * D * sizeof(float));
	VkcBuf dsim = vkc_buffer((size_t)H * S * S * sizeof(float));
	VkcBuf dims = vkc_buffer(4 * sizeof(float));
	memcpy(Q.map, in.data(), (size_t)S * D * sizeof(float));
	memcpy(K.map, in.data() + S * D, (size_t)S * D * sizeof(float));
	memcpy(V.map, in.data() + 2 * S * D, (size_t)S * D * sizeof(float));
	memcpy(dOut.map, in.data() + 3 * S * D, (size_t)S * D * sizeof(float));
	float *d = (float *)dims.map;
	d[0] = (float)S;
	d[1] = (float)D;
	d[2] = (float)H;
	d[3] = (float)DK;

	VkcPipeline fwd{}, bds{}, bq{}, bk{}, bv{};
	if (!vkc_pipeline(spv, "attn_fwd", 11, &fwd) || !vkc_pipeline(spv, "attn_bwd_dsim", 11, &bds) ||
	    !vkc_pipeline(spv, "attn_bwd_dq", 11, &bq) || !vkc_pipeline(spv, "attn_bwd_dk", 11, &bk) ||
	    !vkc_pipeline(spv, "attn_bwd_dv", 11, &bv)) {
		return 1;
	}
	VkcBuf bb[11] = {Q, K, V, Out, P, dOut, dQ, dK, dV, dsim, dims};
	int gx = (S + 15) / 16, gy = (H + 15) / 16;
	vkc_run(&fwd, bb, 11, gx, gy, 1);
	vkc_run(&bds, bb, 11, gx, gy, 1);
	vkc_run(&bq, bb, 11, gx, gy, 1);
	vkc_run(&bk, bb, 11, gx, gy, 1);
	vkc_run(&bv, bb, 11, gx, gy, 1);

	std::vector<float> out;
	auto push = [&](VkcBuf &buf) {
		float *m = (float *)buf.map;
		for (int i = 0; i < S * D; i++) {
			out.push_back(m[i]);
		}
	};
	push(Out);
	push(dQ);
	push(dK);
	push(dV);
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(out.data(), sizeof(float), out.size(), o);
	fclose(o);
	printf("attention S=%d D=%d H=%d done\n", S, D, H);
	return 0;
}
