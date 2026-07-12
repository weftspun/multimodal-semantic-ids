// SPDX-License-Identifier: MIT
// Keystone: orchestrate the tiled kernels into a full pre-norm EncoderLayer
// forward at real-ish dims (the repeating unit of the production net).  Chains
// ln_fwd, gemm_nt + bias_add, attn_fwd, add, relu_fwd through intermediate
// buffers.  Verified vs torch EncoderLayer in check_enc_tiled.py.
//   enc_tiled.exe <gemm.spv> <lin.spv> <ln.spv> <attn.spv> <ew.spv> <S> <D> <H> <F>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

static VkcBuf g_dims;
static void dims2(float a, float b) {
	float *d = (float *)g_dims.map;
	d[0] = a;
	d[1] = b;
}
static void dims3(float a, float b, float c) {
	float *d = (float *)g_dims.map;
	d[0] = a;
	d[1] = b;
	d[2] = c;
}

int main(int argc, char **argv) {
	if (argc < 10) {
		fprintf(stderr, "usage: enc_tiled <gemm> <lin> <ln> <attn> <ew> <S> <D> <H> <F>\n");
		return 2;
	}
	int S = atoi(argv[6]), D = atoi(argv[7]), H = atoi(argv[8]), F = atoi(argv[9]);
	int DK = D / H;
	if (!vkc_init()) {
		return 1;
	}
	// inputs.bin: x(S*D) then weights in EncoderLayer order
	int total = S * D + 2 * D + 4 * (D * D + D) + 2 * D + (F * D + F) + (D * F + D);
	std::vector<float> in(total);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "enc_tiled: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);
	float *p = in.data();
	auto load = [&](int n) {
		VkcBuf b = vkc_buffer((size_t)n * sizeof(float));
		memcpy(b.map, p, (size_t)n * sizeof(float));
		p += n;
		return b;
	};
	auto scratch = [&](int n) { return vkc_buffer((size_t)n * sizeof(float)); };

	VkcBuf x = load(S * D);
	VkcBuf g1 = load(D), b1 = load(D);
	VkcBuf Wq = load(D * D), bq = load(D), Wk = load(D * D), bk = load(D);
	VkcBuf Wv = load(D * D), bv = load(D), Wo = load(D * D), bo = load(D);
	VkcBuf g2 = load(D), b2 = load(D);
	VkcBuf W1 = load(F * D), b1f = load(F), W2 = load(D * F), b2f = load(D);

	VkcBuf x2 = scratch(S * D), Q = scratch(S * D), K = scratch(S * D), V = scratch(S * D);
	VkcBuf Aa = scratch(S * D), Ao = scratch(S * D), y = scratch(S * D), y2 = scratch(S * D);
	VkcBuf h1 = scratch(S * F), hr = scratch(S * F), h2 = scratch(S * D), out = scratch(S * D);
	VkcBuf mean = scratch(S), inv = scratch(S), P = scratch(H * S * S);
	VkcBuf z1 = scratch(H * S * S), z2 = scratch(S * D), z3 = scratch(S * D), z4 = scratch(S * D),
	       z5 = scratch(S * D);  // attn fwd unused bindings
	g_dims = vkc_buffer(8 * sizeof(float));

	VkcPipeline lnf{}, nt{}, badd{}, af{}, padd{}, relu{};
	if (!vkc_pipeline(argv[1], "gemm_nt", 4, &nt) || !vkc_pipeline(argv[2], "bias_add", 3, &badd) ||
	    !vkc_pipeline(argv[3], "ln_fwd", 7, &lnf) || !vkc_pipeline(argv[4], "attn_fwd", 11, &af) ||
	    !vkc_pipeline(argv[5], "add", 4, &padd) || !vkc_pipeline(argv[5], "relu_fwd", 4, &relu)) {
		return 1;
	}

	auto lnorm = [&](VkcBuf X, VkcBuf g, VkcBuf b, VkcBuf Y) {
		dims2(S, D);
		VkcBuf bb[7] = {X, g, b, Y, mean, inv, g_dims};
		vkc_run(&lnf, bb, 7, (S + 63) / 64, 1, 1);
	};
	auto linear = [&](VkcBuf X, VkcBuf W, VkcBuf b, VkcBuf Y, int In, int Out) {
		dims3(S, Out, In);
		VkcBuf gg[4] = {X, W, Y, g_dims};
		vkc_run(&nt, gg, 4, (Out + 15) / 16, (S + 15) / 16, 1);
		dims2(S, Out);
		VkcBuf ba[3] = {Y, b, g_dims};
		vkc_run(&badd, ba, 3, (Out + 15) / 16, (S + 15) / 16, 1);
	};
	auto eadd = [&](VkcBuf A, VkcBuf B, VkcBuf C, int n) {
		((float *)g_dims.map)[0] = (float)n;
		VkcBuf bb[4] = {A, B, C, g_dims};
		vkc_run(&padd, bb, 4, (n + 63) / 64, 1, 1);
	};

	lnorm(x, g1, b1, x2);
	linear(x2, Wq, bq, Q, D, D);
	linear(x2, Wk, bk, K, D, D);
	linear(x2, Wv, bv, V, D, D);
	{
		float *d = (float *)g_dims.map;
		d[0] = (float)S;
		d[1] = (float)D;
		d[2] = (float)H;
		d[3] = (float)DK;
		VkcBuf bb[11] = {Q, K, V, Aa, P, z2, z3, z4, z5, z1, g_dims};
		vkc_run(&af, bb, 11, (S + 15) / 16, (H + 15) / 16, 1);
	}
	linear(Aa, Wo, bo, Ao, D, D);
	eadd(Ao, x, y, S * D);
	lnorm(y, g2, b2, y2);
	linear(y2, W1, b1f, h1, D, F);
	{
		((float *)g_dims.map)[0] = (float)(S * F);
		VkcBuf bb[4] = {h1, h1, hr, g_dims};
		vkc_run(&relu, bb, 4, (S * F + 63) / 64, 1, 1);
	}
	linear(hr, W2, b2f, h2, F, D);
	eadd(h2, y, out, S * D);

	FILE *o = fopen("outputs.bin", "wb");
	fwrite(out.map, sizeof(float), (size_t)S * D, o);
	fclose(o);
	printf("enc_tiled S=%d D=%d H=%d F=%d done\n", S, D, H, F);
	return 0;
}
