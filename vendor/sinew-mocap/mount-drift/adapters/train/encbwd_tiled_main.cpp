// SPDX-License-Identifier: MIT
// Backward keystone: tiled EncoderLayer fwd (caching activations) + full backward
// by orchestrating the verified bwd kernels (gemm nn/tn, col_sum, ln_bwd, attn_bwd,
// relu_bwd, add) in reverse, with gradient accumulation at the residual splits
// (dy, dX, dx2 each sum multiple paths).  Verified vs torch EncoderLayer grads in
// check_encbwd_tiled.py.  Reuses only verified kernels — orchestration, not new math.
//   encbwd.exe <gemm> <lin> <ln> <attn> <ew> <S> <D> <H> <F>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

static VkcBuf g_dims;
static int S, D, H, F, DK;
static VkcPipeline nt, badd, lnf, af, relu, padd;                           // forward
static VkcPipeline nn, tn, csum, lndx, lndgdb, adsim, adq, adk, adv, rbwd;  // backward

static void dimsN(float a, float b, float c, float d) {
	float *p = (float *)g_dims.map;
	p[0] = a;
	p[1] = b;
	p[2] = c;
	p[3] = d;
}
static void run4(VkcPipeline *p, VkcBuf a, VkcBuf b, VkcBuf c, VkcBuf d, int gx, int gy) {
	VkcBuf bb[4] = {a, b, c, d};
	vkc_run(p, bb, 4, gx, gy, 1);
}
// forward linear: Y = X·Wᵀ + b
static void linear(VkcBuf X, VkcBuf W, VkcBuf b, VkcBuf Y, int rows, int In, int Out) {
	dimsN(rows, Out, In, 0);
	run4(&nt, X, W, Y, g_dims, (Out + 15) / 16, (rows + 15) / 16);
	dimsN(rows, Out, 0, 0);
	VkcBuf ba[3] = {Y, b, g_dims};
	vkc_run(&badd, ba, 3, (Out + 15) / 16, (rows + 15) / 16, 1);
}
static void lnorm(VkcBuf X, VkcBuf g, VkcBuf b, VkcBuf Y, VkcBuf mn, VkcBuf iv) {
	dimsN(S, D, 0, 0);
	VkcBuf bb[7] = {X, g, b, Y, mn, iv, g_dims};
	vkc_run(&lnf, bb, 7, (S + 63) / 64, 1, 1);
}
static void eadd(VkcBuf A, VkcBuf B, VkcBuf C, int n) {
	((float *)g_dims.map)[0] = (float)n;
	run4(&padd, A, B, C, g_dims, (n + 63) / 64, 1);
}
static void colsum(VkcBuf Y, VkcBuf vec, int rows, int cols) {
	dimsN(rows, cols, 0, 0);
	VkcBuf bb[3] = {Y, vec, g_dims};
	vkc_run(&csum, bb, 3, (cols + 63) / 64, 1, 1);
}
// backward linear: given dY (rows,Out), W (Out,In), X (rows,In): dX, dW, db.
static void linbwd(VkcBuf dY, VkcBuf W, VkcBuf X, VkcBuf dX, VkcBuf dW, VkcBuf db, int rows, int In,
                   int Out) {
	dimsN(rows, In, Out, 0);
	run4(&nn, dY, W, dX, g_dims, (In + 15) / 16, (rows + 15) / 16);
	dimsN(Out, In, rows, 0);
	run4(&tn, dY, X, dW, g_dims, (In + 15) / 16, (Out + 15) / 16);
	colsum(dY, db, rows, Out);
}
static void lnbwd(VkcBuf X, VkcBuf g, VkcBuf dY, VkcBuf dX, VkcBuf dg, VkcBuf db, VkcBuf mn,
                  VkcBuf iv) {
	dimsN(S, D, 0, 0);
	VkcBuf a[7] = {X, g, dY, dX, mn, iv, g_dims};
	vkc_run(&lndx, a, 7, (S + 63) / 64, 1, 1);
	VkcBuf b[7] = {X, dY, dg, db, mn, iv, g_dims};
	vkc_run(&lndgdb, b, 7, (D + 63) / 64, 1, 1);
}

int main(int argc, char **argv) {
	if (argc < 10) {
		fprintf(stderr, "usage: encbwd <gemm><lin><ln><attn><ew> S D H F\n");
		return 2;
	}
	S = atoi(argv[6]);
	D = atoi(argv[7]);
	H = atoi(argv[8]);
	F = atoi(argv[9]);
	DK = D / H;
	if (!vkc_init()) {
		return 1;
	}
	int total = S * D + 2 * D + 4 * (D * D + D) + 2 * D + (F * D + F) + (D * F + D) + S * D;
	std::vector<float> in(total);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "encbwd: cannot read inputs.bin\n");
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
	auto sc = [&](int n) { return vkc_buffer((size_t)n * sizeof(float)); };

	VkcBuf X = load(S * D);
	VkcBuf g1 = load(D), b1 = load(D);
	VkcBuf Wq = load(D * D), bq = load(D), Wk = load(D * D), bk = load(D);
	VkcBuf Wv = load(D * D), bv = load(D), Wo = load(D * D), bo = load(D);
	VkcBuf g2 = load(D), b2 = load(D);
	VkcBuf W1 = load(F * D), b1f = load(F), W2 = load(D * F), b2f = load(D);
	VkcBuf dOut = load(S * D);

	// cached forward activations
	VkcBuf x2 = sc(S * D), Q = sc(S * D), K = sc(S * D), Vv = sc(S * D), Aa = sc(S * D);
	VkcBuf Ao = sc(S * D), y = sc(S * D), y2 = sc(S * D), h1 = sc(S * F), hr = sc(S * F);
	VkcBuf h2 = sc(S * D), Out = sc(S * D), P = sc(H * S * S);
	VkcBuf mean1 = sc(S), inv1 = sc(S), mean2 = sc(S), inv2 = sc(S);
	// gradients / temporaries
	VkcBuf dh2 = sc(S * D), dy = sc(S * D), dy0 = sc(S * D), dhr = sc(S * F), dh1 = sc(S * F);
	VkcBuf dy2 = sc(S * D), dyln = sc(S * D), dAo = sc(S * D), dX = sc(S * D), dXln = sc(S * D);
	VkcBuf dAa = sc(S * D), dQ = sc(S * D), dK = sc(S * D), dV = sc(S * D), dsim = sc(H * S * S);
	VkcBuf dx2 = sc(S * D), dx2q = sc(S * D), dx2k = sc(S * D), dx2v = sc(S * D), dx2qk = sc(S * D);
	VkcBuf dg1 = sc(D), db1 = sc(D), dg2 = sc(D), db2 = sc(D);
	VkcBuf dWq = sc(D * D), dbq = sc(D), dWk = sc(D * D), dbk = sc(D), dWv = sc(D * D), dbv = sc(D);
	VkcBuf dWo = sc(D * D), dbo = sc(D), dW1 = sc(F * D), db1f = sc(F), dW2 = sc(D * F),
	       db2f = sc(D);
	VkcBuf scOut = sc(S * D);
	g_dims = vkc_buffer(8 * sizeof(float));

	if (!vkc_pipeline(argv[1], "gemm_nt", 4, &nt) || !vkc_pipeline(argv[1], "gemm_nn", 4, &nn) ||
	    !vkc_pipeline(argv[1], "gemm_tn", 4, &tn) || !vkc_pipeline(argv[2], "bias_add", 3, &badd) ||
	    !vkc_pipeline(argv[2], "col_sum", 3, &csum) || !vkc_pipeline(argv[3], "ln_fwd", 7, &lnf) ||
	    !vkc_pipeline(argv[3], "ln_bwd_dx", 7, &lndx) ||
	    !vkc_pipeline(argv[3], "ln_bwd_dgdb", 7, &lndgdb) ||
	    !vkc_pipeline(argv[4], "attn_fwd", 11, &af) ||
	    !vkc_pipeline(argv[4], "attn_bwd_dsim", 11, &adsim) ||
	    !vkc_pipeline(argv[4], "attn_bwd_dq", 11, &adq) ||
	    !vkc_pipeline(argv[4], "attn_bwd_dk", 11, &adk) ||
	    !vkc_pipeline(argv[4], "attn_bwd_dv", 11, &adv) ||
	    !vkc_pipeline(argv[5], "add", 4, &padd) || !vkc_pipeline(argv[5], "relu_fwd", 4, &relu) ||
	    !vkc_pipeline(argv[5], "relu_bwd", 4, &rbwd)) {
		return 1;
	}
	int agx = (S + 15) / 16, agy = (H + 15) / 16;
	auto attnFwd = [&]() {
		dimsN(S, D, H, DK);
		VkcBuf bb[11] = {Q, K, Vv, Aa, P, dOut, dQ, dK, dV, dsim, g_dims};
		vkc_run(&af, bb, 11, agx, agy, 1);
	};
	auto attnBwd = [&]() {
		dimsN(S, D, H, DK);
		VkcBuf bb[11] = {Q, K, Vv, scOut, P, dAa, dQ, dK, dV, dsim, g_dims};
		vkc_run(&adsim, bb, 11, agx, agy, 1);
		vkc_run(&adq, bb, 11, agx, agy, 1);
		vkc_run(&adk, bb, 11, agx, agy, 1);
		vkc_run(&adv, bb, 11, agx, agy, 1);
	};

	// ---- forward (cache) ----
	lnorm(X, g1, b1, x2, mean1, inv1);
	linear(x2, Wq, bq, Q, S, D, D);
	linear(x2, Wk, bk, K, S, D, D);
	linear(x2, Wv, bv, Vv, S, D, D);
	attnFwd();
	linear(Aa, Wo, bo, Ao, S, D, D);
	eadd(Ao, X, y, S * D);
	lnorm(y, g2, b2, y2, mean2, inv2);
	linear(y2, W1, b1f, h1, S, D, F);
	{
		((float *)g_dims.map)[0] = (float)(S * F);
		run4(&relu, h1, h1, hr, g_dims, (S * F + 63) / 64, 1);
	}
	linear(hr, W2, b2f, h2, S, F, D);
	eadd(h2, y, Out, S * D);

	// ---- backward ----
	// out = h2 + y → dh2 = dOut, dy0 = dOut
	memcpy(dh2.map, dOut.map, (size_t)S * D * sizeof(float));
	memcpy(dy0.map, dOut.map, (size_t)S * D * sizeof(float));
	linbwd(dh2, W2, hr, dhr, dW2, db2f, S, F, D);  // h2 = hr·W2ᵀ+b2f
	{
		((float *)g_dims.map)[0] = (float)(S * F);
		run4(&rbwd, h1, dhr, dh1, g_dims, (S * F + 63) / 64, 1);  // relu
	}
	linbwd(dh1, W1, y2, dy2, dW1, db1f, S, D, F);    // h1 = y2·W1ᵀ+b1f
	lnbwd(y, g2, dy2, dyln, dg2, db2, mean2, inv2);  // y2 = ln(y)
	eadd(dy0, dyln, dy, S * D);                      // dy = residual + ln2
	// y = Ao + X → dAo = dy, dX starts = dy
	memcpy(dAo.map, dy.map, (size_t)S * D * sizeof(float));
	linbwd(dAo, Wo, Aa, dAa, dWo, dbo, S, D, D);  // Ao = Aa·Woᵀ+bo
	attnBwd();                                    // dAa → dQ,dK,dV
	linbwd(dQ, Wq, x2, dx2q, dWq, dbq, S, D, D);
	linbwd(dK, Wk, x2, dx2k, dWk, dbk, S, D, D);
	linbwd(dV, Wv, x2, dx2v, dWv, dbv, S, D, D);
	eadd(dx2q, dx2k, dx2qk, S * D);
	eadd(dx2qk, dx2v, dx2, S * D);                   // dx2 = q+k+v
	lnbwd(X, g1, dx2, dXln, dg1, db1, mean1, inv1);  // x2 = ln(X)
	eadd(dy, dXln, dX, S * D);                       // dX = residual(dy) + ln1

	std::vector<float> out;
	auto push = [&](VkcBuf &b, int n) {
		float *m = (float *)b.map;
		for (int i = 0; i < n; i++) {
			out.push_back(m[i]);
		}
	};
	push(dX, S * D);
	push(dg1, D);
	push(db1, D);
	push(dWq, D * D);
	push(dbq, D);
	push(dWk, D * D);
	push(dbk, D);
	push(dWv, D * D);
	push(dbv, D);
	push(dWo, D * D);
	push(dbo, D);
	push(dg2, D);
	push(db2, D);
	push(dW1, F * D);
	push(db1f, F);
	push(dW2, D * F);
	push(db2f, D);
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(out.data(), sizeof(float), out.size(), o);
	fclose(o);
	printf("encbwd S=%d D=%d H=%d F=%d done\n", S, D, H, F);
	return 0;
}
