// SPDX-License-Identifier: MIT
// Tiled linear layer driver: orchestrates gemm + bias kernels into a full linear
// fwd/bwd (Y=X·Wᵀ+b; dX=dY·W; dW=dYᵀ·X; db=Σ dY) over persistent buffers, the
// pattern the production trainer uses per layer.  Verified by check_lin.py.
//   lin.exe <gemm.spv> <lin.spv> <S> <In> <Out>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

static VkcBuf g_dims;
static void setdims(float a, float b, float c) {
	float *d = (float *)g_dims.map;
	d[0] = a;
	d[1] = b;
	d[2] = c;
}

int main(int argc, char **argv) {
	if (argc < 6) {
		fprintf(stderr, "usage: lin <gemm.spv> <lin.spv> <S> <In> <Out>\n");
		return 2;
	}
	int S = atoi(argv[3]), In = atoi(argv[4]), Out = atoi(argv[5]);
	if (!vkc_init()) {
		return 1;
	}
	std::vector<float> in(S * In + Out * In + Out + S * Out);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "lin: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);

	VkcBuf X = vkc_buffer((size_t)S * In * sizeof(float));
	VkcBuf W = vkc_buffer((size_t)Out * In * sizeof(float));
	VkcBuf b = vkc_buffer((size_t)Out * sizeof(float));
	VkcBuf dY = vkc_buffer((size_t)S * Out * sizeof(float));
	VkcBuf Y = vkc_buffer((size_t)S * Out * sizeof(float));
	VkcBuf dX = vkc_buffer((size_t)S * In * sizeof(float));
	VkcBuf dW = vkc_buffer((size_t)Out * In * sizeof(float));
	VkcBuf db = vkc_buffer((size_t)Out * sizeof(float));
	g_dims = vkc_buffer(4 * sizeof(float));
	float *p = in.data();
	memcpy(X.map, p, (size_t)S * In * sizeof(float));
	p += S * In;
	memcpy(W.map, p, (size_t)Out * In * sizeof(float));
	p += Out * In;
	memcpy(b.map, p, (size_t)Out * sizeof(float));
	p += Out;
	memcpy(dY.map, p, (size_t)S * Out * sizeof(float));

	VkcPipeline nt{}, nn{}, tn{}, badd{}, csum{};
	if (!vkc_pipeline(argv[1], "gemm_nt", 4, &nt) || !vkc_pipeline(argv[1], "gemm_nn", 4, &nn) ||
	    !vkc_pipeline(argv[1], "gemm_tn", 4, &tn) || !vkc_pipeline(argv[2], "bias_add", 3, &badd) ||
	    !vkc_pipeline(argv[2], "col_sum", 3, &csum)) {
		return 1;
	}
	int gx, gy;
	// forward: Y = X(S,In) · W(Out,In)ᵀ  → (S,Out), then + b
	setdims(S, Out, In);
	{
		VkcBuf bb[4] = {X, W, Y, g_dims};
		vkc_run(&nt, bb, 4, (Out + 15) / 16, (S + 15) / 16, 1);
	}
	{
		VkcBuf bb[3] = {Y, b, g_dims};
		vkc_run(&badd, bb, 3, (Out + 15) / 16, (S + 15) / 16, 1);
	}
	// dX = dY(S,Out) · W(Out,In) → (S,In)
	setdims(S, In, Out);
	{
		VkcBuf bb[4] = {dY, W, dX, g_dims};
		vkc_run(&nn, bb, 4, (In + 15) / 16, (S + 15) / 16, 1);
	}
	// dW = dY(S,Out)ᵀ · X(S,In) → (Out,In)
	setdims(Out, In, S);
	{
		VkcBuf bb[4] = {dY, X, dW, g_dims};
		vkc_run(&tn, bb, 4, (In + 15) / 16, (Out + 15) / 16, 1);
	}
	// db = Σ_rows dY
	setdims(S, Out, 0);
	{
		VkcBuf bb[3] = {dY, db, g_dims};
		vkc_run(&csum, bb, 3, (Out + 63) / 64, 1, 1);
	}

	std::vector<float> out;
	auto push = [&](VkcBuf &buf, int n) {
		float *m = (float *)buf.map;
		for (int i = 0; i < n; i++) {
			out.push_back(m[i]);
		}
	};
	push(Y, S * Out);
	push(dX, S * In);
	push(dW, Out * In);
	push(db, Out);
	FILE *o = fopen("outputs.bin", "wb");
	fwrite(out.data(), sizeof(float), out.size(), o);
	fclose(o);
	(void)gx;
	(void)gy;
	printf("linear %dx%dx%d done\n", S, In, Out);
	return 0;
}
