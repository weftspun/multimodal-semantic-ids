// SPDX-License-Identifier: MIT
// TIC forward as a reusable library (see netfwd.h): embed (linear·√D) -> STACK
// encoder layers -> two TPM heads (layer -> col-mean -> mapping).  The host
// pre-scales embed weights by √D and head mappings by 1/S (so col_sum gives the
// mean).  GPU/Slang via vkc.  Verified vs torch TIC.forward in check_netfwd.py.
#include "netfwd.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "vkc.h"

namespace {
struct LW {
	VkcBuf g1, b1, Wq, bq, Wk, bk, Wv, bv, Wo, bo, g2, b2, W1, b1f, W2, b2f;
};
}  // namespace

struct NetfwdCtx {
	NetfwdCfg c;
	int DK;
	VkcBuf g_dims;
	VkcPipeline nt, badd, csum, lnf, af, padd, relu;
	// shared scratch
	VkcBuf mean, inv, P, z2, z3, z4, z5, z1;
	VkcBuf x2, Qb, Kb, Vb, Aa, Ao, yb, y2, h1, hr, h2;
	VkcBuf Xa, Xb;
	// loaded weights
	VkcBuf xin, We, be, gMap, gMapB, lMap, lMapB;
	std::vector<LW> layers;
	LW gHead, lHead;
	VkcBuf out;

	void setdims(float a, float b, float cc) {
		float *d = (float *)g_dims.map;
		d[0] = a;
		d[1] = b;
		d[2] = cc;
	}
	void linear(VkcBuf X, VkcBuf W, VkcBuf b, VkcBuf Y, int rows, int In, int Out) {
		setdims(rows, Out, In);
		VkcBuf gg[4] = {X, W, Y, g_dims};
		vkc_run(&nt, gg, 4, (Out + 15) / 16, (rows + 15) / 16, 1);
		setdims(rows, Out, 0);
		VkcBuf ba[3] = {Y, b, g_dims};
		vkc_run(&badd, ba, 3, (Out + 15) / 16, (rows + 15) / 16, 1);
	}
	void lnorm(VkcBuf X, VkcBuf g, VkcBuf b, VkcBuf Y) {
		setdims(c.S, c.D, 0);
		VkcBuf bb[7] = {X, g, b, Y, mean, inv, g_dims};
		vkc_run(&lnf, bb, 7, (c.S + 63) / 64, 1, 1);
	}
	void eadd(VkcBuf A, VkcBuf B, VkcBuf C, int n) {
		((float *)g_dims.map)[0] = (float)n;
		VkcBuf bb[4] = {A, B, C, g_dims};
		vkc_run(&padd, bb, 4, (n + 63) / 64, 1, 1);
	}
	// One pre-norm encoder layer: X(S*D) -> outBuf(S*D).
	void layer(VkcBuf X, LW &w, VkcBuf outBuf) {
		const int S = c.S, D = c.D, F = c.F, H = c.H;
		lnorm(X, w.g1, w.b1, x2);
		linear(x2, w.Wq, w.bq, Qb, S, D, D);
		linear(x2, w.Wk, w.bk, Kb, S, D, D);
		linear(x2, w.Wv, w.bv, Vb, S, D, D);
		float *d = (float *)g_dims.map;
		d[0] = (float)S;
		d[1] = (float)D;
		d[2] = (float)H;
		d[3] = (float)DK;
		VkcBuf ab[11] = {Qb, Kb, Vb, Aa, P, z2, z3, z4, z5, z1, g_dims};
		vkc_run(&af, ab, 11, (S + 15) / 16, (H + 15) / 16, 1);
		linear(Aa, w.Wo, w.bo, Ao, S, D, D);
		eadd(Ao, X, yb, S * D);
		lnorm(yb, w.g2, w.b2, y2);
		linear(y2, w.W1, w.b1f, h1, S, D, F);
		((float *)g_dims.map)[0] = (float)(S * F);
		VkcBuf rb[4] = {h1, h1, hr, g_dims};
		vkc_run(&relu, rb, 4, (S * F + 63) / 64, 1, 1);
		linear(hr, w.W2, w.b2f, h2, S, F, D);
		eadd(h2, yb, outBuf, S * D);
	}
};

size_t netfwd_weight_count(const NetfwdCfg *cp) {
	const NetfwdCfg c = *cp;
	int WL = 2 * c.D + 4 * (c.D * c.D + c.D) + 2 * c.D + (c.F * c.D + c.F) + (c.D * c.F + c.D);
	return (size_t)(c.D * c.NIN + c.D) + (size_t)(c.STACK + 2) * WL + (size_t)2 * (c.NOUT * c.D + c.NOUT);
}

NetfwdCtx *netfwd_create(const NetfwdCfg *cp, const char *spv[5], const float *weights, size_t nweights) {
	const NetfwdCfg c = *cp;
	if (nweights != netfwd_weight_count(cp)) {
		return nullptr;
	}
	if (!vkc_init()) {
		return nullptr;
	}
	NetfwdCtx *t = new NetfwdCtx();
	t->c = c;
	t->DK = c.D / c.H;
	const int S = c.S, NIN = c.NIN, D = c.D, F = c.F, NOUT = c.NOUT;

	// Walk the packed weights, pre-scaling embed by √D and head mappings by 1/S.
	const float *p = weights;
	float sqrtD = sqrtf((float)D);
	auto load = [&](int n, float scale) {
		VkcBuf b = vkc_buffer((size_t)n * sizeof(float));
		float *m = (float *)b.map;
		for (int i = 0; i < n; i++) {
			m[i] = p[i] * scale;
		}
		p += n;
		return b;
	};
	auto loadLW = [&]() {
		LW w;
		w.g1 = load(D, 1);
		w.b1 = load(D, 1);
		w.Wq = load(D * D, 1);
		w.bq = load(D, 1);
		w.Wk = load(D * D, 1);
		w.bk = load(D, 1);
		w.Wv = load(D * D, 1);
		w.bv = load(D, 1);
		w.Wo = load(D * D, 1);
		w.bo = load(D, 1);
		w.g2 = load(D, 1);
		w.b2 = load(D, 1);
		w.W1 = load(F * D, 1);
		w.b1f = load(F, 1);
		w.W2 = load(D * F, 1);
		w.b2f = load(D, 1);
		return w;
	};

	t->We = load(D * NIN, sqrtD);
	t->be = load(D, sqrtD);
	for (int l = 0; l < c.STACK; l++) {
		t->layers.push_back(loadLW());
	}
	t->gHead = loadLW();
	t->gMap = load(NOUT * D, 1.0f / S);
	t->gMapB = load(NOUT, 1);
	t->lHead = loadLW();
	t->lMap = load(NOUT * D, 1.0f / S);
	t->lMapB = load(NOUT, 1);

	auto scratch = [&](int n) { return vkc_buffer((size_t)n * sizeof(float)); };
	t->xin = scratch(S * NIN);
	t->g_dims = vkc_buffer(8 * sizeof(float));
	t->mean = scratch(S);
	t->inv = scratch(S);
	t->P = scratch(c.H * S * S);
	t->z1 = scratch(c.H * S * S);
	t->z2 = scratch(S * D);
	t->z3 = scratch(S * D);
	t->z4 = scratch(S * D);
	t->z5 = scratch(S * D);
	t->x2 = scratch(S * D);
	t->Qb = scratch(S * D);
	t->Kb = scratch(S * D);
	t->Vb = scratch(S * D);
	t->Aa = scratch(S * D);
	t->Ao = scratch(S * D);
	t->yb = scratch(S * D);
	t->y2 = scratch(S * D);
	t->h1 = scratch(S * F);
	t->hr = scratch(S * F);
	t->h2 = scratch(S * D);
	t->Xa = scratch(S * D);
	t->Xb = scratch(S * D);
	t->out = vkc_buffer((size_t)2 * NOUT * sizeof(float));

	if (!vkc_pipeline(spv[0], "gemm_nt", 4, &t->nt) || !vkc_pipeline(spv[1], "bias_add", 3, &t->badd) ||
	    !vkc_pipeline(spv[1], "col_sum", 3, &t->csum) || !vkc_pipeline(spv[2], "ln_fwd", 7, &t->lnf) ||
	    !vkc_pipeline(spv[3], "attn_fwd", 11, &t->af) || !vkc_pipeline(spv[4], "add", 4, &t->padd) ||
	    !vkc_pipeline(spv[4], "relu_fwd", 4, &t->relu)) {
		delete t;
		return nullptr;
	}
	return t;
}

int netfwd_forward(NetfwdCtx *t, const float *x, float *out) {
	const int S = t->c.S, NIN = t->c.NIN, D = t->c.D, NOUT = t->c.NOUT;
	memcpy(t->xin.map, x, (size_t)S * NIN * sizeof(float));

	// embed: X = (xin · Weᵀ + be) [We,be already ×√D]
	t->linear(t->xin, t->We, t->be, t->Xa, S, NIN, D);
	VkcBuf cur = t->Xa, nxt = t->Xb;
	for (int l = 0; l < t->c.STACK; l++) {
		t->layer(cur, t->layers[l], nxt);
		VkcBuf tmp = cur;
		cur = nxt;
		nxt = tmp;
	}
	// heads: layer → colmean (col_sum, mapping pre-scaled 1/S) → mapping linear
	auto head = [&](LW &hw, VkcBuf Map, VkcBuf MapB, int outoff) {
		t->layer(cur, hw, nxt);
		t->setdims(S, D, 0);
		VkcBuf cs[3] = {nxt, t->z2, t->g_dims};
		vkc_run(&t->csum, cs, 3, (D + 63) / 64, 1, 1);  // z2[0..D) = Σ_rows
		t->linear(t->z2, Map, MapB, t->z3, 1, D, NOUT);
		float *m = (float *)t->z3.map;
		float *o = (float *)t->out.map;
		for (int j = 0; j < NOUT; j++) {
			o[outoff + j] = m[j];
		}
	};
	head(t->gHead, t->gMap, t->gMapB, 0);
	head(t->lHead, t->lMap, t->lMapB, NOUT);

	memcpy(out, t->out.map, (size_t)2 * NOUT * sizeof(float));
	return true;
}

void netfwd_destroy(NetfwdCtx *t) {
	delete t;  // vkc buffers are process-lifetime mapped; matches the prior exe's leak-on-exit
}
