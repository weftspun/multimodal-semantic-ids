// SPDX-License-Identifier: MIT
// M7 step 2: train the engine-free net on the real 15-sensor calibration dataset.
// Same tiled fwd/bwd/Adam as nettrain (verified vs torch), but the (x, target) is
// swapped each step from caldata.bin (gen_caldata.py) rather than a fixed pair;
// weights start from winit.bin.  Reports the training loss curve — it dropping on
// real data is the proof the engine-free trainer learns the calibration.
//   caltrain.exe <gemm><lin><ln><attn><ew><adamn> S NIN D H F NOUT STACK STEPS LR NWIN
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkc.h"

static VkcBuf g_dims;
static int S, NIN, D, H, F, NOUT, STACK, DK;
static VkcPipeline nt, badd, nn, tn, csum, lnf, lndx, lndgdb, af, adsim, adq, adk, adv, padd, relu,
    rbwd, adamp;
static int AGX, AGY;

static void dN(float a, float b, float c, float d) {
	float *p = (float *)g_dims.map;
	p[0] = a;
	p[1] = b;
	p[2] = c;
	p[3] = d;
}
static void R(VkcPipeline *p, std::vector<VkcBuf> b, int gx, int gy) {
	vkc_run(p, b.data(), (uint32_t)b.size(), gx, gy, 1);
}
static void linear(VkcBuf X, VkcBuf W, VkcBuf b, VkcBuf Y, int rows, int In, int Out) {
	dN(rows, Out, In, 0);
	R(&nt, {X, W, Y, g_dims}, (Out + 15) / 16, (rows + 15) / 16);
	dN(rows, Out, 0, 0);
	R(&badd, {Y, b, g_dims}, (Out + 15) / 16, (rows + 15) / 16);
}
static void linbwd(VkcBuf dY, VkcBuf W, VkcBuf X, VkcBuf dX, VkcBuf dW, VkcBuf db, int rows, int In,
                   int Out) {
	dN(rows, In, Out, 0);
	R(&nn, {dY, W, dX, g_dims}, (In + 15) / 16, (rows + 15) / 16);
	dN(Out, In, rows, 0);
	R(&tn, {dY, X, dW, g_dims}, (In + 15) / 16, (Out + 15) / 16);
	dN(rows, Out, 0, 0);
	R(&csum, {dY, db, g_dims}, (Out + 63) / 64, 1);
}
static void lnorm(VkcBuf X, VkcBuf g, VkcBuf b, VkcBuf Y, VkcBuf mn, VkcBuf iv) {
	dN(S, D, 0, 0);
	R(&lnf, {X, g, b, Y, mn, iv, g_dims}, (S + 63) / 64, 1);
}
static void lnbwd(VkcBuf X, VkcBuf g, VkcBuf dY, VkcBuf dX, VkcBuf dg, VkcBuf db, VkcBuf mn,
                  VkcBuf iv) {
	dN(S, D, 0, 0);
	R(&lndx, {X, g, dY, dX, mn, iv, g_dims}, (S + 63) / 64, 1);
	R(&lndgdb, {X, dY, dg, db, mn, iv, g_dims}, (D + 63) / 64, 1);
}
static void eadd(VkcBuf A, VkcBuf B, VkcBuf C, int n) {
	((float *)g_dims.map)[0] = (float)n;
	R(&padd, {A, B, C, g_dims}, (n + 63) / 64, 1);
}

struct LW {
	VkcBuf g1, b1, Wq, bq, Wk, bk, Wv, bv, Wo, bo, g2, b2, W1, b1f, W2, b2f;
};
typedef LW LG;
struct Cache {
	VkcBuf Xin, x2, Q, K, V, P, Aa, y, y2, h1, hr, mn1, iv1, mn2, iv2;
};
static VkcBuf Ao, h2, dh2, dy, dy0, dhr, dh1, dy2, dyln, dAo, dAa, dQ, dK, dV, dsim, scAttn;
static VkcBuf dx2q, dx2k, dx2v, dx2t, dx2, dXln;

static void layerFwd(Cache &c, LW &w, VkcBuf Xin, VkcBuf Out) {
	c.Xin = Xin;
	lnorm(Xin, w.g1, w.b1, c.x2, c.mn1, c.iv1);
	linear(c.x2, w.Wq, w.bq, c.Q, S, D, D);
	linear(c.x2, w.Wk, w.bk, c.K, S, D, D);
	linear(c.x2, w.Wv, w.bv, c.V, S, D, D);
	dN(S, D, H, DK);
	R(&af, {c.Q, c.K, c.V, c.Aa, c.P, scAttn, dQ, dK, dV, dsim, g_dims}, AGX, AGY);
	linear(c.Aa, w.Wo, w.bo, Ao, S, D, D);
	eadd(Ao, Xin, c.y, S * D);
	lnorm(c.y, w.g2, w.b2, c.y2, c.mn2, c.iv2);
	linear(c.y2, w.W1, w.b1f, c.h1, S, D, F);
	((float *)g_dims.map)[0] = (float)(S * F);
	R(&relu, {c.h1, c.h1, c.hr, g_dims}, (S * F + 63) / 64, 1);
	linear(c.hr, w.W2, w.b2f, h2, S, F, D);
	eadd(h2, c.y, Out, S * D);
}
static void layerBwd(Cache &c, LW &w, LG &g, VkcBuf dOut, VkcBuf dXin) {
	memcpy(dh2.map, dOut.map, (size_t)S * D * sizeof(float));
	memcpy(dy0.map, dOut.map, (size_t)S * D * sizeof(float));
	linbwd(dh2, w.W2, c.hr, dhr, g.W2, g.b2f, S, F, D);
	((float *)g_dims.map)[0] = (float)(S * F);
	R(&rbwd, {c.h1, dhr, dh1, g_dims}, (S * F + 63) / 64, 1);
	linbwd(dh1, w.W1, c.y2, dy2, g.W1, g.b1f, S, D, F);
	lnbwd(c.y, w.g2, dy2, dyln, g.g2, g.b2, c.mn2, c.iv2);
	eadd(dy0, dyln, dy, S * D);
	memcpy(dAo.map, dy.map, (size_t)S * D * sizeof(float));
	linbwd(dAo, w.Wo, c.Aa, dAa, g.Wo, g.bo, S, D, D);
	dN(S, D, H, DK);
	R(&adsim, {c.Q, c.K, c.V, scAttn, c.P, dAa, dQ, dK, dV, dsim, g_dims}, AGX, AGY);
	R(&adq, {c.Q, c.K, c.V, scAttn, c.P, dAa, dQ, dK, dV, dsim, g_dims}, AGX, AGY);
	R(&adk, {c.Q, c.K, c.V, scAttn, c.P, dAa, dQ, dK, dV, dsim, g_dims}, AGX, AGY);
	R(&adv, {c.Q, c.K, c.V, scAttn, c.P, dAa, dQ, dK, dV, dsim, g_dims}, AGX, AGY);
	linbwd(dQ, w.Wq, c.x2, dx2q, g.Wq, g.bq, S, D, D);
	linbwd(dK, w.Wk, c.x2, dx2k, g.Wk, g.bk, S, D, D);
	linbwd(dV, w.Wv, c.x2, dx2v, g.Wv, g.bv, S, D, D);
	eadd(dx2q, dx2k, dx2t, S * D);
	eadd(dx2t, dx2v, dx2, S * D);
	lnbwd(c.Xin, w.g1, dx2, dXln, g.g1, g.b1, c.mn1, c.iv1);
	eadd(dy, dXln, dXin, S * D);
}

int main(int argc, char **argv) {
	if (argc < 17) {
		fprintf(stderr,
		        "usage: caltrain <gemm><lin><ln><attn><ew><adamn> S NIN D H F NOUT STACK STEPS LR "
		        "NWIN\n");
		return 2;
	}
	S = atoi(argv[7]);
	NIN = atoi(argv[8]);
	D = atoi(argv[9]);
	H = atoi(argv[10]);
	F = atoi(argv[11]);
	NOUT = atoi(argv[12]);
	STACK = atoi(argv[13]);
	int STEPS = atoi(argv[14]);
	float LR = (float)atof(argv[15]);
	int NWIN = atoi(argv[16]);
	DK = D / H;
	if (!vkc_init()) {
		return 1;
	}
	int WL = 2 * D + 4 * (D * D + D) + 2 * D + (F * D + F) + (D * F + D);
	int NW = D * NIN + D + (STACK + 2) * WL + 2 * (NOUT * D + NOUT);
	std::vector<float> in(NW);
	FILE *fw = fopen("winit.bin", "rb");
	if (!fw || fread(in.data(), sizeof(float), in.size(), fw) != in.size()) {
		fprintf(stderr, "caltrain: cannot read winit.bin\n");
		return 1;
	}
	fclose(fw);
	std::vector<float> X((size_t)NWIN * S * NIN), Y((size_t)NWIN * 2 * NOUT);
	FILE *fd = fopen("caldata.bin", "rb");
	if (!fd || fread(X.data(), sizeof(float), X.size(), fd) != X.size() ||
	    fread(Y.data(), sizeof(float), Y.size(), fd) != Y.size()) {
		fprintf(stderr, "caltrain: cannot read caldata.bin\n");
		return 1;
	}
	fclose(fd);
	float sqrtD = sqrtf((float)D);
	float *p = in.data();
	auto sc = [&](int n) { return vkc_buffer((size_t)n * sizeof(float)); };
	auto load = [&](int n) {
		VkcBuf b = sc(n);
		memcpy(b.map, p, (size_t)n * sizeof(float));
		p += n;
		return b;
	};
	g_dims = vkc_buffer(8 * sizeof(float));
	AGX = (S + 15) / 16;
	AGY = (H + 15) / 16;

	VkcBuf xin = sc(S * NIN);
	VkcBuf We = load(D * NIN), be = load(D);
	auto loadLW = [&]() {
		LW w;
		w.g1 = load(D);
		w.b1 = load(D);
		w.Wq = load(D * D);
		w.bq = load(D);
		w.Wk = load(D * D);
		w.bk = load(D);
		w.Wv = load(D * D);
		w.bv = load(D);
		w.Wo = load(D * D);
		w.bo = load(D);
		w.g2 = load(D);
		w.b2 = load(D);
		w.W1 = load(F * D);
		w.b1f = load(F);
		w.W2 = load(D * F);
		w.b2f = load(D);
		return w;
	};
	std::vector<LW> L;
	for (int i = 0; i < STACK; i++) {
		L.push_back(loadLW());
	}
	LW gH = loadLW();
	VkcBuf gMap = load(NOUT * D), gMapB = load(NOUT);
	LW lH = loadLW();
	VkcBuf lMap = load(NOUT * D), lMapB = load(NOUT);

	struct AW {
		VkcBuf W, G, M, V;
		int n;
	};
	std::vector<AW> aws;
	auto reg = [&](VkcBuf W, int n) {
		AW a{W, sc(n), sc(n), sc(n), n};
		aws.push_back(a);
		return a.G;
	};
	auto regLW = [&](LW &w) {
		LG g;
		g.g1 = reg(w.g1, D);
		g.b1 = reg(w.b1, D);
		g.Wq = reg(w.Wq, D * D);
		g.bq = reg(w.bq, D);
		g.Wk = reg(w.Wk, D * D);
		g.bk = reg(w.bk, D);
		g.Wv = reg(w.Wv, D * D);
		g.bv = reg(w.bv, D);
		g.Wo = reg(w.Wo, D * D);
		g.bo = reg(w.bo, D);
		g.g2 = reg(w.g2, D);
		g.b2 = reg(w.b2, D);
		g.W1 = reg(w.W1, F * D);
		g.b1f = reg(w.b1f, F);
		g.W2 = reg(w.W2, D * F);
		g.b2f = reg(w.b2f, D);
		return g;
	};
	VkcBuf dWe = reg(We, D * NIN), dbe = reg(be, D);
	std::vector<LG> LGs;
	for (int i = 0; i < STACK; i++) {
		LGs.push_back(regLW(L[i]));
	}
	LG gHG = regLW(gH);
	VkcBuf dgMap = reg(gMap, NOUT * D), dgMapB = reg(gMapB, NOUT);
	LG lHG = regLW(lH);
	VkcBuf dlMap = reg(lMap, NOUT * D), dlMapB = reg(lMapB, NOUT);

	auto mkC = [&]() {
		Cache c;
		c.x2 = sc(S * D);
		c.Q = sc(S * D);
		c.K = sc(S * D);
		c.V = sc(S * D);
		c.P = sc(H * S * S);
		c.Aa = sc(S * D);
		c.y = sc(S * D);
		c.y2 = sc(S * D);
		c.h1 = sc(S * F);
		c.hr = sc(S * F);
		c.mn1 = sc(S);
		c.iv1 = sc(S);
		c.mn2 = sc(S);
		c.iv2 = sc(S);
		return c;
	};
	std::vector<Cache> C;
	for (int i = 0; i < STACK + 2; i++) {
		C.push_back(mkC());
	}
	std::vector<VkcBuf> Xs;
	for (int i = 0; i < STACK + 1; i++) {
		Xs.push_back(sc(S * D));
	}
	VkcBuf Yg = sc(S * D), Yl = sc(S * D), sumg = sc(D), suml = sc(D), meang = sc(D), meanl = sc(D);
	VkcBuf og = sc(NOUT), ol = sc(NOUT), dog = sc(NOUT), dol = sc(NOUT);
	VkcBuf dmeang = sc(D), dmeanl = sc(D), dYg = sc(S * D), dYl = sc(S * D);
	VkcBuf dXg = sc(S * D), dXl = sc(S * D), dXbb = sc(S * D), dcur = sc(S * D), dnext = sc(S * D);
	VkcBuf dxinScratch = sc(S * NIN);
	Ao = sc(S * D);
	h2 = sc(S * D);
	dh2 = sc(S * D);
	dy = sc(S * D);
	dy0 = sc(S * D);
	dhr = sc(S * F);
	dh1 = sc(S * F);
	dy2 = sc(S * D);
	dyln = sc(S * D);
	dAo = sc(S * D);
	dAa = sc(S * D);
	dQ = sc(S * D);
	dK = sc(S * D);
	dV = sc(S * D);
	dsim = sc(H * S * S);
	scAttn = sc(S * D);
	dx2q = sc(S * D);
	dx2k = sc(S * D);
	dx2v = sc(S * D);
	dx2t = sc(S * D);
	dx2 = sc(S * D);
	dXln = sc(S * D);
	VkcBuf aparams = vkc_buffer(8 * sizeof(float));

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
	    !vkc_pipeline(argv[5], "relu_bwd", 4, &rbwd) ||
	    !vkc_pipeline(argv[6], "adam_n", 5, &adamp)) {
		return 1;
	}

	std::vector<float> losses;
	for (int step = 1; step <= STEPS; step++) {
		int idx = (step - 1) % NWIN;
		memcpy(xin.map, &X[(size_t)idx * S * NIN], (size_t)S * NIN * sizeof(float));
		const float *tgt = &Y[(size_t)idx * 2 * NOUT];

		linear(xin, We, be, Xs[0], S, NIN, D);
		{
			float *m = (float *)Xs[0].map;
			for (int i = 0; i < S * D; i++) {
				m[i] *= sqrtD;
			}
		}
		for (int l = 0; l < STACK; l++) {
			layerFwd(C[l], L[l], Xs[l], Xs[l + 1]);
		}
		auto headFwd = [&](Cache &c, LW &w, VkcBuf Yb, VkcBuf sum, VkcBuf meanv, VkcBuf Map,
		                   VkcBuf MapB, VkcBuf o) {
			layerFwd(c, w, Xs[STACK], Yb);
			dN(S, D, 0, 0);
			R(&csum, {Yb, sum, g_dims}, (D + 63) / 64, 1);
			float *sm = (float *)sum.map, *mv = (float *)meanv.map;
			for (int j = 0; j < D; j++) {
				mv[j] = sm[j] / S;
			}
			linear(meanv, Map, MapB, o, 1, D, NOUT);
		};
		headFwd(C[STACK], gH, Yg, sumg, meang, gMap, gMapB, og);
		headFwd(C[STACK + 1], lH, Yl, suml, meanl, lMap, lMapB, ol);

		float *ogm = (float *)og.map, *olm = (float *)ol.map;
		float *dogm = (float *)dog.map, *dolm = (float *)dol.map;
		float loss = 0;
		for (int j = 0; j < NOUT; j++) {
			float eg = ogm[j] - tgt[j], el = olm[j] - tgt[NOUT + j];
			loss += eg * eg + el * el;
			dogm[j] = eg / NOUT;
			dolm[j] = el / NOUT;
		}
		losses.push_back(loss / (2 * NOUT));

		auto headBwd = [&](Cache &c, LW &w, LG &g, VkcBuf do_, VkcBuf meanv, VkcBuf Map,
		                   VkcBuf MapB, VkcBuf dMap, VkcBuf dMapB, VkcBuf dmean, VkcBuf dY,
		                   VkcBuf dX) {
			linbwd(do_, Map, meanv, dmean, dMap, dMapB, 1, D, NOUT);
			float *dm = (float *)dmean.map, *dy_ = (float *)dY.map;
			for (int s = 0; s < S; s++) {
				for (int j = 0; j < D; j++) {
					dy_[s * D + j] = dm[j] / S;
				}
			}
			layerBwd(c, w, g, dY, dX);
		};
		headBwd(C[STACK], gH, gHG, dog, meang, gMap, gMapB, dgMap, dgMapB, dmeang, dYg, dXg);
		headBwd(C[STACK + 1], lH, lHG, dol, meanl, lMap, lMapB, dlMap, dlMapB, dmeanl, dYl, dXl);
		eadd(dXg, dXl, dXbb, S * D);
		memcpy(dcur.map, dXbb.map, (size_t)S * D * sizeof(float));
		for (int l = STACK - 1; l >= 0; l--) {
			layerBwd(C[l], L[l], LGs[l], dcur, dnext);
			memcpy(dcur.map, dnext.map, (size_t)S * D * sizeof(float));
		}
		{
			float *m = (float *)dcur.map;
			for (int i = 0; i < S * D; i++) {
				m[i] *= sqrtD;
			}
		}
		linbwd(dcur, We, xin, dxinScratch, dWe, dbe, S, NIN, D);

		float *ap = (float *)aparams.map;
		ap[0] = (float)step;
		ap[1] = LR;
		ap[2] = 0.9f;
		ap[3] = 0.999f;
		ap[4] = 1e-8f;
		for (AW &a : aws) {
			ap[5] = (float)a.n;
			vkc_run(&adamp, (VkcBuf[5]){a.W, a.G, a.M, a.V, aparams}, 5, (a.n + 63) / 64, 1, 1);
		}
		if (step % 200 == 0 || step == 1) {
			printf("step %5d  loss %.5f\n", step, losses.back());
		}
	}
	FILE *o = fopen("losses.bin", "wb");
	fwrite(losses.data(), sizeof(float), losses.size(), o);
	fclose(o);
	// trained weights, in the same order as winit.bin / netfwd reads (for inference)
	FILE *wf = fopen("wtrained.bin", "wb");
	for (AW &a : aws) {
		fwrite(a.W.map, sizeof(float), (size_t)a.n, wf);
	}
	fclose(wf);
	printf("caltrain %d steps; loss %.5f -> %.5f; wrote wtrained.bin\n", STEPS, losses.front(),
	       losses.back());
	return 0;
}
