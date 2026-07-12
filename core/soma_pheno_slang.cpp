// SPDX-License-Identifier: MIT
// Slang→CPU phenotype host: implements the soma_pheno_api.h pheno_eval() by
// running the Lean-emitted Sinew.SlangCodegen.Pheno kernels compiled to the CPU
// (slangc -target cpp).  Drop-in replacement for the hand-written soma_pheno.c —
// the deform math now lives entirely in the Slang kernels (verified in
// spec/tests/pheno_cpu).  The only host code is the tiny phenotype coefficient
// calc and buffer plumbing.
//
// The generated kernels (gen/*.gen.cpp) and the vendored slang C++ prelude
// (slang_rt/) are produced by viz_native/regen_pheno_slang.sh.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "soma_pheno_api.h"
#include "soma_pheno.h"  // PH_* dims + PH_OFF_* byte offsets

#include "slang_rt/slang-cpp-prelude.h"  // once, globally; per-namespace re-includes are guarded out

namespace kblend {
#include "gen/pheno_blendshape.gen.cpp"
}
namespace ktet {
#include "gen/pheno_bary_tet.gen.cpp"
}
namespace kgather {
#include "gen/pheno_bary_gather.gen.cpp"
}
namespace krbf {
#include "gen/pheno_rbf.gen.cpp"
}
namespace kskel {
#include "gen/pheno_skeleton_fit.gen.cpp"
}

// One workgroup range covering n threads, per namespace's ComputeVaryingInput.
template <class CVI>
static CVI dispatch(uint32_t n) {
	CVI vi{};
	vi.startGroupID.x = vi.startGroupID.y = vi.startGroupID.z = 0;
	vi.endGroupID.x = (n + 63) / 64;
	vi.endGroupID.y = 1;
	vi.endGroupID.z = 1;
	return vi;
}

// ── loaded asset (soma_pheno.bin) ────────────────────────────────────────────
static uint8_t *g_blob;
static const float *g_templ, *g_blend, *g_mask, *g_bary, *g_BW, *g_BSh, *g_SW, *g_rbf_val;
static const int32_t *g_F_src, *g_F_tet, *g_fids, *g_par, *g_rbf_crow, *g_rbf_col;
static float *g_vanny, *g_p3, *g_jpos;  // per-eval scratch

extern "C" int pheno_load(const char *path) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		return 1;
	}
	g_blob = (uint8_t *)malloc(PH_BIN_BYTES);
	size_t n = fread(g_blob, 1, PH_BIN_BYTES, f);
	fclose(f);
	if (n != (size_t)PH_BIN_BYTES) {
		return 2;
	}
	g_templ = (const float *)(g_blob + PH_OFF_template);
	g_blend = (const float *)(g_blob + PH_OFF_blend);
	g_mask = (const float *)(g_blob + PH_OFF_mask);
	g_F_src = (const int32_t *)(g_blob + PH_OFF_F_src);
	g_F_tet = (const int32_t *)(g_blob + PH_OFF_F_tet);
	g_bary = (const float *)(g_blob + PH_OFF_bary);
	g_fids = (const int32_t *)(g_blob + PH_OFF_fids);
	g_BW = (const float *)(g_blob + PH_OFF_BW);
	g_BSh = (const float *)(g_blob + PH_OFF_BSh);
	g_SW = (const float *)(g_blob + PH_OFF_SW);
	g_par = (const int32_t *)(g_blob + PH_OFF_par);
	g_rbf_crow = (const int32_t *)(g_blob + PH_OFF_rbf_crow);
	g_rbf_col = (const int32_t *)(g_blob + PH_OFF_rbf_col);
	g_rbf_val = (const float *)(g_blob + PH_OFF_rbf_val);
	g_vanny = (float *)malloc(sizeof(float) * PH_P * 3);
	g_p3 = (float *)malloc(sizeof(float) * PH_NFS * 3);
	g_jpos = (float *)malloc(sizeof(float) * PH_J * 3);
	return 0;
}

// ── phenotype coefficients (host: per-feature anchor interp + multilinear mask) ─
static const float ANC_age[5] = {-1.f / 3.f, 0.f, 1.f / 3.f, 2.f / 3.f, 1.f};
static const float ANC_2[2] = {0.f, 1.f};
static const float ANC_3[3] = {0.f, 0.5f, 1.f};

static void interp(float v, const float *a, int n, float *w) {
	for (int i = 0; i < n; i++) {
		w[i] = 0;
	}
	int idx = 0;
	while (idx < n && a[idx] < v) {
		idx++;
	}
	if (idx < 1) {
		idx = 1;
	}
	if (idx > n - 1) {
		idx = n - 1;
	}
	float al = (v - a[idx - 1]) / (a[idx] - a[idx - 1]);
	if (al < 0) {
		al = 0;
	}
	if (al > 1) {
		al = 1;
	}
	w[idx - 1] = 1 - al;
	w[idx] = al;
}

extern "C" void pheno_eval(const float ident[11], float *v0, float *bw) {
	float wg[2], wa[5], wm[3], ww[3], wh[2], wp[2], wc[3], wfm[3];
	interp(ident[0], ANC_2, 2, wg);
	interp(ident[1], ANC_age, 5, wa);
	interp(ident[2], ANC_3, 3, wm);
	interp(ident[3], ANC_3, 3, ww);
	interp(ident[4], ANC_2, 2, wh);
	interp(ident[5], ANC_2, 2, wp);
	interp(ident[6], ANC_3, 3, wc);
	interp(ident[7], ANC_3, 3, wfm);
	float race[3] = {ident[8], ident[9], ident[10]};
	float rs = race[0] + race[1] + race[2];
	if (rs > 0) {
		for (int i = 0; i < 3; i++) {
			race[i] /= rs;
		}
	} else {
		race[0] = race[1] = race[2] = 1.f / 3.f;
	}
	float phens[26];
	int o = 0;
	for (int i = 0; i < 3; i++) {
		phens[o++] = race[i];
	}
	for (int i = 0; i < 2; i++) {
		phens[o++] = wg[i];
	}
	for (int i = 0; i < 5; i++) {
		phens[o++] = wa[i];
	}
	for (int i = 0; i < 3; i++) {
		phens[o++] = wm[i];
	}
	for (int i = 0; i < 3; i++) {
		phens[o++] = ww[i];
	}
	for (int i = 0; i < 2; i++) {
		phens[o++] = wh[i];
	}
	for (int i = 0; i < 2; i++) {
		phens[o++] = wp[i];
	}
	for (int i = 0; i < 3; i++) {
		phens[o++] = wc[i];
	}
	for (int i = 0; i < 3; i++) {
		phens[o++] = wfm[i];
	}
	static float wi[PH_NB];
	for (int k = 0; k < PH_NB; k++) {
		float pr = 1.f;
		const float *mr = g_mask + k * PH_NVAR;
		for (int j = 0; j < PH_NVAR; j++) {
			pr *= mr[j] != 0.f ? phens[j] * mr[j] + (1.f - mr[j]) : 1.f;
		}
		wi[k] = pr;
	}

	{  // blendshape → vanny
		using namespace kblend;
		GlobalParams_0 gp{};
		gp.templ_0.data = (Vector<float, 3> *)g_templ;
		gp.templ_0.count = PH_P;
		gp.blend_0.data = (Vector<float, 3> *)g_blend;
		gp.blend_0.count = (size_t)PH_NB * PH_P;
		gp.coeffs_0.data = wi;
		gp.coeffs_0.count = PH_NB;
		gp.vanny_0.data = (Vector<float, 3> *)g_vanny;
		gp.vanny_0.count = PH_P;
		ComputeVaryingInput vi = dispatch<ComputeVaryingInput>(PH_P);
		blendshape(&vi, nullptr, &gp);
	}
	{  // bary_tet → p3
		using namespace ktet;
		GlobalParams_0 gp{};
		gp.vanny_0.data = (Vector<float, 3> *)g_vanny;
		gp.vanny_0.count = PH_P;
		gp.Fsrc_0.data = (uint32_t *)g_F_src;
		gp.Fsrc_0.count = (size_t)PH_NFS * 3;
		gp.p3_0.data = (Vector<float, 3> *)g_p3;
		gp.p3_0.count = PH_NFS;
		ComputeVaryingInput vi = dispatch<ComputeVaryingInput>(PH_NFS);
		bary_tet(&vi, nullptr, &gp);
	}
	{  // bary_gather → v0
		using namespace kgather;
		GlobalParams_0 gp{};
		gp.vanny_0.data = (Vector<float, 3> *)g_vanny;
		gp.vanny_0.count = PH_P;
		gp.p3_0.data = (Vector<float, 3> *)g_p3;
		gp.p3_0.count = PH_NFS;
		gp.Ftet_0.data = (uint32_t *)g_F_tet;
		gp.Ftet_0.count = (size_t)PH_NFS * 4;
		gp.fids_0.data = (uint32_t *)g_fids;
		gp.fids_0.count = PH_V;
		gp.bary_0.data = (float *)g_bary;
		gp.bary_0.count = (size_t)PH_V * 4;
		gp.v0_0.data = (Vector<float, 3> *)v0;
		gp.v0_0.count = PH_V;
		ComputeVaryingInput vi = dispatch<ComputeVaryingInput>(PH_V);
		bary_gather(&vi, nullptr, &gp);
	}
	{  // rbf → jpos
		using namespace krbf;
		GlobalParams_0 gp{};
		gp.v0_0.data = (Vector<float, 3> *)v0;
		gp.v0_0.count = PH_V;
		gp.crow_0.data = (uint32_t *)g_rbf_crow;
		gp.crow_0.count = PH_J + 1;
		gp.col_0.data = (uint32_t *)g_rbf_col;
		gp.col_0.count = PH_NNZ;
		gp.val_0.data = (float *)g_rbf_val;
		gp.val_0.count = PH_NNZ;
		gp.jpos_0.data = (Vector<float, 3> *)g_jpos;
		gp.jpos_0.count = PH_J;
		ComputeVaryingInput vi = dispatch<ComputeVaryingInput>(PH_J);
		rbf(&vi, nullptr, &gp);
	}
	{  // skeleton_fit → bind world (single thread)
		using namespace kskel;
		GlobalParams_0 gp{};
		gp.v0_0.data = v0;
		gp.v0_0.count = (size_t)PH_V * 3;
		gp.bsh_0.data = (float *)g_BSh;
		gp.bsh_0.count = (size_t)PH_V * 3;
		gp.bw_0.data = (float *)g_BW;
		gp.bw_0.count = (size_t)PH_J * 16;
		gp.jpos_0.data = g_jpos;
		gp.jpos_0.count = (size_t)PH_J * 3;
		gp.sw_0.data = (float *)g_SW;
		gp.sw_0.count = (size_t)PH_V * PH_J;
		gp.parents_0.data = (int *)g_par;
		gp.parents_0.count = PH_J;
		gp.outbw_0.data = bw;
		gp.outbw_0.count = (size_t)PH_J * 16;
		ComputeVaryingInput vi = dispatch<ComputeVaryingInput>(1);
		skeleton_fit(&vi, nullptr, &gp);
	}
}
