// SPDX-License-Identifier: MIT
// Reusable TIC forward (embed -> STACK encoder layers -> two TPM heads), the
// engine-free Slang/Vulkan inference the calibrator deploys with — no Python.
// netfwd_main.cpp is a thin CLI over this; the native driver calibration stage
// (src/sinew_driver.c, C) links the same library to run the net in the deploy path.
//
// C-linkable: the API is extern "C" with pointer (not reference) parameters and an
// int return for the predicate, so sinew_driver.c calls it directly — the only C++
// (vkc, std::vector) stays inside netfwd.cpp.
//
// Backend: GPU via vkc (SPIR-V).  A CPU fallback (slangc -target cpp, the
// soma_pheno_slang pattern) slots in behind this same API.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NetfwdCfg {
	int S, NIN, D, H, F, NOUT, STACK;
} NetfwdCfg;

// Packed weight count for a config (pack_weights.py layout: embed -> STACK+2 encoder
// layers -> 2 head mappings).  The weights buffer passed to netfwd_create has this length.
size_t netfwd_weight_count(const NetfwdCfg *c);

typedef struct NetfwdCtx NetfwdCtx;  // opaque GPU/Slang context (pipelines, scratch, weights)

// Load the 5 kernels (paths in order: gemm, lin, ln, attn, ew) and the packed weights once.
// Returns null on failure.  weights length must be netfwd_weight_count(c).
NetfwdCtx *netfwd_create(const NetfwdCfg *c, const char *spv_paths[5], const float *weights, size_t nweights);

// One window forward.  x: S*NIN floats (per-sensor [accel(3), mag(3), rot(9)], accel already
// unit-normalized per frame to match SINEW_ACCEL=grav training).  out: 2*NOUT floats [global; local].
// Returns 1 on success, 0 on failure.
int netfwd_forward(NetfwdCtx *ctx, const float *x, float *out);

void netfwd_destroy(NetfwdCtx *ctx);

#ifdef __cplusplus
}
#endif
