// SPDX-License-Identifier: MIT
// Tiled production kernels, brick 4: multi-head scaled-dot-product attention
// fwd/bwd over Q,K,V (the linear projections are separate linear bricks).  Head h
// occupies columns [h*DK,(h+1)*DK).  Forward caches the normalised softmax probs P
// for the backward.  Multi-pass (recomputes dot products) so no runtime-sized local
// arrays are needed.  Verified vs torch in check_attn.py.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> Q;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> K;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> V;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Out;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> P;      // H*S*S softmax probs
[[vk::binding(5, 0)]] RWStructuredBuffer<float> dOut;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> dQ;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> dK;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> dV;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> dsim;   // H*S*S
[[vk::binding(10, 0)]] RWStructuredBuffer<float> dims;  // [S, D, H, DK]

[shader("compute")]
[numthreads(16, 16, 1)]
void attn_fwd(uint3 tid: SV_DispatchThreadID) {
    int S = int(dims[0]), D = int(dims[1]), H = int(dims[2]), DK = int(dims[3]);
    int i = int(tid.x), h = int(tid.y);
    if (i >= S || h >= H) {
        return;
    }
    float scale = 1.0f / sqrt((float)DK);
    int base = h * DK;
    float m = -1e30f;
    for (int j = 0; j < S; j++) {
        float d = 0.0f;
        for (int c = 0; c < DK; c++) {
            d += Q[i * D + base + c] * K[j * D + base + c];
        }
        m = max(m, d * scale);
    }
    float s = 0.0f;
    for (int j = 0; j < S; j++) {
        float d = 0.0f;
        for (int c = 0; c < DK; c++) {
            d += Q[i * D + base + c] * K[j * D + base + c];
        }
        float e = exp(d * scale - m);
        P[h * S * S + i * S + j] = e;
        s += e;
    }
    for (int j = 0; j < S; j++) {
        P[h * S * S + i * S + j] /= s;
    }
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) {
            acc += P[h * S * S + i * S + j] * V[j * D + base + c];
        }
        Out[i * D + base + c] = acc;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void attn_bwd_dsim(uint3 tid: SV_DispatchThreadID) {
    int S = int(dims[0]), D = int(dims[1]), H = int(dims[2]), DK = int(dims[3]);
    int i = int(tid.x), h = int(tid.y);
    if (i >= S || h >= H) {
        return;
    }
    float scale = 1.0f / sqrt((float)DK);
    int base = h * DK;
    float ds = 0.0f;
    for (int j = 0; j < S; j++) {
        float dp = 0.0f;
        for (int c = 0; c < DK; c++) {
            dp += dOut[i * D + base + c] * V[j * D + base + c];
        }
        ds += P[h * S * S + i * S + j] * dp;
    }
    for (int j = 0; j < S; j++) {
        float dp = 0.0f;
        for (int c = 0; c < DK; c++) {
            dp += dOut[i * D + base + c] * V[j * D + base + c];
        }
        dsim[h * S * S + i * S + j] = P[h * S * S + i * S + j] * (dp - ds) * scale;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void attn_bwd_dq(uint3 tid: SV_DispatchThreadID) {
    int S = int(dims[0]), D = int(dims[1]), H = int(dims[2]), DK = int(dims[3]);
    int i = int(tid.x), h = int(tid.y);
    if (i >= S || h >= H) {
        return;
    }
    int base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) {
            acc += dsim[h * S * S + i * S + j] * K[j * D + base + c];
        }
        dQ[i * D + base + c] = acc;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void attn_bwd_dk(uint3 tid: SV_DispatchThreadID) {
    int S = int(dims[0]), D = int(dims[1]), H = int(dims[2]), DK = int(dims[3]);
    int j = int(tid.x), h = int(tid.y);
    if (j >= S || h >= H) {
        return;
    }
    int base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++) {
            acc += dsim[h * S * S + i * S + j] * Q[i * D + base + c];
        }
        dK[j * D + base + c] = acc;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void attn_bwd_dv(uint3 tid: SV_DispatchThreadID) {
    int S = int(dims[0]), D = int(dims[1]), H = int(dims[2]), DK = int(dims[3]);
    int j = int(tid.x), h = int(tid.y);
    if (j >= S || h >= H) {
        return;
    }
    int base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++) {
            acc += P[h * S * S + i * S + j] * dOut[i * D + base + c];
        }
        dV[j * D + base + c] = acc;
    }
}
