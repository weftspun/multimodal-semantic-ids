// SPDX-License-Identifier: MIT
// Milestone 2: differentiable primitives in Slang, gradient-checked against
// torch.autograd (check_prims.py).  Each entry reads a flat input buffer and
// writes a flat output buffer; one thread runs the small test case (forward,
// then reverse-mode for the input gradients).
[[vk::binding(0, 0)]] RWStructuredBuffer<float> inB;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outB;

// ---- linear: y = x·Wᵀ + b ----  in [x(B*N) W(M*N) b(M) dy(B*M)]  out [y dx dW db]
static const int B = 2;
static const int N = 4;
static const int M = 3;

[Differentiable]
void linear(float x[B * N], float W[M * N], float b[M], out float y[B * M]) {
    [ForceUnroll]
    for (int i = 0; i < B; i++) {
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            float s = b[j];
            [ForceUnroll]
            for (int k = 0; k < N; k++) {
                s += x[i * N + k] * W[j * N + k];
            }
            y[i * M + j] = s;
        }
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void linear_test(uint3 tid: SV_DispatchThreadID) {
    float x[B * N], W[M * N], b[M], dy[B * M];
    int o = 0;
    [ForceUnroll]
    for (int i = 0; i < B * N; i++) {
        x[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < M * N; i++) {
        W[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < M; i++) {
        b[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < B * M; i++) {
        dy[i] = inB[o++];
    }

    float y[B * M];
    linear(x, W, b, y);
    var dpx = diffPair(x);
    var dpW = diffPair(W);
    var dpb = diffPair(b);
    bwd_diff(linear)(dpx, dpW, dpb, dy);

    int w = 0;
    [ForceUnroll]
    for (int i = 0; i < B * M; i++) {
        outB[w++] = y[i];
    }
    [ForceUnroll]
    for (int i = 0; i < B * N; i++) {
        outB[w++] = dpx.d[i];
    }
    [ForceUnroll]
    for (int i = 0; i < M * N; i++) {
        outB[w++] = dpW.d[i];
    }
    [ForceUnroll]
    for (int i = 0; i < M; i++) {
        outB[w++] = dpb.d[i];
    }
}

// ---- 6D → rotation (Zhou 2019 Gram-Schmidt) ----  in [d(6) dR(9)]  out [R(9) dd(6)]
// R columns b1,b2,b3; flat row-major R[r*3+c].  The calibrator's output rep.
[Differentiable]
float3 nrm(float3 v) { return v / sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

[Differentiable]
void sixd_to_rotmat(float d[6], out float R[9]) {
    float3 a1 = float3(d[0], d[1], d[2]);
    float3 a2 = float3(d[3], d[4], d[5]);
    float3 b1 = nrm(a1);
    float3 b2 = nrm(a2 - dot(b1, a2) * b1);
    float3 b3 = cross(b1, b2);
    R[0] = b1.x; R[1] = b2.x; R[2] = b3.x;
    R[3] = b1.y; R[4] = b2.y; R[5] = b3.y;
    R[6] = b1.z; R[7] = b2.z; R[8] = b3.z;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void sixd_test(uint3 tid: SV_DispatchThreadID) {
    float d[6], dR[9];
    [ForceUnroll]
    for (int i = 0; i < 6; i++) {
        d[i] = inB[i];
    }
    [ForceUnroll]
    for (int i = 0; i < 9; i++) {
        dR[i] = inB[6 + i];
    }
    float R[9];
    sixd_to_rotmat(d, R);
    var dpd = diffPair(d);
    bwd_diff(sixd_to_rotmat)(dpd, dR);
    [ForceUnroll]
    for (int i = 0; i < 9; i++) {
        outB[i] = R[i];
    }
    [ForceUnroll]
    for (int i = 0; i < 6; i++) {
        outB[9 + i] = dpd.d[i];
    }
}

// ---- vector primitives over LN features ----
static const int LN = 8;

// GELU (tanh approximation, matching torch approximate='tanh').
[Differentiable]
float gelu(float x) {
    float c = 0.7978845608028654f;  // sqrt(2/pi)
    float u = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(u));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void gelu_test(uint3 tid: SV_DispatchThreadID) {  // in [x(LN) dy(LN)]  out [y dx]
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        float x = inB[i];
        var dpx = diffPair(x);
        bwd_diff(gelu)(dpx, inB[LN + i]);
        outB[i] = gelu(x);
        outB[LN + i] = dpx.d;
    }
}

// LayerNorm over LN features (eps 1e-5, matching torch default).
[Differentiable]
void layernorm(float x[LN], float g[LN], float b[LN], out float y[LN]) {
    float mean = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        mean += x[i];
    }
    mean /= LN;
    float var = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        var += (x[i] - mean) * (x[i] - mean);
    }
    var /= LN;
    float inv = 1.0f / sqrt(var + 1e-5f);
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        y[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layernorm_test(uint3 tid: SV_DispatchThreadID) {  // in [x g b dy]  out [y dx dg db]
    float x[LN], g[LN], b[LN], dy[LN];
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        x[i] = inB[i];
        g[i] = inB[LN + i];
        b[i] = inB[2 * LN + i];
        dy[i] = inB[3 * LN + i];
    }
    float y[LN];
    layernorm(x, g, b, y);
    var dpx = diffPair(x);
    var dpg = diffPair(g);
    var dpb = diffPair(b);
    bwd_diff(layernorm)(dpx, dpg, dpb, dy);
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        outB[i] = y[i];
        outB[LN + i] = dpx.d[i];
        outB[2 * LN + i] = dpg.d[i];
        outB[3 * LN + i] = dpb.d[i];
    }
}

// Softmax over LN (max-subtraction omitted; same result/grad for the test inputs).
[Differentiable]
void softmax(float x[LN], out float y[LN]) {
    float s = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        s += exp(x[i]);
    }
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        y[i] = exp(x[i]) / s;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void softmax_test(uint3 tid: SV_DispatchThreadID) {  // in [x(LN) dy(LN)]  out [y dx]
    float x[LN], dy[LN];
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        x[i] = inB[i];
        dy[i] = inB[LN + i];
    }
    float y[LN];
    softmax(x, y);
    var dpx = diffPair(x);
    bwd_diff(softmax)(dpx, dy);
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        outB[i] = y[i];
        outB[LN + i] = dpx.d[i];
    }
}

// MSE loss = mean((p - t)^2).
[Differentiable]
float mse(float p[LN], float t[LN]) {
    float s = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        s += (p[i] - t[i]) * (p[i] - t[i]);
    }
    return s / LN;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mse_test(uint3 tid: SV_DispatchThreadID) {  // in [p(LN) t(LN)]  out [loss(1) dp(LN)]
    float p[LN], t[LN];
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        p[i] = inB[i];
        t[i] = inB[LN + i];
    }
    var dpp = diffPair(p);
    var dpt = diffPair(t);
    bwd_diff(mse)(dpp, dpt, 1.0f);
    outB[0] = mse(p, t);
    [ForceUnroll]
    for (int i = 0; i < LN; i++) {
        outB[1 + i] = dpp.d[i];
    }
}
