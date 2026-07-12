// SPDX-License-Identifier: MIT
// Tiled production kernels, brick 3: layernorm fwd/bwd (eps 1e-5, matching torch).
// Forward caches per-row mean/inv for the backward.  dx is a per-row kernel; dg/db
// are per-feature reductions over rows.  Verified vs torch in check_ln.py.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> A;     // x
[[vk::binding(1, 0)]] RWStructuredBuffer<float> B;     // fwd: g  | bwd_dx: g  | dgdb: dy
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;     // fwd: b  | bwd_dx: dy | dgdb: dg
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Dbuf;  // fwd: y  | bwd_dx: dx | dgdb: db
[[vk::binding(4, 0)]] RWStructuredBuffer<float> mean;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> inv;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> dims;  // [R, D]

[shader("compute")]
[numthreads(64, 1, 1)]
void ln_fwd(uint3 tid: SV_DispatchThreadID) {  // A=x B=g C=b Dbuf=y
    int R = int(dims[0]), D = int(dims[1]);
    int r = int(tid.x);
    if (r >= R) {
        return;
    }
    float m = 0.0f;
    for (int i = 0; i < D; i++) {
        m += A[r * D + i];
    }
    m /= D;
    float v = 0.0f;
    for (int i = 0; i < D; i++) {
        v += (A[r * D + i] - m) * (A[r * D + i] - m);
    }
    v /= D;
    float iv = 1.0f / sqrt(v + 1e-5f);
    mean[r] = m;
    inv[r] = iv;
    for (int i = 0; i < D; i++) {
        Dbuf[r * D + i] = (A[r * D + i] - m) * iv * B[i] + C[i];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ln_bwd_dx(uint3 tid: SV_DispatchThreadID) {  // A=x B=g C=dy Dbuf=dx
    int R = int(dims[0]), D = int(dims[1]);
    int r = int(tid.x);
    if (r >= R) {
        return;
    }
    float m = mean[r], iv = inv[r];
    float s1 = 0.0f, s2 = 0.0f;
    for (int j = 0; j < D; j++) {
        float xhat = (A[r * D + j] - m) * iv;
        float dyg = C[r * D + j] * B[j];
        s1 += dyg;
        s2 += dyg * xhat;
    }
    for (int i = 0; i < D; i++) {
        float xhat = (A[r * D + i] - m) * iv;
        float dyg = C[r * D + i] * B[i];
        Dbuf[r * D + i] = iv / D * (D * dyg - s1 - xhat * s2);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ln_bwd_dgdb(uint3 tid: SV_DispatchThreadID) {  // A=x B=dy C=dg Dbuf=db
    int R = int(dims[0]), D = int(dims[1]);
    int i = int(tid.x);
    if (i >= D) {
        return;
    }
    float sdg = 0.0f, sdb = 0.0f;
    for (int r = 0; r < R; r++) {
        float xhat = (A[r * D + i] - mean[r]) * inv[r];
        sdg += B[r * D + i] * xhat;
        sdb += B[r * D + i];
    }
    C[i] = sdg;
    Dbuf[i] = sdb;
}
