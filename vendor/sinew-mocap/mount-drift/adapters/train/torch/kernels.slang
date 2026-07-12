// SPDX-License-Identifier: MIT
// CUDA (slangtorch) entry points for the TIC training bricks — the SAME math as the Vulkan
// kernels in slangtrain/*.slang, re-bound for PyTorch: TensorView<float> params + flat 1-D
// indexing (tensors are passed reshaped to 1-D contiguous) instead of [[vk::binding]]
// RWStructuredBuffer + a dims storage buffer.  Scalar dims become real kernel parameters.
// Parity vs the Vulkan path and vs torch is gated in check_tk.py; that test is the anti-drift
// guarantee between this file and the Vulkan source it mirrors.

// ── GEMM (brick 1, mirrors gemm.slang) ───────────────────────────────────────
//   nn: C = A·B    (dX = dY·W)     A is M×K, B is K×N
//   nt: C = A·Bᵀ   (Y  = X·Wᵀ)     A is M×K, B is N×K
//   tn: C = Aᵀ·B   (dW = dYᵀ·X)    A is K×M, B is K×N
// grid: x covers N (col), y covers M (row), 16×16 block — same as the Vulkan dispatch.
[AutoPyBindCUDA]
[CudaKernel]
void gemm_nn(TensorView<float> A, TensorView<float> B, TensorView<float> C, int M, int N, int K) {
    uint row = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint col = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (row >= M || col >= N) { return; }
    float acc = 0.0f;
    for (int k = 0; k < K; k++) { acc += A[row * K + k] * B[k * N + col]; }
    C[row * N + col] = acc;
}

[AutoPyBindCUDA]
[CudaKernel]
void gemm_nt(TensorView<float> A, TensorView<float> B, TensorView<float> C, int M, int N, int K) {
    uint row = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint col = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (row >= M || col >= N) { return; }
    float acc = 0.0f;
    for (int k = 0; k < K; k++) { acc += A[row * K + k] * B[col * K + k]; }
    C[row * N + col] = acc;
}

[AutoPyBindCUDA]
[CudaKernel]
void gemm_tn(TensorView<float> A, TensorView<float> B, TensorView<float> C, int M, int N, int K) {
    uint row = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint col = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (row >= M || col >= N) { return; }
    float acc = 0.0f;
    for (int k = 0; k < K; k++) { acc += A[k * M + row] * B[k * N + col]; }
    C[row * N + col] = acc;
}

// ── tiled GEMM (T8: shared-memory reuse; same result as gemm_nt) ──────────────
// C = A·Bᵀ, A:M×K, B:N×K. 16×16 tiles staged in groupshared cut global loads ~16×, closing
// most of the naive kernel's gap to cuBLAS at large dims (no tensor cores; fp32 CUDA cores).
static const int TILE = 16;
groupshared float As[TILE][TILE];
groupshared float Bs[TILE][TILE];

[AutoPyBindCUDA]
[CudaKernel]
void gemm_nt_tiled(TensorView<float> A, TensorView<float> B, TensorView<float> C, int M, int N, int K) {
    uint tx = cudaThreadIdx().x, ty = cudaThreadIdx().y;
    uint row = cudaBlockIdx().y * TILE + ty;
    uint col = cudaBlockIdx().x * TILE + tx;
    float acc = 0.0f;
    int ntile = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntile; t++) {
        int aK = t * TILE + tx;
        int bK = t * TILE + ty;
        As[ty][tx] = (row < M && aK < K) ? A[row * K + aK] : 0.0f;       // A[row, t*TILE+tx]
        Bs[ty][tx] = (col < N && bK < K) ? B[col * K + bK] : 0.0f;       // B[col, t*TILE+ty] (Bᵀ)
        GroupMemoryBarrierWithGroupSync();
        for (int k = 0; k < TILE; k++) { acc += As[ty][k] * Bs[k][tx]; }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) { C[row * N + col] = acc; }
}

// ── linear pieces (brick 2, mirrors lin.slang) ───────────────────────────────
[AutoPyBindCUDA]
[CudaKernel]
void bias_add(TensorView<float> Y, TensorView<float> vec, int rows, int cols) {  // Y[r,c] += vec[c]
    uint r = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint c = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (r >= rows || c >= cols) { return; }
    Y[r * cols + c] = Y[r * cols + c] + vec[c];
}

[AutoPyBindCUDA]
[CudaKernel]
void col_sum(TensorView<float> Y, TensorView<float> vec, int rows, int cols) {  // vec[c] = Σ_r Y[r,c]
    uint c = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (c >= cols) { return; }
    float s = 0.0f;
    for (int r = 0; r < rows; r++) { s += Y[r * cols + c]; }
    vec[c] = s;
}

// ── layernorm (brick 3, mirrors ln.slang; eps 1e-5) ──────────────────────────
[AutoPyBindCUDA]
[CudaKernel]
void ln_fwd(TensorView<float> x, TensorView<float> g, TensorView<float> b, TensorView<float> y,
            TensorView<float> mean, TensorView<float> inv, int R, int D) {
    uint r = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (r >= R) { return; }
    float m = 0.0f;
    for (int i = 0; i < D; i++) { m += x[r * D + i]; }
    m /= D;
    float v = 0.0f;
    for (int i = 0; i < D; i++) { v += (x[r * D + i] - m) * (x[r * D + i] - m); }
    v /= D;
    float iv = 1.0f / sqrt(v + 1e-5f);
    mean[r] = m;
    inv[r] = iv;
    for (int i = 0; i < D; i++) { y[r * D + i] = (x[r * D + i] - m) * iv * g[i] + b[i]; }
}

[AutoPyBindCUDA]
[CudaKernel]
void ln_bwd_dx(TensorView<float> x, TensorView<float> g, TensorView<float> dy, TensorView<float> dx,
               TensorView<float> mean, TensorView<float> inv, int R, int D) {
    uint r = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (r >= R) { return; }
    float m = mean[r], iv = inv[r];
    float s1 = 0.0f, s2 = 0.0f;
    for (int j = 0; j < D; j++) {
        float xhat = (x[r * D + j] - m) * iv;
        float dyg = dy[r * D + j] * g[j];
        s1 += dyg;
        s2 += dyg * xhat;
    }
    for (int i = 0; i < D; i++) {
        float xhat = (x[r * D + i] - m) * iv;
        float dyg = dy[r * D + i] * g[i];
        dx[r * D + i] = iv / D * (D * dyg - s1 - xhat * s2);
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void ln_bwd_dgdb(TensorView<float> x, TensorView<float> dy, TensorView<float> dg, TensorView<float> db,
                 TensorView<float> mean, TensorView<float> inv, int R, int D) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= D) { return; }
    float sdg = 0.0f, sdb = 0.0f;
    for (int r = 0; r < R; r++) {
        float xhat = (x[r * D + i] - mean[r]) * inv[r];
        sdg += dy[r * D + i] * xhat;
        sdb += dy[r * D + i];
    }
    dg[i] = sdg;
    db[i] = sdb;
}

// ── multi-head attention (brick 4, mirrors attn.slang) ───────────────────────
// grid: x covers S, y covers H.  P caches normalized softmax probs (H*S*S).
[AutoPyBindCUDA]
[CudaKernel]
void attn_fwd(TensorView<float> Q, TensorView<float> K, TensorView<float> V, TensorView<float> Out,
              TensorView<float> P, int S, int D, int H, int DK) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (i >= S || h >= H) { return; }
    float scale = 1.0f / sqrt((float)DK);
    int base = h * DK;
    float m = -1e30f;
    for (int j = 0; j < S; j++) {
        float d = 0.0f;
        for (int c = 0; c < DK; c++) { d += Q[i * D + base + c] * K[j * D + base + c]; }
        m = max(m, d * scale);
    }
    float s = 0.0f;
    for (int j = 0; j < S; j++) {
        float d = 0.0f;
        for (int c = 0; c < DK; c++) { d += Q[i * D + base + c] * K[j * D + base + c]; }
        float e = exp(d * scale - m);
        P[h * S * S + i * S + j] = e;
        s += e;
    }
    for (int j = 0; j < S; j++) { P[h * S * S + i * S + j] /= s; }
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) { acc += P[h * S * S + i * S + j] * V[j * D + base + c]; }
        Out[i * D + base + c] = acc;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dsim(TensorView<float> V, TensorView<float> P,
                   TensorView<float> dOut, TensorView<float> dsim, int S, int D, int H, int DK) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (i >= S || h >= H) { return; }
    float scale = 1.0f / sqrt((float)DK);
    int base = h * DK;
    float ds = 0.0f;
    for (int j = 0; j < S; j++) {
        float dp = 0.0f;
        for (int c = 0; c < DK; c++) { dp += dOut[i * D + base + c] * V[j * D + base + c]; }
        ds += P[h * S * S + i * S + j] * dp;
    }
    for (int j = 0; j < S; j++) {
        float dp = 0.0f;
        for (int c = 0; c < DK; c++) { dp += dOut[i * D + base + c] * V[j * D + base + c]; }
        dsim[h * S * S + i * S + j] = P[h * S * S + i * S + j] * (dp - ds) * scale;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dq(TensorView<float> K, TensorView<float> dsim, TensorView<float> dQ,
                 int S, int D, int H, int DK) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (i >= S || h >= H) { return; }
    int base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) { acc += dsim[h * S * S + i * S + j] * K[j * D + base + c]; }
        dQ[i * D + base + c] = acc;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dk(TensorView<float> Q, TensorView<float> dsim, TensorView<float> dK,
                 int S, int D, int H, int DK) {
    uint j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (j >= S || h >= H) { return; }
    int base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++) { acc += dsim[h * S * S + i * S + j] * Q[i * D + base + c]; }
        dK[j * D + base + c] = acc;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dv(TensorView<float> P, TensorView<float> dOut, TensorView<float> dV,
                 int S, int D, int H, int DK) {
    uint j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (j >= S || h >= H) { return; }
    int base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++) { acc += P[h * S * S + i * S + j] * dOut[i * D + base + c]; }
        dV[j * D + base + c] = acc;
    }
}

// ── batched multi-head attention (T7: B windows/step) ────────────────────────
// Same math as the single-window kernels; window index w = cudaBlockIdx().z (gridDim.z = B).
// Q/K/V/Out laid out (B·S × D), P laid out (B × H·S·S).  Row-wise ops (embed/LN/linear/relu)
// batch for free as (B·S)-row GEMMs; only attention must stay block-diagonal per window.
[AutoPyBindCUDA]
[CudaKernel]
void attn_fwd_b(TensorView<float> Q, TensorView<float> K, TensorView<float> V, TensorView<float> Out,
                TensorView<float> P, int B, int S, int D, int H, int DK) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint w = cudaBlockIdx().z;
    if (i >= S || h >= H || w >= B) { return; }
    int qo = w * S * D, po = w * H * S * S, base = h * DK;
    float scale = 1.0f / sqrt((float)DK);
    float m = -1e30f;
    for (int j = 0; j < S; j++) {
        float d = 0.0f;
        for (int c = 0; c < DK; c++) { d += Q[qo + i * D + base + c] * K[qo + j * D + base + c]; }
        m = max(m, d * scale);
    }
    float s = 0.0f;
    for (int j = 0; j < S; j++) {
        float d = 0.0f;
        for (int c = 0; c < DK; c++) { d += Q[qo + i * D + base + c] * K[qo + j * D + base + c]; }
        float e = exp(d * scale - m);
        P[po + h * S * S + i * S + j] = e;
        s += e;
    }
    for (int j = 0; j < S; j++) { P[po + h * S * S + i * S + j] /= s; }
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) { acc += P[po + h * S * S + i * S + j] * V[qo + j * D + base + c]; }
        Out[qo + i * D + base + c] = acc;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dsim_b(TensorView<float> V, TensorView<float> P, TensorView<float> dOut,
                     TensorView<float> dsim, int B, int S, int D, int H, int DK) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint w = cudaBlockIdx().z;
    if (i >= S || h >= H || w >= B) { return; }
    int qo = w * S * D, po = w * H * S * S, base = h * DK;
    float scale = 1.0f / sqrt((float)DK);
    float ds = 0.0f;
    for (int j = 0; j < S; j++) {
        float dp = 0.0f;
        for (int c = 0; c < DK; c++) { dp += dOut[qo + i * D + base + c] * V[qo + j * D + base + c]; }
        ds += P[po + h * S * S + i * S + j] * dp;
    }
    for (int j = 0; j < S; j++) {
        float dp = 0.0f;
        for (int c = 0; c < DK; c++) { dp += dOut[qo + i * D + base + c] * V[qo + j * D + base + c]; }
        dsim[po + h * S * S + i * S + j] = P[po + h * S * S + i * S + j] * (dp - ds) * scale;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dq_b(TensorView<float> K, TensorView<float> dsim, TensorView<float> dQ,
                   int B, int S, int D, int H, int DK) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint w = cudaBlockIdx().z;
    if (i >= S || h >= H || w >= B) { return; }
    int qo = w * S * D, po = w * H * S * S, base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int j = 0; j < S; j++) { acc += dsim[po + h * S * S + i * S + j] * K[qo + j * D + base + c]; }
        dQ[qo + i * D + base + c] = acc;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dk_b(TensorView<float> Q, TensorView<float> dsim, TensorView<float> dK,
                   int B, int S, int D, int H, int DK) {
    uint j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint w = cudaBlockIdx().z;
    if (j >= S || h >= H || w >= B) { return; }
    int qo = w * S * D, po = w * H * S * S, base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++) { acc += dsim[po + h * S * S + i * S + j] * Q[qo + i * D + base + c]; }
        dK[qo + j * D + base + c] = acc;
    }
}

[AutoPyBindCUDA]
[CudaKernel]
void attn_bwd_dv_b(TensorView<float> P, TensorView<float> dOut, TensorView<float> dV,
                   int B, int S, int D, int H, int DK) {
    uint j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint h = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    uint w = cudaBlockIdx().z;
    if (j >= S || h >= H || w >= B) { return; }
    int qo = w * S * D, po = w * H * S * S, base = h * DK;
    for (int c = 0; c < DK; c++) {
        float acc = 0.0f;
        for (int i = 0; i < S; i++) { acc += P[po + h * S * S + i * S + j] * dOut[qo + i * D + base + c]; }
        dV[qo + j * D + base + c] = acc;
    }
}

// ── elementwise (mirrors ew.slang) ───────────────────────────────────────────
[AutoPyBindCUDA]
[CudaKernel]
void add(TensorView<float> A, TensorView<float> B, TensorView<float> C, int n) {  // residual
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= n) { return; }
    C[i] = A[i] + B[i];
}

[AutoPyBindCUDA]
[CudaKernel]
void relu_fwd(TensorView<float> A, TensorView<float> C, int n) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= n) { return; }
    C[i] = max(0.0f, A[i]);
}

[AutoPyBindCUDA]
[CudaKernel]
void relu_bwd(TensorView<float> A, TensorView<float> B, TensorView<float> C, int n) {  // A=x B=dy C=dx
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= n) { return; }
    C[i] = A[i] > 0.0f ? B[i] : 0.0f;
}

// ── Adam (mirrors adamn.slang; bias-corrected, eps outside sqrt) ──────────────
[AutoPyBindCUDA]
[CudaKernel]
void adam_n(TensorView<float> W, TensorView<float> G, TensorView<float> M, TensorView<float> V,
            float t, float lr, float b1, float b2, float eps, int n) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= n) { return; }
    float g = G[i];
    float m = b1 * M[i] + (1.0f - b1) * g;
    float v = b2 * V[i] + (1.0f - b2) * g * g;
    M[i] = m;
    V[i] = v;
    float mhat = m / (1.0f - pow(b1, t));
    float vhat = v / (1.0f - pow(b2, t));
    W[i] = W[i] - lr * mhat / (sqrt(vhat) + eps);
}
