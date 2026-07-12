// SPDX-License-Identifier: MIT
// Tiled production kernels, brick 1: GEMM.  One thread per output element, 2D
// dispatch over (row, col).  Three forms cover the transformer's linear forward
// and both gradient products:
//   nn: C = A·B            (dX = dY·W)
//   nt: C = A·Bᵀ           (Y  = X·Wᵀ)
//   tn: C = Aᵀ·B           (dW = dYᵀ·X)
// dims buffer holds [M, N, K]; A is M×K (or K×M for tn), B is K×N (or N×K for nt),
// C is M×N.  Verified against torch in check_gemm.py.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> A;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dims;  // [M, N, K]

[shader("compute")]
[numthreads(16, 16, 1)]
void gemm_nn(uint3 tid: SV_DispatchThreadID) {
    int M = int(dims[0]), N = int(dims[1]), K = int(dims[2]);
    int row = int(tid.y), col = int(tid.x);
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gemm_nt(uint3 tid: SV_DispatchThreadID) {  // B is N×K, use Bᵀ
    int M = int(dims[0]), N = int(dims[1]), K = int(dims[2]);
    int row = int(tid.y), col = int(tid.x);
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gemm_tn(uint3 tid: SV_DispatchThreadID) {  // A is K×M, use Aᵀ
    int M = int(dims[0]), N = int(dims[1]), K = int(dims[2]);
    int row = int(tid.y), col = int(tid.x);
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[k * M + row] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
