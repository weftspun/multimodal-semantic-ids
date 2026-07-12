// SPDX-License-Identifier: MIT
// Tiled production kernels, brick 2: the pieces a linear layer needs beyond GEMM.
// bias_add folds the bias into the forward; col_sum is the bias gradient (Σ over
// rows of dY).  Combined with gemm (nt forward, nn/tn gradients) these form the
// full tiled linear fwd/bwd, verified against torch nn.Linear in check_lin.py.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> Y;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> vec;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dims;  // [rows, cols]

[shader("compute")]
[numthreads(16, 16, 1)]
void bias_add(uint3 tid: SV_DispatchThreadID) {
    int rows = int(dims[0]), cols = int(dims[1]);
    int r = int(tid.y), c = int(tid.x);
    if (r >= rows || c >= cols) {
        return;
    }
    Y[r * cols + c] = Y[r * cols + c] + vec[c];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void col_sum(uint3 tid: SV_DispatchThreadID) {  // vec[c] = Σ_r Y[r,c]
    int rows = int(dims[0]), cols = int(dims[1]);
    int c = int(tid.x);
    if (c >= cols) {
        return;
    }
    float s = 0.0f;
    for (int r = 0; r < rows; r++) {
        s += Y[r * cols + c];
    }
    vec[c] = s;
}
