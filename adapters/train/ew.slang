// SPDX-License-Identifier: MIT
// Tiled production kernels: elementwise pieces for assembly — residual add and
// ReLU fwd/bwd.  1D dispatch over n.  Verified vs torch in check_ew.py.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> A;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dims;  // [n]

[shader("compute")]
[numthreads(64, 1, 1)]
void add(uint3 tid: SV_DispatchThreadID) {  // C = A + B (residual)
    int n = int(dims[0]);
    int i = int(tid.x);
    if (i >= n) {
        return;
    }
    C[i] = A[i] + B[i];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void relu_fwd(uint3 tid: SV_DispatchThreadID) {  // C = max(0, A)
    int n = int(dims[0]);
    int i = int(tid.x);
    if (i >= n) {
        return;
    }
    C[i] = max(0.0f, A[i]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void relu_bwd(uint3 tid: SV_DispatchThreadID) {  // C = (A>0) ? B : 0   (A=x, B=dy, C=dx)
    int n = int(dims[0]);
    int i = int(tid.x);
    if (i >= n) {
        return;
    }
    C[i] = A[i] > 0.0f ? B[i] : 0.0f;
}
