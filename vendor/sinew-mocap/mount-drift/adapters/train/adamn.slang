// SPDX-License-Identifier: MIT
// General Adam update over an arbitrary-length weight tensor (n from the params
// buffer), so the real-dim training loop can update every weight tensor with one
// kernel.  Matches torch.optim.Adam (bias-corrected, eps outside the sqrt).
// params = [t, lr, beta1, beta2, eps, n].  Verified vs numpy in check_adamn.py.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> W;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> G;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> M;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> V;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void adam_n(uint3 tid: SV_DispatchThreadID) {
    int n = int(params[5]);
    int i = int(tid.x);
    if (i >= n) {
        return;
    }
    float t = params[0], lr = params[1], b1 = params[2], b2 = params[3], eps = params[4];
    float g = G[i];
    float m = b1 * M[i] + (1.0f - b1) * g;
    float v = b2 * V[i] + (1.0f - b2) * g * g;
    M[i] = m;
    V[i] = v;
    float mhat = m / (1.0f - pow(b1, t));
    float vhat = v / (1.0f - pow(b2, t));
    W[i] = W[i] - lr * mhat / (sqrt(vhat) + eps);
}
