// SPDX-License-Identifier: MIT
// Milestone 1: end-to-end Slang autodiff on the GPU.  Reverse-mode through
// f(x)=x^3 must write df/dx(2)=12 and f(2)=8 to the output buffer — the
// foundation the differentiable transformer trainer builds on.

[[vk::binding(0, 0)]] RWStructuredBuffer<float> outBuf;

[Differentiable]
float f(float x) { return x * x * x; }

[shader("compute")]
[numthreads(1, 1, 1)]
void probe(uint3 tid : SV_DispatchThreadID) {
    var dpx = diffPair(2.0f);
    bwd_diff(f)(dpx, 1.0f);   // reverse mode: seeds dResult=1, accumulates dpx.d
    outBuf[0] = dpx.d;         // expect 12
    outBuf[1] = f(2.0f);       // expect 8
}
