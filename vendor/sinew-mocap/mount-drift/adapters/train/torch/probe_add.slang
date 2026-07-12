// slangtorch toolchain probe: a trivial elementwise add as a CUDA autograd-extension kernel.
// Confirms slangtorch can JIT-compile a [CudaKernel] against the torch cu124 headers with the
// system nvcc + MSVC cl before we port the real bricks.
[AutoPyBindCUDA]
[CudaKernel]
void add(TensorView<float> a, TensorView<float> b, TensorView<float> result) {
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= result.size(0)) { return; }
    result[i] = a[i] + b[i];
}
