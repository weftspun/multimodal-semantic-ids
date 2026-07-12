#!/usr/bin/env bash
# Build and run the slangtrain milestones on the GPU.  slangc compiles the Slang
# autodiff kernels to SPIR-V; g++ builds the headless Vulkan runner (volk +
# Vulkan-Headers, both vendored).  Engine-free: no Python, no ONNX in the runtime.
# The gradient/parity checks (check_prims.py, check_enc.py) use torch only as the
# numerical oracle and are run separately.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/.."
cor="$here/../../core"
tp="$here/../../../gpu/thirdparty"
inc=(-I"$cor" -I"$tp/volk" -I"$tp/Vulkan-Headers/include")

# Single-entry kernels need -fvk-use-entrypoint-name (else the SPIR-V entry is "main").
slangc "$here/probe.slang" -entry probe -stage compute -fvk-use-entrypoint-name \
    -target spirv -o "$here/probe.spv"
slangc "$here/enc.slang" -entry encoder_test -stage compute -fvk-use-entrypoint-name \
    -target spirv -o "$here/enc.spv"
slangc "$here/adamn.slang" -entry adam_n -stage compute -fvk-use-entrypoint-name \
    -target spirv -o "$here/adamn.spv"
slangc "$here/tic.slang" -entry tic_test -stage compute -fvk-use-entrypoint-name \
    -target spirv -o "$here/tic.spv"
slangc "$here/encgrad.slang" -entry enc_grad -stage compute -fvk-use-entrypoint-name \
    -target spirv -o "$here/encgrad.spv"
slangc "$here/ticgrad.slang" -entry tic_grad -stage compute -fvk-use-entrypoint-name \
    -target spirv -o "$here/ticgrad.spv"
# Multi-entry training kernel (tgrad + adam): slang keeps the distinct entry names.
slangc "$here/train.slang" -target spirv -o "$here/train.spv"
# Tiled production kernels (multi-entry).
slangc "$here/gemm.slang" -target spirv -o "$here/gemm.spv"
slangc "$here/lin.slang" -target spirv -o "$here/lin.spv"
slangc "$here/ln.slang" -target spirv -o "$here/ln.spv"
slangc "$here/attn.slang" -target spirv -o "$here/attn.spv"
slangc "$here/ew.slang" -target spirv -o "$here/ew.spv"
# Multi-entry kernel: slang keeps the distinct [shader] entry names.
slangc "$here/prims.slang" -target spirv -o "$here/prims.spv"

g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/probe_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/probe.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/prims_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/prims.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/train_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/train.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/gemm_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/gemm.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/lin_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/lin.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/ln_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/ln.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/attn_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/attn.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/ew_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/ew.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/enc_tiled_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/enc_tiled.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/netfwd_main.cpp" "$cor/netfwd.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/netfwd.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/encbwd_tiled_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/encbwd.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/adamn_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/adamn.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/nettrain_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/nettrain.exe"
g++ -std=c++17 -DVK_NO_PROTOTYPES "${inc[@]}" \
    "$here/caltrain_main.cpp" "$cor/vkc.cpp" "$tp/volk/volk.c" -o "$here/caltrain.exe"

# Milestone 1: end-to-end Slang autodiff on the GPU (df/dx(2)=12).
"$here/probe.exe" "$here/probe.spv"
