"""Mesh encoder — TRELLIS.2 O-Voxel voxelization (Windows, FOSS-clean).

Stage 1 (implemented, verified on Windows + torch 2.8 + RTX 4090): mesh -> O-Voxel sparse voxel grid
via the MIT `o_voxel` wheel from the stableprojectorz fork. Stage 2 (TODO, planner action
a_p2_mesh_encoder cont.): O-Voxel -> SLAT via the TRELLIS.2 shape-VAE encoder (709 MB MIT weights) ->
canonical-normalize -> pool to a fixed mesh vector. See decisions/20260712-multimodal-foss-encoder-stack.md.

Runs in the isolated pixi `mesh` env (torch 2.8 matches the cp311 wheels): `pixi run -e mesh ...`.

Two `o_voxel` quirks are handled here:
  1. `o_voxel/__init__.py` eagerly imports the render/decode side (postprocess -> flex_gemm -> triton,
     nvdiffrast, cv2). We only voxelize (encode), so we register a bare `o_voxel` package and import the
     voxelizer submodule directly — no triton/nvdiffrast needed, keeping the path FOSS-clean.
  2. The voxelizer `.cuda()`s the auto-computed aabb but calls a `*_cpu` kernel (device-mismatch bug),
     so we pass `aabb` explicitly on CPU.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import types


def load_voxelizer():
    """Import ONLY the MIT o_voxel voxelizer, bypassing the render-side package __init__."""
    if "o_voxel.convert" not in sys.modules:
        ovox = os.path.join(sysconfig.get_paths()["purelib"], "o_voxel")
        pkg = types.ModuleType("o_voxel")
        pkg.__path__ = [ovox]
        pkg.__package__ = "o_voxel"
        pkg.__file__ = os.path.join(ovox, "__init__.py")
        sys.modules["o_voxel"] = pkg
    import o_voxel.convert as convert  # noqa: E402  (deliberate late import)

    return convert


def voxelize(vertices, faces, grid_size: int = 64):
    """Mesh (vertices Nx3, faces Mx3) -> occupied voxel indices (K, 3), all on CPU.

    ``vertices``/``faces`` are torch tensors (float32 / int32). Returns the O-Voxel occupied-voxel
    index tensor. Verified: an icosphere (2562 v / 5120 f) -> 18752 voxels at grid 64.
    """
    import torch

    convert = load_voxelizer()
    v = vertices.detach().to(torch.float32).cpu()
    f = faces.detach().to(torch.int32).cpu()
    mn, mx = v.min(0).values, v.max(0).values
    pad = (mx - mn) / (grid_size - 1)
    aabb = torch.stack([mn - pad * 0.5, mx + pad * 0.5], dim=0).float()  # CPU — skips internal .cuda()
    out = convert.mesh_to_flexible_dual_grid(v, f, grid_size=grid_size, aabb=aabb)
    return out[0] if isinstance(out, (tuple, list)) else out


class MeshEncoder:
    modality = "mesh"
    relation = "asset_mesh_embedding"

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size

    def voxelize_trimesh(self, mesh, grid_size: int | None = None):
        """A trimesh.Trimesh -> O-Voxel occupied voxel indices."""
        import torch

        v = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        f = torch.as_tensor(mesh.faces, dtype=torch.int32)
        return voxelize(v, f, grid_size or self.grid_size)

    def encode(self, meshes):  # -> np.ndarray (n, dim)
        raise NotImplementedError(
            "Batch encode is per-catalog; use load_shape_encoder() + encode_to_slat() per mesh. "
            "See scripts/mesh_encode_linux.py for the offline batch job."
        )


def load_shape_encoder(weights_prefix: str, trellis2_path: str, device: str = "cuda"):
    """Load the TRELLIS.2 FlexiDualGridVaeEncoder (MIT weights). Stage-2 mesh encoder.

    VERIFIED on WSL2 Linux (torch 2.6/cu124, RTX 4090) with FlexGEMM + o-voxel built from source —
    FlexGEMM-from-source has the triton GEMM kernels the Windows prebuilt wheel lacked. Sets the sparse
    backends and stubs the decode/render-only libs (cumesh/nvdiffrast/cv2) the encoder never uses.
    ``weights_prefix``: path prefix to ``{prefix}.json`` + ``{prefix}.safetensors``.
    """
    import os
    import sys
    import types

    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, trellis2_path)
    for n in ["cumesh", "nvdiffrast", "nvdiffrast.torch", "cv2", "nvdiffrec", "nvdiffrec_render"]:
        sys.modules.setdefault(n, types.ModuleType(n))
    import trellis2.models as models  # noqa: E402

    return models.from_pretrained(weights_prefix).eval().to(device)


def canonical_order(coords, mode: str = "hilbert"):
    """Argsort (N,3) voxel coords into O-Voxel's **vetted** canonical order.

    Uses `o_voxel.serialize.encode_seq` (the `_C` implementation). ``mode="hilbert"`` is a *true* 3D
    Hilbert curve — verified: all consecutive steps = 1.0 on a dense grid (strictly better locality than
    z-order's avg 1.44 / max-jump 9.95). ``mode="z_order"`` is Morton. Token order matters for the
    downstream sequence model; we are NOT matching any reference (the sparse-conv encoder is
    order-independent), so we choose Hilbert for locality. (My earlier hand-rolled Hilbert was buggy —
    always prefer O-Voxel's `_C` version.)
    """
    import torch

    load_voxelizer()  # registers the bare o_voxel package
    import o_voxel.serialize as serialize

    return torch.argsort(serialize.encode_seq(coords.int(), mode=mode))


def encode_to_slat(vertices, faces, encoder, grid_size: int = 64, device: str = "cuda"):
    """Mesh -> O-Voxel -> **structured, Hilbert-ordered** SHAPE SLAT. Returns ``(feats, coords)``, NOT pooled.

    ``feats``: (N, 32) — N structured shape-SLAT tokens (one per active voxel). ``coords``: (N, 3) — voxel
    coords. **No mean/average pooling** — averaging destroys structure. Tokens are ordered by O-Voxel's
    vetted **Hilbert** curve (order matters for the sequence model; better locality than z-order).
    Downstream, each token is FSQ-quantized into a semantic code, so the mesh contributes an *ordered set*
    of codes (like text/image). Deterministic: the VAE returns the posterior mean (``sample_posterior=False``,
    ``.eval()``); repeated encodes diff = 0.0. This is GEOMETRY only — see ``encode_texture_to_slat`` for the
    PBR/appearance half (full mesh = shape ⊕ texture).
    """
    import torch

    convert = load_voxelizer()
    import trellis2.modules.sparse as sp

    v = vertices.detach().to(torch.float32).cpu()
    f = faces.detach().to(torch.int32).cpu()
    mn, mx = v.min(0).values, v.max(0).values
    pad = (mx - mn) / (grid_size - 1)
    aabb = torch.stack([mn - pad * 0.5, mx + pad * 0.5], dim=0).float()
    coords, dual, inter = convert.mesh_to_flexible_dual_grid(v, f, grid_size=grid_size, aabb=aabb)
    cb = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1).int()
    verts = sp.SparseTensor(dual.float(), cb)
    inter_st = verts.replace(inter.float())
    with torch.no_grad():
        z = encoder(verts.to(device), inter_st.to(device))  # posterior mean; deterministic
    feats = z.feats.float().cpu()
    coords = z.coords[:, 1:].cpu()
    order = canonical_order(coords, mode="hilbert")  # O-Voxel's vetted Hilbert; token order matters
    return feats[order], coords[order]  # (N, C) structured shape tokens + (N, 3) coords, Hilbert-ordered


# `load_shape_encoder` is generic (from_pretrained reads the model name from the config), so it also
# loads the texture encoder — alias for clarity.
load_texture_encoder = load_shape_encoder


def encode_texture_to_slat(asset, tex_encoder, grid_size: int = 64, device: str = "cuda"):
    """Textured mesh -> PBR volumetric attrs -> **structured, Hilbert-ordered** TEXTURE SLAT.

    ``asset``: a ``trimesh.Scene``/``Trimesh`` (or glb path) WITH PBR materials/textures. Uses O-Voxel's
    ``textured_mesh_to_volumetric_attr`` to sample per-voxel PBR = ``[base_color(3), metallic, roughness,
    alpha]`` (6 channels, normalized to [-1, 1]), then the ``SparseUnetVaeEncoder`` (`tex_enc`). Returns
    ``(feats, coords)`` — (N, 32) structured texture tokens + coords, Hilbert-ordered, no pooling. The full
    mesh representation is the shape tokens ⊕ these texture tokens.
    """
    import torch

    convert = load_voxelizer()
    import trellis2.modules.sparse as sp

    voxel_indices, attributes = convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=grid_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    )
    parts = []
    for k in ("base_color", "metallic", "roughness", "alpha"):
        a = attributes[k]
        parts.append(a if a.ndim == 2 else a[:, None])
    feats = torch.cat(parts, dim=-1).float()  # (N, 6)
    if feats.numel() == 0:  # geometry-only mesh (no PBR texture / vertex colors) -> no texture tokens
        return torch.zeros((0, 32)), torch.zeros((0, 3), dtype=torch.int32)
    feats = feats / 255.0 * 2 - 1 if feats.max() > 1.5 else feats * 2 - 1  # -> [-1, 1]
    cb = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1).int()
    x = sp.SparseTensor(feats, cb)
    with torch.no_grad():
        z = tex_encoder(x.to(device))  # single-input (unlike the shape encoder); posterior mean
    tf = z.feats.float().cpu()
    tc = z.coords[:, 1:].cpu()
    order = canonical_order(tc, mode="hilbert")
    return tf[order], tc[order]
