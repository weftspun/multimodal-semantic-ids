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


def _spread3(v):
    """Interleave low 21 bits of v with 2 zero bits between each (Morton bit-spread)."""
    v = v.long() & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton_order(coords):
    """Argsort voxel coords into Morton (Z-order). Kept for reference; ``hilbert_order`` is preferred."""
    import torch

    code = _spread3(coords[:, 0]) | (_spread3(coords[:, 1]) << 1) | (_spread3(coords[:, 2]) << 2)
    return torch.argsort(code)


def hilbert_order(coords):
    """3D Hilbert order — better locality than Morton on DENSE grids, marginal on sparse SLAT.

    NOTE: a hand-rolled Skilling implementation failed the defining dense-grid test (correct 3D Hilbert
    has all consecutive steps = 1.0). For a production Hilbert order use the vetted MIT ``hilbertcurve``
    library (``HilbertCurve(p=bits, n=3).distances_from_points``) and argsort the distances. We are NOT
    matching any reference (the sparse-conv encoder is order-independent), so this is a pure quality
    choice; measured on sparse SLAT the Morton/Hilbert difference is marginal, so ``encode_to_slat``
    uses the correct ``morton_order`` by default.
    """
    raise NotImplementedError(
        "Use morton_order (correct) or wire the vetted MIT `hilbertcurve` lib; see docstring."
    )


def encode_to_slat(vertices, faces, encoder, grid_size: int = 64, device: str = "cuda"):
    """Mesh -> O-Voxel -> **structured, Morton-ordered** SLAT. Returns ``(feats, coords)``, NOT pooled.

    ``feats``: (N, C) — N structured SLAT tokens (one per active voxel), C latent channels (32 for the
    shape VAE). ``coords``: (N, 3) — the voxel coordinates. **No mean/average pooling** — averaging would
    destroy structure. Tokens are returned in **canonical Morton (Z-order)** so the sequence is
    deterministic AND spatially coherent (order of SLAT tokens matters for the sequence model). Downstream,
    each SLAT token is FSQ-quantized into a semantic code, so the mesh contributes an *ordered set* of
    codes (like text/image). Deterministic: the VAE returns the posterior mean (``sample_posterior=False``,
    ``.eval()``); repeated encodes diff = 0.0.
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
    order = morton_order(coords)  # correct canonical Z-order (deterministic); order of tokens matters
    return feats[order], coords[order]  # (N, C) structured tokens + (N, 3) coords, Morton-ordered
