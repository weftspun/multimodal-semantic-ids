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
            "Stage 2 (O-Voxel -> SLAT shape-VAE embedding -> pooled vector) is the remaining part of "
            "planner action a_p2_mesh_encoder; needs the TRELLIS.2 shape encoder + 709 MB MIT weights. "
            "Stage 1 voxelization is available via voxelize()/voxelize_trimesh()."
        )
