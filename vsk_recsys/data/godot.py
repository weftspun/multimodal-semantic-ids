"""Godot corpus ingestion → ETNF entity + junction relations (a_p2_encoders).

Parses a `godot-demo-projects` checkout (the bootstrap proxy — real source is uro backpacks;
decisions/20260712-phase2-foss-datasets.md). Items = `.tscn` scenes (incl. the decoded `.scn`), sessions
= projects (dir with `project.godot`). Each scene is assigned to its NEAREST enclosing project. Writes
`assets` / `sessions` / `session_assets` parquet via the ETNF lake helper (UUIDv5 keys, PK-unique,
FK-resolves). `scene_text` = the `.tscn` scene-graph text (fed to the Qwen3-VL text encoder).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from vsk_recsys.data.etnf import asset_uuid, session_uuid
from vsk_recsys.data.lake import check_fk, write_relation

MAX_SCENE_TEXT = 20000  # cap per-scene text (the encoder truncates anyway)
MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply")


def mesh_asset_uuid(rel: str) -> str:
    """Stable content key for a mesh asset from its corpus-relative path (matches the scene convention).

    The mesh SLAT + render encoders MUST key their relations with this so the ETNF foreign keys resolve
    into ``assets``. Using the relative path (not an absolute ``/mnt/c/...`` path) keeps keys portable.
    """
    return str(asset_uuid(f"godot_mesh:{rel}"))


def discover_mesh_files(root: Path) -> list[Path]:
    """Return sorted mesh files under ``root`` (recursive), skipping Godot's ``.godot`` import cache."""
    found = [p for p in root.rglob("*") if p.suffix.lower() in MESH_EXTENSIONS and ".godot" not in p.parts]
    return sorted(found)


def mesh_assets(root: str) -> list[dict]:
    """Build ``assets`` rows (kind=``mesh_asset``) for every mesh file in the corpus.

    Registers the meshes as first-class assets so the mesh/image feature relations have a parent row to
    reference (they are a different asset population than the ``godot_scene`` rows).
    """
    root = Path(root)
    rows = []
    for path in discover_mesh_files(root):
        rel = path.relative_to(root).as_posix()
        rows.append({
            "asset_uuid": mesh_asset_uuid(rel), "natural_key": rel, "kind": "mesh_asset",
            "display_name": path.stem, "slug": rel, "scene_text": None,
        })
    return rows


def _projects(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("project.godot"))


def _nearest_project(scene: Path, proj_set: set) -> Path | None:
    d = scene.parent
    while d != d.parent:
        if d in proj_set:
            return d
        d = d.parent
    return None


def ingest(root: str, out_dir: str = "lake") -> dict:
    root = Path(root)
    projects = _projects(root)
    proj_set = set(projects)
    sessions, suid_of = [], {}
    for proj in projects:
        rel = proj.relative_to(root).as_posix()
        suid = str(session_uuid(f"godot:{rel}"))
        suid_of[proj] = suid
        sessions.append({"session_uuid": suid, "natural_key": rel, "project": proj.name})

    by_proj: dict[Path, list[Path]] = defaultdict(list)
    for scene in root.rglob("*.tscn"):
        if ".godot" in scene.parts:
            continue
        proj = _nearest_project(scene, proj_set)
        if proj is not None:
            by_proj[proj].append(scene)

    assets, session_assets, seen = [], [], set()
    for proj, scenes in by_proj.items():
        suid = suid_of[proj]
        for pos, scene in enumerate(sorted(scenes)):
            rel = scene.relative_to(root).as_posix()
            auid = str(asset_uuid(f"godot_scene:{rel}"))
            if auid not in seen:
                text = scene.read_text(encoding="utf-8", errors="replace")[:MAX_SCENE_TEXT]
                assets.append({
                    "asset_uuid": auid, "natural_key": rel, "kind": "godot_scene",
                    "display_name": scene.stem, "slug": rel, "scene_text": text,
                })
                seen.add(auid)
            session_assets.append({"session_uuid": suid, "asset_uuid": auid, "position": pos})

    write_relation("assets", assets, out_dir, pk=["asset_uuid"])
    write_relation("sessions", sessions, out_dir, pk=["session_uuid"])
    write_relation("session_assets", session_assets, out_dir, pk=["session_uuid", "asset_uuid"])
    check_fk(session_assets, "asset_uuid", assets, "asset_uuid")
    check_fk(session_assets, "session_uuid", sessions, "session_uuid")
    return {"assets": len(assets), "sessions": len(sessions), "session_assets": len(session_assets)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/godot-demo-projects")
    ap.add_argument("--out-dir", default="lake")
    args = ap.parse_args()
    print("ingested:", ingest(args.root, args.out_dir), "->", args.out_dir)


if __name__ == "__main__":
    main()
