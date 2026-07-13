"""Assemble + verify the ETNF lake (planner action a_p2_store).

1. Register the mesh files as first-class ``assets`` (kind=``mesh_asset``) alongside the ``godot_scene``
   rows, so the mesh/image feature relations have a parent to reference.
2. Enforce ETNF integrity: PK-unique on every entity/junction, FK-resolves for every feature relation.
3. Build the DuckDB ``training_features`` join view and report per-modality coverage.

Run (default env): pixi run python scripts/assemble_lake.py --root data/godot-demo-projects
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from vsk_recsys.data.godot import mesh_assets
from vsk_recsys.data.lake import check_fk, training_features_view, write_relation

# Each feature relation and the FK column that must resolve into assets.asset_uuid.
FEATURE_RELATIONS = (
    "asset_text_embedding",
    "asset_image_embedding",
    "asset_mesh_shape_slat",
    "asset_mesh_texture_slat",
    "asset_audio_clap",
    "asset_body_phenotype",
)


def _read_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist() if path.exists() else []


def register_mesh_assets(lake: Path, root: str) -> list[dict]:
    """Union the scene assets with fresh mesh_asset rows (idempotent) and rewrite ``assets.parquet``."""
    existing = _read_rows(lake / "assets.parquet")

    # Drop any prior mesh_asset rows so re-runs don't duplicate, then re-add the current mesh set.
    scenes = [r for r in existing if r.get("kind") != "mesh_asset"]
    meshes = mesh_assets(root)

    assets = scenes + meshes
    write_relation("assets", assets, str(lake), pk=["asset_uuid"])
    return assets


def verify_foreign_keys(lake: Path, assets: list[dict]) -> dict:
    """Assert every present feature relation's asset_uuid resolves into ``assets``; return row counts."""
    counts = {}
    for name in FEATURE_RELATIONS:
        rows = _read_rows(lake / f"{name}.parquet")
        if not rows:
            continue
        check_fk(rows, "asset_uuid", assets, "asset_uuid")  # raises on any orphan
        counts[name] = len(rows)
    return counts


def report_coverage(lake: Path) -> None:
    """Build the DuckDB join view and print per-modality non-null coverage."""
    import duckdb

    con = training_features_view(str(lake), duckdb.connect())
    total = con.execute("SELECT count(*) FROM training_features").fetchone()[0]
    print(f"training_features view: {total} assets")
    for col, label in (
        ("text_embedding", "text"),
        ("image_embedding", "image"),
        ("slat_feats", "mesh"),  # both shape + texture expose slat_feats; DuckDB suffixes the dup
        ("audio_embedding", "audio"),
        ("phenotype_params", "phenotype"),
    ):
        try:
            n = con.execute(
                f"SELECT count(*) FROM training_features WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            print(f"  {label:6s}: {n}")
        except duckdb.BinderException:
            pass  # column absent (relation not built yet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/godot-demo-projects", help="corpus root for mesh assets")
    ap.add_argument("--lake", default="lake")
    args = ap.parse_args()

    lake = Path(args.lake)

    # Register mesh assets, then verify the whole lake resolves before assembling the training view.
    assets = register_mesh_assets(lake, args.root)
    scenes = sum(1 for r in assets if r["kind"] == "godot_scene")
    meshes = sum(1 for r in assets if r["kind"] == "mesh_asset")
    print(f"assets: {len(assets)} ({scenes} scenes + {meshes} meshes)")

    counts = verify_foreign_keys(lake, assets)
    print("FK-resolves OK:", counts)

    report_coverage(lake)


if __name__ == "__main__":
    main()
