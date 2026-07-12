"""ETNF parquet lake helper — one relation per ``lake/*.parquet``, integrity checks, DuckDB join view.

Per decisions/20260712-parquet-feature-store-etnf.md: deterministic UUIDv5 keys, all-key junctions,
per-modality 1:1 / 1:N derived-extension relations keyed by ``asset_uuid``, PK-unique + FK-resolves,
training features assembled by a DuckDB join view (no redundant transitive columns).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_relation(name: str, rows: list[dict], out_dir: str = "lake", pk: list[str] | None = None) -> Path:
    """Write ``rows`` to ``<out_dir>/<name>.parquet`` (one relation per file).

    If ``pk`` (list of key columns) is given, assert it is unique (PK integrity).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if pk:
        seen: set = set()
        for r in rows:
            key = tuple(r[c] for c in pk)
            if key in seen:
                raise ValueError(f"{name}: duplicate PK {pk} = {key}")
            seen.add(key)
    path = out / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def check_fk(child_rows: list[dict], fk_col: str, parent_rows: list[dict], parent_key: str) -> None:
    """Assert every ``child[fk_col]`` resolves into ``parent[parent_key]`` (no orphans)."""
    parent_keys = {r[parent_key] for r in parent_rows}
    orphans = [r[fk_col] for r in child_rows if r[fk_col] not in parent_keys]
    if orphans:
        raise ValueError(f"{len(orphans)} orphan FK {fk_col} not in parent {parent_key}: {orphans[:3]}")


def training_features_view(lake_dir: str = "lake", con=None):
    """Build a DuckDB view LEFT-JOINing ``assets`` with every ``asset_*`` relation on ``asset_uuid``.

    Returns the duckdb connection with a ``training_features`` view registered. Assembles features by
    JOIN — no derived columns stored on ``assets``.
    """
    import duckdb

    con = con or duckdb.connect()
    lake = Path(lake_dir)
    assets = lake / "assets.parquet"
    joins = ""
    for i, rel in enumerate(sorted(lake.glob("asset_*.parquet"))):
        alias = f"r{i}"
        joins += f"\n  LEFT JOIN read_parquet('{rel.as_posix()}') {alias} ON a.asset_uuid = {alias}.asset_uuid"
    con.execute(
        f"CREATE OR REPLACE VIEW training_features AS "
        f"SELECT a.* FROM read_parquet('{assets.as_posix()}') a{joins}"
    )
    return con
