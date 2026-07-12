"""Essential Tuple Normal Form (ETNF) identity for the parquet feature lake.

Per V-Sekai-fire/study-data-vsk/SCHEMA.md and decisions/20260712-parquet-feature-store-etnf.md:
one relation per ``lake/*.parquet``; entity keys are deterministic UUIDv5 over a canonical
natural key, so a re-imported / rescaled duplicate asset resolves to the SAME uuid (allocentric,
content-addressable identity — free cold-start dedup).

Stdlib-only (``uuid``) so this imports without torch/numpy.
"""

from __future__ import annotations

import uuid

# Fixed lake namespace — derived once, deterministically, from the repo URL.
NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/V-Sekai-fire/vsk-session-item-recommendation-01",
)


def entity_uuid(entity_type: str, natural_key: str) -> uuid.UUID:
    """``uuid5(NAMESPACE, "<entity_type>:<natural_key>")`` — the SCHEMA.md key rule.

    e.g. ``entity_uuid("asset", "scene:forest#<content-hash>")``.
    """
    if not entity_type or not natural_key:
        raise ValueError("entity_type and natural_key must both be non-empty")
    return uuid.uuid5(NAMESPACE, f"{entity_type}:{natural_key}")


def asset_uuid(canonical_key: str) -> uuid.UUID:
    """Asset identity keyed by its *canonical* natural key (content hash / normalized slug)."""
    return entity_uuid("asset", canonical_key)


def user_uuid(natural_key: str) -> uuid.UUID:
    return entity_uuid("user", natural_key)


def session_uuid(natural_key: str) -> uuid.UUID:
    return entity_uuid("session", natural_key)
