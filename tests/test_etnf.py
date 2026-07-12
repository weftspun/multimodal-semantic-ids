"""Stdlib-only tests for ETNF keys. Run: ``python -m tests.test_etnf``."""

import uuid

from vsk_recsys.data.etnf import asset_uuid, entity_uuid


def main() -> None:
    # Deterministic + content-addressable: same canonical key -> same uuid (dedup).
    assert asset_uuid("scene:forest#abc") == asset_uuid("scene:forest#abc")
    assert asset_uuid("hash:deadbeef") == entity_uuid("asset", "hash:deadbeef")

    # Type-namespaced: same natural key, different entity type -> different uuid.
    assert entity_uuid("asset", "x") != entity_uuid("user", "x")

    # Valid RFC-4122 v5.
    u = asset_uuid("k")
    assert isinstance(u, uuid.UUID) and u.version == 5

    # Empty components rejected.
    for bad in [("", "x"), ("asset", "")]:
        try:
            entity_uuid(*bad)
            raise AssertionError("expected ValueError for empty component")
        except ValueError:
            pass

    print("test_etnf: OK")


if __name__ == "__main__":
    main()
