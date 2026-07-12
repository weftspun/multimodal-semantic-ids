"""Uro backpack → ETNF ingestion, verified on a synthetic fixture (no real uro data yet).

Run: ``python -m tests.test_uro`` (needs pyarrow). Proves the mapping + PK/FK integrity so real uro
exports drop straight in. See vsk_recsys/data/uro.py.
"""

import tempfile

from vsk_recsys.data.uro import UroItemKind, ingest_backpacks

# Two users, three backpacks; one asset (a map) is shared across two backpacks to exercise edge dedup.
SHARED_MAP = "map-forest"
FIXTURE = [
    {"id": "bp-1", "owner_id": "user-a", "joins": [
        {"map_id": SHARED_MAP, "inserted_at": "2026-04-15T10:00:00Z"},
        {"avatar_id": "av-robot", "inserted_at": "2026-04-15T10:05:00Z"},
    ]},
    {"id": "bp-2", "owner_id": "user-a", "joins": [
        {"map_id": SHARED_MAP, "inserted_at": "2026-04-16T09:00:00Z"},  # same user + map again
        {"prop_id": "prop-sword", "inserted_at": "2026-04-16T09:01:00Z"},
    ]},
    {"id": "bp-3", "owner_id": "user-b", "joins": [
        {"avatar_id": "av-robot", "inserted_at": "2026-04-17T12:00:00Z"},  # shared avatar, other user
    ]},
]


def main() -> None:
    with tempfile.TemporaryDirectory() as out:
        counts = ingest_backpacks(FIXTURE, out)

    # 2 distinct owners; 3 distinct items (map, avatar, prop); 3 backpacks = 3 sessions.
    assert counts["users"] == 2, counts
    assert counts["assets_uro"] == 3, counts
    assert counts["sessions_uro"] == 3, counts

    # session_assets = one row per join (2 + 2 + 1); no dedup across sessions.
    assert counts["session_assets_uro"] == 5, counts

    # user_item_edges dedup per (user, asset): user-a owns {map, avatar, prop}=3; user-b owns {avatar}=1.
    assert counts["user_item_edges"] == 4, counts

    # A malformed join (zero or multiple item refs) must be rejected, not silently mis-keyed.
    for bad in ([{"inserted_at": "t"}], [{"map_id": "m", "prop_id": "p", "inserted_at": "t"}]):
        try:
            ingest_backpacks([{"id": "x", "owner_id": "u", "joins": bad}], tempfile.mkdtemp())
            raise AssertionError("expected ValueError for malformed backpack_join")
        except ValueError:
            pass

    assert set(UroItemKind) == {UroItemKind.MAP, UroItemKind.AVATAR, UroItemKind.PROP}
    print("test_uro: OK", counts)


if __name__ == "__main__":
    main()
