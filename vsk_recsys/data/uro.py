"""V-Sekai `uro` backpacks → ETNF entity + junction relations (planner action a_p2_uro_backpack).

The REAL user→item source (the Godot corpus is only the bootstrap proxy). Schema — `V-Sekai/uro`
(`benbot/backpack-inventory`, `Uro.Inventory.Backpack`):
  - ``backpacks``      : id, owner_id → users, timestamps
  - ``backpack_join``  : id, backpack_id, ONE of {map_id, avatar_id, prop_id}, timestamps

A backpack = a user's timestamped set of owned Maps/Avatars/Props = a session/"basket". ETNF mapping:
  owner            → ``users``
  Map/Avatar/Prop  → ``assets`` (kind = uro_map|uro_avatar|uro_prop)
  backpack         → ``sessions`` (owner_uuid)
  backpack_join    → ``session_assets`` (ordered by join time) + ``user_item_edges`` (user↔item, timestamp)

Swapping the Godot proxy for uro is a data-source change, not a schema change — this module produces the
same relation shapes. Wired now against the schema; runs on real uro exports once available.
"""

from __future__ import annotations

from enum import Enum

from vsk_recsys.data.etnf import asset_uuid, session_uuid, user_uuid
from vsk_recsys.data.lake import check_fk, write_relation


class UroItemKind(str, Enum):
    """A backpack_join references exactly one content kind — never a bare string at a call site."""

    MAP = "uro_map"
    AVATAR = "uro_avatar"
    PROP = "uro_prop"


# Which nullable join column carries the item id, and the kind it denotes.
_JOIN_FIELD_KIND = {
    "map_id": UroItemKind.MAP,
    "avatar_id": UroItemKind.AVATAR,
    "prop_id": UroItemKind.PROP,
}


def _join_item(join: dict) -> tuple[UroItemKind, str]:
    """Resolve a backpack_join row to (kind, uro_item_id); exactly one of map/avatar/prop must be set."""
    present = [(kind, join[field]) for field, kind in _JOIN_FIELD_KIND.items() if join.get(field)]
    if len(present) != 1:
        raise ValueError(f"backpack_join must reference exactly one item, got {len(present)}: {join}")
    return present[0]


def _asset_uuid(kind: UroItemKind, item_id: str) -> str:
    return str(asset_uuid(f"{kind.value}:{item_id}"))


def ingest_backpacks(backpacks: list[dict], out_dir: str = "lake") -> dict:
    """Map uro backpack records → ETNF parquet relations with PK-unique + FK-resolves integrity.

    ``backpacks``: rows shaped like the uro export —
        {"id": <backpack_uuid>, "owner_id": <user_uuid>,
         "joins": [{"map_id"|"avatar_id"|"prop_id": <uuid>, "inserted_at": <iso8601>}, ...]}
    """
    users: dict[str, dict] = {}
    assets: dict[str, dict] = {}
    sessions, session_assets = [], []
    edges: dict[tuple[str, str], dict] = {}  # (user_uuid, asset_uuid) → aggregated interaction edge

    for backpack in backpacks:
        # The owner is a user; the backpack is that user's session/basket.
        owner_uuid = str(user_uuid(f"uro:{backpack['owner_id']}"))
        users.setdefault(owner_uuid, {"user_uuid": owner_uuid, "natural_key": f"uro:{backpack['owner_id']}"})

        suid = str(session_uuid(f"uro_backpack:{backpack['id']}"))
        sessions.append({"session_uuid": suid, "natural_key": f"uro_backpack:{backpack['id']}",
                         "owner_uuid": owner_uuid})

        # Items are ordered by when they were added to the backpack (join timestamp).
        joins = sorted(backpack.get("joins", []), key=lambda j: j.get("inserted_at") or "")
        for position, join in enumerate(joins):
            kind, item_id = _join_item(join)
            auid = _asset_uuid(kind, item_id)
            assets.setdefault(auid, {
                "asset_uuid": auid, "natural_key": f"{kind.value}:{item_id}", "kind": kind.value,
                "display_name": item_id, "slug": f"{kind.value}:{item_id}", "scene_text": None,
            })

            session_assets.append({"session_uuid": suid, "asset_uuid": auid, "position": position})

            # One user↔item interaction edge, deduped across backpacks (keep the earliest timestamp).
            ts = join.get("inserted_at")
            edge = edges.get((owner_uuid, auid))
            if edge is None:
                edges[(owner_uuid, auid)] = {"user_uuid": owner_uuid, "asset_uuid": auid,
                                             "ts_first": ts, "n_backpacks": 1}
            else:
                edge["n_backpacks"] += 1
                if ts and (edge["ts_first"] is None or ts < edge["ts_first"]):
                    edge["ts_first"] = ts

    user_rows, asset_rows, edge_rows = list(users.values()), list(assets.values()), list(edges.values())

    # Persist each relation with its primary key, then assert every foreign key resolves.
    write_relation("users", user_rows, out_dir, pk=["user_uuid"])
    write_relation("assets_uro", asset_rows, out_dir, pk=["asset_uuid"])
    write_relation("sessions_uro", sessions, out_dir, pk=["session_uuid"])
    write_relation("session_assets_uro", session_assets, out_dir, pk=["session_uuid", "asset_uuid"])
    write_relation("user_item_edges", edge_rows, out_dir, pk=["user_uuid", "asset_uuid"])

    check_fk(sessions, "owner_uuid", user_rows, "user_uuid")
    check_fk(session_assets, "session_uuid", sessions, "session_uuid")
    check_fk(session_assets, "asset_uuid", asset_rows, "asset_uuid")
    check_fk(edge_rows, "user_uuid", user_rows, "user_uuid")
    check_fk(edge_rows, "asset_uuid", asset_rows, "asset_uuid")

    return {"users": len(user_rows), "assets_uro": len(asset_rows), "sessions_uro": len(sessions),
            "session_assets_uro": len(session_assets), "user_item_edges": len(edge_rows)}
