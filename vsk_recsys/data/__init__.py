"""ETNF parquet feature-store layer. See decisions/20260712-parquet-feature-store-etnf.md."""

from .etnf import NAMESPACE, entity_uuid, asset_uuid, user_uuid, session_uuid

__all__ = ["NAMESPACE", "entity_uuid", "asset_uuid", "user_uuid", "session_uuid"]
