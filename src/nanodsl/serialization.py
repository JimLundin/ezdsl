"""Serialization functions for Node, Ref, and TypeDef."""

from __future__ import annotations

import json
from typing import Any

from nanodsl.adapters import JSONAdapter
from nanodsl.nodes import Node, Ref
from nanodsl.types import TypeDef

_adapter = JSONAdapter()


def to_dict(obj: Node[Any] | Ref[Any] | TypeDef) -> dict[str, Any]:
    """Serialize to dict."""
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}
    if isinstance(obj, Node):
        return _adapter.serialize_node(obj)
    if isinstance(obj, TypeDef):
        return _adapter.serialize_typedef(obj)
    msg = f"Cannot serialize {type(obj)}"
    raise ValueError(msg)


def from_dict(data: dict[str, Any]) -> Node[Any] | Ref[Any] | TypeDef:
    """Deserialize from dict."""
    tag = data["tag"]
    if tag == "ref":
        return Ref[Any](id=data["id"])
    if tag in Node.registry:
        return _adapter.deserialize_node(data)
    if tag in TypeDef.registry:
        return _adapter.deserialize_typedef(data)
    msg = f"Unknown tag: {tag}"
    raise ValueError(msg)


def to_json(obj: Node[Any] | Ref[Any] | TypeDef) -> str:
    """Serialize to JSON string."""
    return json.dumps(to_dict(obj), indent=2)


def from_json(s: str) -> Node[Any] | Ref[Any] | TypeDef:
    """Deserialize from JSON string."""
    return from_dict(json.loads(s))
