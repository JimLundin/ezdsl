"""Core AST node infrastructure with automatic registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, dataclass_transform


@dataclass(frozen=True)
class Ref[X]:
    """Reference to X by ID."""

    id: str


@dataclass(frozen=True)
class Signature:
    """Node signature composed of positional and keyword arguments."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]  # Preserves insertion order (Python 3.7+)

    def compose(self, delimiter: str = ".") -> str:
        """Compose signature parts into a tag string."""
        parts = list(self.args) + list(self.kwargs.values())
        return delimiter.join(str(p) for p in parts) if parts else ""


@dataclass(frozen=True)
@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    _tag: ClassVar[str]
    _signature: ClassVar[Signature]
    registry: ClassVar[dict[str, type[Node[Any]]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        dataclass(frozen=True)(cls)

        # Store signature (kwargs only, preserves insertion order)
        cls._signature = Signature(args=(), kwargs=kwargs)

        # Compose tag from signature (fixed "." delimiter)
        composed = cls._signature.compose(".")
        cls._tag = composed if composed else cls.__name__.lower().removesuffix("node")

        if existing := Node.registry.get(cls._tag):
            if existing is not cls:
                msg = (
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    "Choose a different tag."
                )
                raise ValueError(msg)

        Node.registry[cls._tag] = cls


type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
