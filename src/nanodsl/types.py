"""
Type system domain for runtime type representation.

This module defines the runtime type representation system used for schema
generation and type extraction. It uses concrete types rather than generic
wrappers to provide clear, self-documenting type definitions.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import dataclass_transform, get_args, get_origin, Any, ClassVar

# =============================================================================
# Type Definition Base
# =============================================================================


@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}
    _custom_types: ClassVar[dict[type, type[TypeDef]]] = (
        {}
    )  # Maps Python types to TypeDef classes

    def __init_subclass__(cls, tag: str | None = None):
        # Always convert to frozen dataclass
        dataclass(frozen=True)(cls)

        # Determine tag
        cls._tag = tag or cls.__name__.lower().removesuffix("type")

        # Check for collisions
        if existing := TypeDef._registry.get(cls._tag):
            if existing is not cls:
                raise ValueError(
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    f"Choose a different tag."
                )

        TypeDef._registry[cls._tag] = cls

    @classmethod
    def register(
        cls,
        python_type: type | None = None,
        *,
        tag: str | None = None,
    ) -> type[TypeDef] | Any:
        """
        Register a custom type with the type system.

        This method can be used either as a decorator or as a regular function call
        to register existing types (like pandas DataFrame, numpy ndarray, etc.) with
        the DSL type system.

        Args:
            python_type: The Python class to register. If None, returns a decorator.
            tag: Optional tag name. Defaults to lowercase class name.

        Returns:
            If used as decorator: returns the original class unchanged
            If used as function: returns the created TypeDef subclass

        Examples:
            # Register an existing type (e.g., pandas DataFrame)
            >>> import pandas as pd
            >>> TypeDef.register(pd.DataFrame)  # tag="dataframe"
            >>> TypeDef.register(pd.DataFrame, tag="df")  # tag="df"

            # Use as decorator for marker classes
            >>> @TypeDef.register
            ... class GraphicsContext:
            ...     '''Marker for graphics context type.'''
            ...     pass

            # Use as decorator with custom tag
            >>> @TypeDef.register(tag="matrix")
            ... class Matrix:
            ...     pass
        """

        def _create_typedef(py_type: type) -> type[TypeDef]:
            """Create and register a TypeDef for the given Python type."""
            # Determine the tag
            type_tag = tag or py_type.__name__.lower()

            # Check if already registered
            if py_type in cls._custom_types:
                existing = cls._custom_types[py_type]
                # If same tag, it's idempotent
                if existing._tag == type_tag:
                    return existing
                raise ValueError(
                    f"Type {py_type} already registered with tag '{existing._tag}'. "
                    f"Cannot re-register with tag '{type_tag}'."
                )

            # Create TypeDef subclass dynamically
            typedef_name = f"{py_type.__name__}Type"

            # Build class dict
            class_dict = {
                "__module__": py_type.__module__,
                "__doc__": f"Custom type definition for {py_type.__name__}.",
            }

            # Create the class - type() will call __init_subclass__ with our kwargs
            typedef_cls = type(
                typedef_name, (TypeDef,), class_dict, tag=type_tag
            )

            # Register the mapping
            cls._custom_types[py_type] = typedef_cls

            return typedef_cls

        # If called without arguments as a decorator: @TypeDef.register
        if python_type is not None:
            _create_typedef(python_type)
            # When used as decorator, return the original class unchanged
            # This allows the marker class to still be used normally
            return python_type

        # If called with arguments: @TypeDef.register(tag="foo")
        # Return a decorator that will be applied to the class
        def decorator(py_type: type) -> type:
            _create_typedef(py_type)
            return py_type

        return decorator

    @classmethod
    def get_registered_type(cls, python_type: type) -> type[TypeDef] | None:
        """
        Get the registered TypeDef for a Python type.

        Args:
            python_type: The Python type to look up

        Returns:
            The TypeDef class registered for this type, or None if not registered
        """
        return cls._custom_types.get(python_type)


# =============================================================================
# Primitive Types (Concrete)
# =============================================================================


class IntType(TypeDef, tag="int"):
    """Integer type."""


class FloatType(TypeDef, tag="float"):
    """Floating point type."""


class StrType(TypeDef, tag="str"):
    """String type."""


class BoolType(TypeDef, tag="bool"):
    """Boolean type."""


class NoneType(TypeDef, tag="none"):
    """None/null type."""


# =============================================================================
# Container Types (Concrete)
# =============================================================================


class ListType(TypeDef, tag="list"):
    """
    List type with element type.

    Example: list[int] → ListType(element=IntType())
    """

    element: TypeDef


class DictType(TypeDef, tag="dict"):
    """
    Dictionary type with key and value types.

    Example: dict[str, int] → DictType(key=StrType(), value=IntType())
    """

    key: TypeDef
    value: TypeDef


class SetType(TypeDef, tag="set"):
    """
    Set type with element type.

    Example: set[int] → SetType(element=IntType())
    """

    element: TypeDef


class TupleType(TypeDef, tag="tuple"):
    """
    Fixed-length heterogeneous tuple type.

    Unlike list (homogeneous), tuple types have:
    - Fixed length (known at schema time)
    - Heterogeneous element types (each position can have different type)

    Examples:
        tuple[int, str, float] → TupleType(elements=(IntType(), StrType(), FloatType()))
        tuple[str, str, str] → TupleType(elements=(StrType(), StrType(), StrType()))
    """

    elements: tuple[TypeDef, ...]


class LiteralType(TypeDef, tag="literal"):
    """
    Literal type representing enumeration of values.

    Maps Python's Literal[...] type to enumeration schema.

    Examples:
        Literal["red", "green", "blue"] → LiteralType(values=("red", "green", "blue"))
        Literal[1, 2, 3] → LiteralType(values=(1, 2, 3))
        Literal[True, False] → LiteralType(values=(True, False))

    Note: Does not support Python enum.Enum at this stage.
    """

    values: tuple[str | int | bool, ...]


# =============================================================================
# Domain Types
# =============================================================================


class NodeType(TypeDef, tag="node"):
    """
    AST Node type with return type.

    Example: Node[float] → NodeType(returns=FloatType())
    """

    returns: TypeDef


class RefType(TypeDef, tag="ref"):
    """
    Reference type pointing to another type.

    Example: Ref[Node[int]] → RefType(target=NodeType(returns=IntType()))
    """

    target: TypeDef


class UnionType(TypeDef, tag="union"):
    """
    Union of multiple types.

    Example: int | str → UnionType(options=(IntType(), StrType()))
    """

    options: tuple[TypeDef, ...]


class TypeParameter(TypeDef, tag="typeparam"):
    """
    Type parameter declaration in PEP 695 generic definitions.

    Represents the DECLARATION of a type parameter (e.g., in class Foo[T]).

    Examples:
        class Foo[T]: ... → TypeParameter(name="T", bound=None)
        class Foo[T: int | float]: ... → TypeParameter(name="T", bound=UnionType(...))

    This is the definition site of the type parameter.
    """

    name: str
    bound: TypeDef | None = None


class TypeParameterRef(TypeDef, tag="typeparamref"):
    """
    Reference to a type parameter within a type expression.

    Represents a USE of a type parameter (e.g., in field: T).

    Examples:
        In class Foo[T]:
            field: T → TypeParameterRef(name="T")
            field: list[T] → ListType(element=TypeParameterRef(name="T"))

    This is the use site that refers back to the TypeParameter declaration.
    """

    name: str


class ExternalType(TypeDef, tag="external"):
    """
    Reference to an externally registered type.

    Used for third-party types like pandas.DataFrame, polars.DataFrame, etc.
    Stores module and name to avoid collisions between different libraries.

    Examples:
        pd.DataFrame → ExternalType(module="pandas.core.frame", name="DataFrame", tag="pd_dataframe")
        pl.DataFrame → ExternalType(module="polars.dataframe.frame", name="DataFrame", tag="pl_dataframe")
    """

    module: str  # Full module path
    name: str    # Class name
    tag: str     # User-supplied tag


class CustomType(TypeDef, tag="custom"):
    """
    Reference to a user-defined custom type.

    Used for types registered with the decorator style, where encode/decode
    are methods on the class itself.

    Example:
        @TypeDef.register(tag="point")
        class Point: ...
        → CustomType(tag="point")
    """

    tag: str


# =============================================================================
# Type Parameter Substitution
# =============================================================================


def _substitute_type_params(type_expr: Any, substitutions: dict[Any, Any]) -> Any:
    """
    Recursively substitute type parameters in a type expression.

    Args:
        type_expr: The type expression to substitute in
        substitutions: Mapping from type parameters to their concrete types

    Returns:
        The type expression with parameters substituted
    """
    # If this is a type parameter, substitute it
    if type_expr in substitutions:
        return substitutions[type_expr]

    # Get origin and args for generic types
    origin = get_origin(type_expr)
    args = get_args(type_expr)

    # If no origin, this is a simple type - return as-is
    if origin is None:
        return type_expr

    # If there are no args, return as-is
    if not args:
        return type_expr

    # Recursively substitute in the arguments
    new_args = tuple(_substitute_type_params(arg, substitutions) for arg in args)

    # Handle UnionType (created by | operator) specially
    if isinstance(type_expr, types.UnionType):
        # Reconstruct union using | operator
        result = new_args[0]
        for arg in new_args[1:]:
            result = result | arg
        return result

    # Reconstruct the type with substituted arguments
    return origin[new_args]
