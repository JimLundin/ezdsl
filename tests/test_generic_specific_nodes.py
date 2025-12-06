"""
Test that generic specific node types work: e.g., Literal[float]

This validates combining both:
- Specific node class (Literal)
- Specific type parameter (float)
"""

from __future__ import annotations

from typing import cast

from nanodsl.nodes import Node, Ref
from nanodsl.schema import node_schema
from nanodsl.types import FloatType, IntType, NodeType, StrType
from nanodsl.ast import AST


# Define a GENERIC specific node type
class Literal[T](Node[T]):
    """A generic literal that can hold any type."""

    value: T


class Variable[T](Node[T]):
    """A generic variable that can reference any type."""

    name: str


# Now use it with specific type parameters
class TypedBinaryOp[T](Node[T]):
    """Binary operation with specific typed children."""

    left: Literal[T]  # Specific node type WITH type parameter
    right: Literal[T]
    operator: str


class MixedTypedOp(Node[float]):
    """Operation mixing different typed literals."""

    float_literal: Literal[float]  # Specific type parameter
    int_literal: Literal[int]  # Different type parameter
    operator: str


class GenericContainer[T](Node[T]):
    """Container with generic specific children."""

    literal: Literal[T]
    variable: Variable[T]


def test_generic_specific_instantiation() -> None:
    """Test that generic specific types work at runtime."""
    # Create typed literals
    float_lit = Literal[float](value=3.14)
    int_lit = Literal[int](value=42)
    str_lit = Literal[str](value="hello")

    # Verify types
    assert float_lit.value == 3.14
    assert int_lit.value == 42
    assert str_lit.value == "hello"

    # Use in typed binary op
    add_op = TypedBinaryOp[float](
        left=Literal[float](value=1.5), right=Literal[float](value=2.5), operator="+"
    )

    assert add_op.left.value == 1.5
    assert add_op.right.value == 2.5


def test_mixed_type_parameters() -> None:
    """Test using the same generic node with different type parameters."""
    mixed = MixedTypedOp(
        float_literal=Literal[float](value=3.14),
        int_literal=Literal[int](value=42),
        operator="mix",
    )

    assert mixed.float_literal.value == 3.14
    assert mixed.int_literal.value == 42

    # Type safety: float_literal is Literal[float], int_literal is Literal[int]
    assert isinstance(mixed.float_literal.value, float)
    assert isinstance(mixed.int_literal.value, int)


def test_type_extraction_for_generic_specific_nodes() -> None:
    """Test that type extraction handles generic specific types."""
    schema = node_schema(MixedTypedOp)

    float_field = next(f for f in schema.fields if f.name == "float_literal")
    int_field = next(f for f in schema.fields if f.name == "int_literal")

    # Both should be NodeType
    assert isinstance(float_field.type, NodeType)
    assert isinstance(int_field.type, NodeType)

    # Should correctly extract the type parameters
    assert isinstance(float_field.type.returns, FloatType)
    assert isinstance(int_field.type.returns, IntType)


def test_generic_container_instantiation() -> None:
    """Test generic containers with generic specific children."""
    # Float container
    float_container = GenericContainer[float](
        literal=Literal[float](value=3.14), variable=Variable[float](name="x")
    )

    assert float_container.literal.value == 3.14
    assert float_container.variable.name == "x"

    # String container
    str_container = GenericContainer[str](
        literal=Literal[str](value="hello"), variable=Variable[str](name="msg")
    )

    assert str_container.literal.value == "hello"
    assert str_container.variable.name == "msg"


def test_references_to_generic_specific_types() -> None:
    """Test that references work with generic specific types."""

    class RefContainer(Node[float]):
        float_lit_ref: Ref[Literal[float]]
        int_lit_ref: Ref[Literal[int]]

    float_lit = Literal[float](value=3.14)
    int_lit = Literal[int](value=42)

    container = RefContainer(
        float_lit_ref=Ref[Literal[float]](id="float_lit"),
        int_lit_ref=Ref[Literal[int]](id="int_lit"),
    )

    ast = AST(
        root="container",
        nodes={
            "container": container,
            "float_lit": float_lit,
            "int_lit": int_lit,
        },
    )

    # Resolve references
    resolved_float = ast.resolve(container.float_lit_ref)
    resolved_int = ast.resolve(container.int_lit_ref)

    # Type inference should work!
    assert resolved_float.value == 3.14
    assert resolved_int.value == 42


def test_type_parameter_inference() -> None:
    """Test that type parameters are correctly inferred."""
    # For a concrete (non-generic) class that uses generic specific types
    schema = node_schema(MixedTypedOp)

    # Fields should be NodeType with correct return types
    left_field = next(f for f in schema.fields if f.name == "float_literal")
    assert isinstance(left_field.type, NodeType)
    assert isinstance(left_field.type.returns, FloatType)

    right_field = next(f for f in schema.fields if f.name == "int_literal")
    assert isinstance(right_field.type, NodeType)
    assert isinstance(right_field.type.returns, IntType)


def test_multiple_generic_nodes_same_container() -> None:
    """Test using multiple generic nodes in one container."""

    class MultiGeneric(Node[str]):
        float_val: Literal[float]
        int_val: Literal[int]
        str_val: Literal[str]

    multi = MultiGeneric(
        float_val=Literal[float](value=3.14),
        int_val=Literal[int](value=42),
        str_val=Literal[str](value="hello"),
    )

    assert multi.float_val.value == 3.14
    assert multi.int_val.value == 42
    assert multi.str_val.value == "hello"

    # Each should maintain its own type
    assert isinstance(multi.float_val.value, float)
    assert isinstance(multi.int_val.value, int)
    assert isinstance(multi.str_val.value, str)


def test_documentation_example_generic_specific() -> None:
    """Example showing generic specific types for documentation."""

    # Define a generic literal
    class Value[T](Node[T]):
        data: T

    # Use it with specific type parameters in different contexts
    class Calculator(Node[float]):
        operand1: Value[float]  # Specific: Value class with float parameter
        operand2: Value[float]

    class Message(Node[str]):
        greeting: Value[str]  # Same generic class, different parameter

    # Both work!
    calc = Calculator(operand1=Value[float](data=1.5), operand2=Value[float](data=2.5))

    msg = Message(greeting=Value[str](data="Hello"))

    assert calc.operand1.data == 1.5
    assert msg.greeting.data == "Hello"

    # Type system knows the difference:
    # calc.operand1 is Value[float]
    # msg.greeting is Value[str]
