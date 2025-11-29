"""Test real-world example of generic Node with type parameter serialization."""

import pytest
from typing import get_type_hints

from ezdsl.nodes import Node, Ref
from ezdsl.schema import extract_type, node_schema
from ezdsl.types import (
    ParameterizedType,
    TypeParameter,
    NodeType,
    RefType,
    PrimitiveType,
)


def test_generic_node_field_extraction():
    """
    Test how a generic Node's fields are extracted.

    Example: class Container[T]:
                 items: list[T]

    When we extract the type of 'items', T should be a TypeParameter,
    not a concrete type.
    """

    # Define a generic node
    class Container(Node[list], tag="container"):
        """Container that holds a list of items of type T."""
        items: list[TypeParameter]  # This is conceptual - in reality we'd use a TypeVar

    # In practice, you'd define it like:
    # class Container[T](Node[list[T]], tag="container"):
    #     items: list[T]

    # But for this test, let's manually extract what list[T] would look like
    from typing import TypeVar
    T = TypeVar("T")

    items_type = extract_type(list[T])

    # The field type is a ParameterizedType (list with arg applied)
    assert isinstance(items_type, ParameterizedType)
    assert items_type.origin.primitive == list

    # The arg is a TypeParameter (the placeholder T), not a concrete type
    assert len(items_type.args) == 1
    assert isinstance(items_type.args[0], TypeParameter)
    assert items_type.args[0].name == "T"


def test_complex_generic_node_field():
    """
    Test: class MyNode[T]:
              args: list[Ref[Node[T]]]

    This should serialize as:
    - ParameterizedType for list[...]
      - args contains ParameterizedType for Ref[...]
        - args contains ParameterizedType for Node[...]
          - args contains TypeParameter(name="T")
    """
    from typing import TypeVar
    T = TypeVar("T")

    # Build the type annotation: list[Ref[Node[T]]]
    # Note: We can't actually use Ref[Node[T]] at runtime in tests easily,
    # so we'll use dict[str, T] as a proxy to show the nesting
    result = extract_type(list[dict[str, T]])

    # Outer layer: list
    assert isinstance(result, ParameterizedType)
    assert result.origin.primitive == list
    assert len(result.args) == 1

    # Middle layer: dict[str, T]
    dict_type = result.args[0]
    assert isinstance(dict_type, ParameterizedType)
    assert dict_type.origin.primitive == dict
    assert len(dict_type.args) == 2

    # dict's first arg: str (concrete type)
    assert isinstance(dict_type.args[0], PrimitiveType)
    assert dict_type.args[0].primitive == str

    # dict's second arg: T (type parameter)
    assert isinstance(dict_type.args[1], TypeParameter)
    assert dict_type.args[1].name == "T"
    assert dict_type.args[1].bound is None


def test_bounded_type_parameter_in_generic_node():
    """
    Test: class NumericNode[T: float]:
              value: T

    The TypeParameter should capture the bound.
    """
    from typing import TypeVar
    T = TypeVar("T", bound=float)

    result = extract_type(T)

    assert isinstance(result, TypeParameter)
    assert result.name == "T"
    assert result.bound is not None
    assert isinstance(result.bound, PrimitiveType)
    assert result.bound.primitive == float


def test_type_parameter_vs_concrete_type():
    """
    Demonstrate the difference between:
    1. A type parameter (T in class definition)
    2. A concrete type argument (int when using the class)
    """
    from typing import TypeVar
    T = TypeVar("T")

    # In the class definition: list[T]
    generic_form = extract_type(list[T])
    assert isinstance(generic_form, ParameterizedType)
    assert isinstance(generic_form.args[0], TypeParameter)  # T is a parameter
    assert generic_form.args[0].name == "T"

    # When using the class: list[int]
    concrete_form = extract_type(list[int])
    assert isinstance(concrete_form, ParameterizedType)
    assert isinstance(concrete_form.args[0], PrimitiveType)  # int is concrete
    assert concrete_form.args[0].primitive == int

    # They're structurally similar (both ParameterizedType)
    # but their args are different (TypeParameter vs PrimitiveType)
