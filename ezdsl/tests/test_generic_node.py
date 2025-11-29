"""Test how generic Node definitions serialize their types."""

import pytest
from typing import TypeVar, get_type_hints

from ezdsl.schema import extract_type
from ezdsl.types import (
    ParameterizedType,
    TypeParameter,
    PrimitiveType,
)


def test_type_parameter_in_annotation():
    """Test extracting a type annotation that uses a type parameter."""
    # Simulate: class MyNode[T]:
    #               args: list[T]
    T = TypeVar("T")

    # What does list[T] extract to?
    result = extract_type(list[T])

    assert isinstance(result, ParameterizedType)
    assert result.origin.primitive == list
    assert len(result.args) == 1

    # The argument is the TypeParameter T, not a concrete type!
    assert isinstance(result.args[0], TypeParameter)
    assert result.args[0].name == "T"


def test_nested_type_parameters():
    """Test extracting nested parameterized types with type parameters."""
    # Simulate: class MyNode[T]:
    #               args: list[dict[str, T]]
    T = TypeVar("T")

    result = extract_type(list[dict[str, T]])

    # Outer: list[...]
    assert isinstance(result, ParameterizedType)
    assert result.origin.primitive == list

    # Middle: dict[str, T]
    dict_param = result.args[0]
    assert isinstance(dict_param, ParameterizedType)
    assert dict_param.origin.primitive == dict
    assert len(dict_param.args) == 2

    # First arg of dict is str (concrete)
    assert isinstance(dict_param.args[0], PrimitiveType)
    assert dict_param.args[0].primitive == str

    # Second arg of dict is T (type parameter)
    assert isinstance(dict_param.args[1], TypeParameter)
    assert dict_param.args[1].name == "T"


def test_bounded_type_parameter_in_annotation():
    """Test extracting type annotations with bounded type parameters."""
    # Simulate: class MyNode[T: int]:
    #               value: T
    T = TypeVar("T", bound=int)

    result = extract_type(T)

    assert isinstance(result, TypeParameter)
    assert result.name == "T"
    assert result.bound is not None
    assert isinstance(result.bound, PrimitiveType)
    assert result.bound.primitive == int


def test_multiple_type_parameters():
    """Test multiple type parameters in one annotation."""
    # Simulate: class MyNode[K, V]:
    #               data: dict[K, V]
    K = TypeVar("K")
    V = TypeVar("V")

    result = extract_type(dict[K, V])

    assert isinstance(result, ParameterizedType)
    assert result.origin.primitive == dict
    assert len(result.args) == 2

    # Both args are type parameters
    assert isinstance(result.args[0], TypeParameter)
    assert result.args[0].name == "K"

    assert isinstance(result.args[1], TypeParameter)
    assert result.args[1].name == "V"
