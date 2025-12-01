"""
Regression test for Union type extraction bug.

Bug: extract_type() only handled types.UnionType (A | B syntax) but not
typing.Union (returned by get_type_hints()). This caused extraction to fail
when processing field annotations in Node classes.

Fixed by checking both isinstance(py_type, types.UnionType) and origin is Union.
"""

from typing import get_type_hints
from nanodsl.nodes import Node
from nanodsl.types import TypeDef, UnionType, NodeType
from nanodsl.schema import extract_type
from nanodsl.serialization import to_dict


@TypeDef.register
class TypeA:
    """Test type A."""
    pass


@TypeDef.register
class TypeB:
    """Test type B."""
    pass


class NodeA(Node[TypeA], tag="node_a"):
    """Node returning TypeA."""
    value: int


class NodeB(Node[TypeB], tag="node_b"):
    """Node returning TypeB."""
    value: str


class TestUnionExtractionFromTypeHints:
    """Test that Union types work when extracted via get_type_hints()."""

    def test_union_with_pipe_operator(self):
        """Test that A | B syntax works."""
        # Direct extraction with | operator
        result = extract_type(int | str)

        assert isinstance(result, UnionType)
        assert len(result.options) == 2

    def test_union_from_get_type_hints(self):
        """
        Test that Union works when extracted from get_type_hints().

        This is the bug that was fixed - get_type_hints() returns typing.Union,
        not types.UnionType, which wasn't being handled.
        """

        class TestNode(Node[TypeA], tag="test_node"):
            field: Node[TypeA] | Node[TypeB]  # Union in annotation

        # When we get type hints, Python converts | to typing.Union
        hints = get_type_hints(TestNode)

        # This used to raise ValueError: Cannot extract type from: typing.Union[...]
        result = extract_type(hints["field"])

        assert isinstance(result, UnionType)
        assert len(result.options) == 2

        # Check the options are correct Node types
        assert all(isinstance(opt, NodeType) for opt in result.options)

        # Serialize to verify structure
        serialized = to_dict(result)
        assert serialized["tag"] == "std.union"
        assert len(serialized["options"]) == 2

    def test_union_in_node_field_real_world(self):
        """Test a real-world scenario: node with union field."""

        class Processor(Node[TypeA], tag="processor"):
            """Processes either TypeA or TypeB nodes."""

            source: Node[TypeA] | Node[TypeB]

        # Should not raise during type extraction
        hints = get_type_hints(Processor)
        source_type = extract_type(hints["source"])

        assert isinstance(source_type, UnionType)

        # Verify we can construct valid instances
        node_a = NodeA(value=42)
        node_b = NodeB(value="test")

        # Both should be valid (though Python won't enforce at runtime)
        proc_a = Processor(source=node_a)
        proc_b = Processor(source=node_b)

        assert proc_a.source == node_a
        assert proc_b.source == node_b

    def test_nested_union_extraction(self):
        """Test union with nested generic types."""

        class ComplexNode(Node[TypeA], tag="complex"):
            data: list[Node[TypeA]] | dict[str, Node[TypeB]]

        hints = get_type_hints(ComplexNode)
        data_type = extract_type(hints["data"])

        assert isinstance(data_type, UnionType)
        # Should have list and dict as options
        assert len(data_type.options) == 2
