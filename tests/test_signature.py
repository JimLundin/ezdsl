"""Tests for node signature composition."""

from typedsl.nodes import Node, Signature
from typedsl.schema import node_schema
from typedsl.serialization import to_dict


class TestSignatureBasics:
    """Test basic signature functionality."""

    def test_signature_creation_with_args(self) -> None:
        """Signature can be created with positional args."""
        sig = Signature(args=("math", "add"), kwargs={})
        assert sig.args == ("math", "add")
        assert sig.kwargs == {}

    def test_signature_creation_with_kwargs(self) -> None:
        """Signature can be created with keyword args."""
        sig = Signature(args=(), kwargs={"ns": "math", "name": "add"})
        assert sig.args == ()
        assert sig.kwargs == {"ns": "math", "name": "add"}

    def test_signature_creation_with_mixed(self) -> None:
        """Signature can be created with both args and kwargs."""
        sig = Signature(args=("math",), kwargs={"name": "add", "version": "1.0"})
        assert sig.args == ("math",)
        assert sig.kwargs == {"name": "add", "version": "1.0"}

    def test_signature_compose_with_args(self) -> None:
        """Signature composes args into tag with delimiter."""
        sig = Signature(args=("math", "add", "v1"), kwargs={})
        assert sig.compose(".") == "math.add.v1"

    def test_signature_compose_with_kwargs(self) -> None:
        """Signature composes kwargs into tag with insertion order."""
        sig = Signature(args=(), kwargs={"ns": "math", "name": "add"})
        assert sig.compose(".") == "math.add"

    def test_signature_compose_with_mixed(self) -> None:
        """Signature composes args first, then kwargs."""
        sig = Signature(args=("math", "add"), kwargs={"version": "1.0"})
        assert sig.compose(".") == "math.add.1.0"

    def test_signature_compose_empty(self) -> None:
        """Empty signature composes to empty string."""
        sig = Signature(args=(), kwargs={})
        assert sig.compose(".") == ""

    def test_signature_compose_custom_delimiter(self) -> None:
        """Signature can use custom delimiter."""
        sig = Signature(args=("math", "add"), kwargs={})
        assert sig.compose(":") == "math:add"
        assert sig.compose("/") == "math/add"
        assert sig.compose("-") == "math-add"


class TestNodeWithSignature:
    """Test nodes with signature parameters."""

    def test_node_with_single_kwarg(self) -> None:
        """Node can use single kwarg for signature."""

        class SimpleNode(Node[int], tag="simple_sig"):
            value: int

        assert SimpleNode._tag == "simple_sig"
        assert SimpleNode._signature.args == ()
        assert SimpleNode._signature.kwargs == {"tag": "simple_sig"}

    def test_node_with_multiple_kwargs(self) -> None:
        """Node can use multiple kwargs for signature."""

        class AddNode(Node[int], ns="math", name="add"):
            left: int
            right: int

        assert AddNode._tag == "math.add"
        assert AddNode._signature.args == ()
        assert AddNode._signature.kwargs == {"ns": "math", "name": "add"}

    def test_node_with_three_part_signature(self) -> None:
        """Node can use three-part signature."""

        class MulNode(Node[int], ns="math", name="mul", version="1.0"):
            left: int
            right: int

        assert MulNode._tag == "math.mul.1.0"
        assert MulNode._signature.args == ()
        assert MulNode._signature.kwargs == {
            "ns": "math",
            "name": "mul",
            "version": "1.0",
        }

    def test_node_without_signature_uses_class_name(self) -> None:
        """Node without signature gets auto-generated tag from class name."""

        class ComputerNode(Node[int]):
            value: int

        assert ComputerNode._tag == "computer"
        assert ComputerNode._signature.args == ()
        assert ComputerNode._signature.kwargs == {}

    def test_node_signature_with_numbers(self) -> None:
        """Node signature can include numbers."""

        class VersionedNode(Node[int], name="my_node", major="2", status="stable"):
            value: int

        assert VersionedNode._tag == "my_node.2.stable"

    def test_node_signature_kwargs_preserve_insertion_order(self) -> None:
        """Kwargs in signature preserve insertion order."""

        class OrderedNode(Node[int], z="last", a="first", m="middle"):
            value: int

        # Python 3.7+ dicts preserve insertion order
        assert OrderedNode._tag == "last.first.middle"
        assert list(OrderedNode._signature.kwargs.keys()) == ["z", "a", "m"]


class TestNodeSchemaWithSignature:
    """Test schema extraction with signatures."""

    def test_schema_includes_signature_kwargs(self) -> None:
        """Node schema includes signature kwargs."""

        class SchemaAddNode(Node[int], ns="math", name="add", test="schema"):
            left: int
            right: int

        schema = node_schema(SchemaAddNode)
        assert schema.tag == "math.add.schema"
        assert schema.signature.args == ()
        assert schema.signature.kwargs == {"ns": "math", "name": "add", "test": "schema"}

    def test_schema_includes_multi_part_signature(self) -> None:
        """Node schema includes multi-part signature."""

        class SubNode(Node[int], ns="calc", name="sub", version="2.0"):
            left: int
            right: int

        schema = node_schema(SubNode)
        assert schema.tag == "calc.sub.2.0"
        assert schema.signature.args == ()
        assert schema.signature.kwargs == {
            "ns": "calc",
            "name": "sub",
            "version": "2.0",
        }

    def test_schema_with_empty_signature(self) -> None:
        """Node schema works with empty signature (auto-generated tag)."""

        class AutoNode(Node[int]):
            value: int

        schema = node_schema(AutoNode)
        assert schema.tag == "auto"
        assert schema.signature.args == ()
        assert schema.signature.kwargs == {}


class TestSignatureSerialization:
    """Test signature serialization."""

    def test_serialize_node_schema_with_signature(self) -> None:
        """Serialized schema includes signature metadata."""

        class MathAdd(Node[float], ns="math", name="add", version="1.0"):
            left: float
            right: float

        schema = node_schema(MathAdd)
        from typedsl.adapters import JSONAdapter

        adapter = JSONAdapter()
        serialized = adapter.serialize_node_schema(schema)

        assert serialized["tag"] == "math.add.1.0"
        assert serialized["signature"]["args"] == []
        assert serialized["signature"]["kwargs"] == {
            "ns": "math",
            "name": "add",
            "version": "1.0",
        }

    def test_serialize_node_instance_uses_composed_tag(self) -> None:
        """Node instances serialize with composed tag, not signature parts."""

        class CalcOp(Node[int], pkg="calculator", name="operation", version="2.0"):
            op: str
            value: int

        node = CalcOp(op="add", value=42)
        result = to_dict(node)

        # Instance serialization uses only the composed tag
        assert result["tag"] == "calculator.operation.2.0"
        # Signature parts are NOT in instance serialization
        assert "signature" not in result

    def test_signature_in_schema_not_in_instance(self) -> None:
        """Signature appears in schema but not in node instances."""

        class TypedNode(Node[str], ns="mylib", name="typed"):
            data: str

        # Schema has signature
        schema = node_schema(TypedNode)
        assert schema.signature.args == ()
        assert schema.signature.kwargs == {"ns": "mylib", "name": "typed"}

        # Instance does not
        instance = TypedNode(data="test")
        result = to_dict(instance)
        assert result == {"tag": "mylib.typed", "data": "test"}


class TestSignatureEdgeCases:
    """Test edge cases for signatures."""

    def test_signature_with_special_characters(self) -> None:
        """Signature parts can contain special characters."""

        class SpecialNode(Node[int], name="my-node", version="v1.0.0"):
            value: int

        assert SpecialNode._tag == "my-node.v1.0.0"

    def test_signature_with_unicode(self) -> None:
        """Signature parts can contain unicode."""

        class UnicodeNode(Node[int], ns="math", name="加法"):
            value: int

        assert UnicodeNode._tag == "math.加法"

    def test_empty_signature_parts_handled(self) -> None:
        """Empty strings in signature are preserved."""

        class EmptyPartNode(Node[int], ns="", name="node"):
            value: int

        assert EmptyPartNode._tag == ".node"

    def test_numeric_signature_values_converted_to_string(self) -> None:
        """Numeric values in signature are converted to strings."""

        class NumericSig(Node[int], name="node", major=42, minor=3.14):
            value: int

        assert NumericSig._tag == "node.42.3.14"
        assert NumericSig._signature.kwargs == {"name": "node", "major": 42, "minor": 3.14}


class TestSignatureCollisions:
    """Test signature collision detection."""

    def test_different_signatures_same_tag_raises_error(self) -> None:
        """Different signature patterns that produce same tag raise error."""
        import pytest

        class First(Node[int], ns="collision", name="test"):
            value: int

        # This would create the same tag
        with pytest.raises(ValueError, match="already registered"):

            class Second(Node[int], tag="collision.test"):  # noqa: F841
                value: int

    def test_same_signature_components_different_order_different_tags(self) -> None:
        """Changing order of kwargs changes the composed tag."""

        class OrderA(Node[int], a="1", b="2"):
            value: int

        class OrderB(Node[int], b="2", a="1"):
            value: int

        # Different insertion order = different tags
        assert OrderA._tag == "1.2"
        assert OrderB._tag == "2.1"
        assert OrderA._tag != OrderB._tag
