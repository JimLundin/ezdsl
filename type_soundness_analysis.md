# Type System Soundness Analysis

## Executive Summary

**Your type system is NOT sound at runtime**, but CAN be made sound with static type checking (mypy/pyright) + runtime validation. The current implementation provides **descriptive schemas** but no **prescriptive validation**.

## Current Architecture: Type-Based Composition

Your approach uses `Node[T]` where `T` represents the return type:

```python
class FetchData(Node[DataFrame], tag="fetch_data"):
    query: str

class FilterData(Node[DataFrame], tag="filter_data"):
    source: Node[DataFrame]  # Expects a node returning DataFrame
    condition: str
```

This is a **good design** that favors composition over inheritance! But it has soundness gaps.

---

## Soundness Issues Discovered

### ❌ Issue 1: No Runtime Type Parameter Validation

**Python does not enforce generic type parameters at runtime.**

```python
# This is Node[DataFrame], not Node[Matrix]
fetch = FetchData(query="SELECT *")

# But Python accepts it here anyway!
matrix_mult = MatrixMultiply(
    left=fetch,  # ❌ Type error, but Python allows it!
    right=fetch
)
```

**Expected:** Type error
**Actual:** Silently creates invalid AST

**Severity:** HIGH - Can create invalid ASTs that will fail later

---

### ❌ Issue 2: Union Types Accept Anything

```python
class ProcessData(Node[DataFrame], tag="process"):
    source: Node[DataFrame] | Node[Matrix]

# Works as expected
process1 = ProcessData(source=fetch_node)  # ✓
process2 = ProcessData(source=matrix_node)  # ✓

# Should fail, but doesn't!
image_node = LoadImage(path="/tmp/img.png")
process3 = ProcessData(source=image_node)  # ❌ But Python allows it!
```

**Severity:** MEDIUM - Less likely in practice but still a hole

---

### ❌ Issue 3: Ref Types Not Validated

```python
# This claims to reference a Matrix
ref = Ref[Node[Matrix]](id="node_123")

# But the ID might actually point to a DataFrame!
# No validation that the ref type matches the actual referenced node
matrix_mult = MatrixMultiply(left=ref, right=ref)
```

**Severity:** HIGH - Can cause runtime errors when dereferencing

---

### ❌ Issue 4: Deserialization Bypasses Type Checks

```python
# Manually crafted invalid data
corrupted = {
    "tag": "matrix_multiply",
    "left": {"tag": "fetch_data", "query": "SELECT *"},  # DataFrame!
    "right": {"tag": "fetch_data", "query": "SELECT *"}
}

# This deserializes successfully
invalid_node = from_dict(corrupted)  # ❌ Should reject this!
```

**Severity:** HIGH - External data can create invalid ASTs

---

## What Static Type Checkers Catch

If you run `mypy` or `pyright`, they **WILL** catch issues #1, #2, and #3:

```python
# mypy error: Argument 1 to "MatrixMultiply" has incompatible type
# "FetchData"; expected "Node[Matrix]"
matrix_mult = MatrixMultiply(left=fetch, right=fetch)
```

**But:** This requires:
1. Running the type checker (not automatic)
2. Full type annotations throughout the codebase
3. Users of your DSL must also run type checkers

---

## Soundness Levels

### Level 0: Current State (Unsound)
- ❌ No runtime validation
- ❌ Type errors slip through
- ✓ Fast (no overhead)
- ✓ Schemas document expected types

### Level 1: Static Type Checking (Sound at Development Time)
- ✓ mypy/pyright catches type errors
- ❌ No runtime protection
- ❌ Requires user discipline
- ✓ Zero runtime overhead

### Level 2: Runtime Validation (Sound at Runtime)
- ✓ Catches errors even without static checking
- ✓ Validates deserialized data
- ✓ Fail-fast behavior
- ❌ Performance overhead
- ✓ Better error messages

### Level 3: Combined (Best)
- ✓ Static checking during development
- ✓ Runtime validation as safety net
- ✓ Validates external data
- ❌ Some performance overhead

---

## Recommended Solution: Add Runtime Validation

### Option A: Validate at Construction Time

```python
@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    def __post_init__(self):
        """Validate field types match schema."""
        if not hasattr(self, '_skip_validation'):
            self._validate_fields()

    def _validate_fields(self):
        """Check that field values match type annotations."""
        hints = get_type_hints(type(self))

        for field in dc_fields(self):
            if field.name.startswith('_'):
                continue

            expected_typedef = extract_type(hints[field.name])
            actual_value = getattr(self, field.name)

            if not self._check_value_matches_type(actual_value, expected_typedef):
                raise TypeError(
                    f"Field '{field.name}' of {type(self).__name__} "
                    f"expects {self._format_type(expected_typedef)}, "
                    f"but got {self._format_actual(actual_value)}"
                )

    def _check_value_matches_type(self, value: Any, typedef: TypeDef) -> bool:
        """Check if a runtime value matches a TypeDef."""
        from nanodsl.types import NodeType, RefType, UnionType

        if isinstance(typedef, NodeType):
            # Value must be a Node with matching return type
            if not isinstance(value, Node):
                return False

            actual_returns = _extract_node_returns(type(value))
            return self._types_equal(actual_returns, typedef.returns)

        elif isinstance(typedef, RefType):
            # Value must be a Ref (we can't validate the target without resolution)
            return isinstance(value, Ref)

        elif isinstance(typedef, UnionType):
            # Value must match at least one option
            return any(
                self._check_value_matches_type(value, option)
                for option in typedef.options
            )

        # Add more cases for other types...
        return True

    def _types_equal(self, a: TypeDef, b: TypeDef) -> bool:
        """Check if two TypeDefs are equal."""
        return to_dict(a) == to_dict(b)
```

### Option B: Validate at Deserialization Time

```python
def from_dict(data: dict, registry: dict[str, type] = None, validate: bool = True) -> Node | Ref | TypeDef:
    """Deserialize from dict using tag lookups."""
    tag = data["tag"]

    if tag == "ref":
        return Ref[Any](id=data["id"])

    registry = registry or Node._registry
    cls = registry.get(tag) or TypeDef._registry.get(tag)

    if cls is None:
        raise ValueError(f"Unknown tag: {tag}")

    # Recursively deserialize fields
    kwargs = {}
    for field in dc_fields(cls):
        raw = data.get(field.name)
        kwargs[field.name] = _deserialize_value(raw, field.type)

    # Create instance
    instance = cls(**kwargs)

    # Validate if requested
    if validate and isinstance(instance, Node):
        instance._validate_fields()

    return instance
```

### Option C: Opt-in Validation Mode

```python
# config.py
class ValidationMode(Enum):
    NONE = "none"          # No validation (current behavior)
    DEVELOPMENT = "dev"    # Validate with warnings
    STRICT = "strict"      # Validate with errors

_validation_mode = ValidationMode.NONE

def set_validation_mode(mode: ValidationMode):
    global _validation_mode
    _validation_mode = mode

def get_validation_mode() -> ValidationMode:
    return _validation_mode
```

---

## Recommendations

### Short Term (Immediate)
1. **Document the soundness gap** in README/docs
2. **Recommend static type checking** to users (mypy/pyright)
3. **Add validation to `from_dict()`** (Option B) - prevents invalid external data

### Medium Term
4. **Add optional runtime validation** (Option A + Option C)
5. **Create strict mode** for development/testing
6. **Add validation utilities** for users to check AST validity

### Long Term
7. **Consider a type witness pattern** for Ref types
8. **Add constraint validation** for bounded type parameters
9. **Build AST validator** that walks the tree and checks all type relationships

---

## Alternative Approaches

### Approach 1: Phantom Types (More Complex)
```python
class NodeMetadata[T](TypedDict):
    returns: type[T]

class Node[T]:
    _metadata: ClassVar[NodeMetadata[T]]

    def __init_subclass__(cls, **kwargs):
        # Store metadata at class level
        cls._metadata = NodeMetadata(returns=...)
```

### Approach 2: Type Witnesses (Runtime Evidence)
```python
@dataclass(frozen=True)
class Ref[X]:
    id: str
    type_witness: TypeDef  # Store expected type

def resolve_ref(ref: Ref[X], ast: AST) -> Node[X]:
    """Resolve ref and validate type matches witness."""
    node = ast.get_node(ref.id)
    actual_type = _extract_node_returns(type(node))

    if to_dict(actual_type) != to_dict(ref.type_witness):
        raise TypeError(f"Ref type mismatch: expected {ref.type_witness}, got {actual_type}")

    return node
```

### Approach 3: Builder Pattern with Validation
```python
class ASTBuilder:
    """Type-safe AST builder that validates as you build."""

    def add_node(self, node: Node[T], *, id: str | None = None) -> Ref[Node[T]]:
        """Add node and return validated reference."""
        # Validate node
        node._validate_fields()

        # Store with metadata
        node_id = id or self._generate_id()
        self._nodes[node_id] = (node, _extract_node_returns(type(node)))

        # Return typed reference with witness
        returns = _extract_node_returns(type(node))
        return Ref[Node[T]](id=node_id, type_witness=returns)
```

---

## Conclusion

**Your type system design is theoretically sound** - the return-type-based approach is excellent! But **the implementation lacks runtime enforcement**, which means:

- ✅ **Sound with static type checking** (mypy/pyright)
- ❌ **Unsound at runtime** (Python doesn't enforce generics)
- ⚠️ **Vulnerable to invalid data** (deserialization bypass)

**Recommended Path Forward:**
1. Add validation to `from_dict()` immediately (prevents external attacks)
2. Add opt-in runtime validation for development
3. Document requirement for static type checking in production code
4. Consider making validation mandatory in future major version

This gives you defense-in-depth: static checking during development, runtime validation for external data, and optional strict mode for testing.
