# DSL Node Type System Design

**Target Python Version:** 3.12+

## Overview

This document describes the design of a type-safe node system for building abstract syntax trees (ASTs) and domain-specific languages (DSLs). The system provides automatic registration, serialization, and schema generation for node types.

The framework enables users to define **nodes** (computation/structure) parameterized by **types** (data containers). Serialization is handled by pluggable **format adapters**.

---

## Core Concepts

### Types vs Nodes

**Types** are data containers. They describe the shape of values that flow between nodes or are embedded within nodes.

```
Types = Python built-ins + User-registered types
      = int, str, float, bool, None, list, dict, ...
      + DataFrame, NDArray, CustomClass, ...
```

**Nodes** are AST elements. They represent computation or structure. Every node is parameterized by the type it produces:

```python
class Add(Node[int]):         # produces int
class Filter(Node[list[T]]):  # produces list[T]
class Query(Node[DataFrame]): # produces DataFrame
```

Node fields can be:
- `Node[T]` — a child node producing type T
- `list[Node[T]]` — multiple child nodes
- `T` — an embedded value of type T (data, not computation)

---

## Type System

### Built-in Types

Python built-ins are always available. No registration required:

- **Primitives**: `int`, `float`, `str`, `bool`, `None`
- **Containers**: `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]`
- **Unions**: `T | U` or `Union[T, U]`

### Unregistered Types

Types that only flow between nodes at runtime. **No registration required.** Just use them:

```python
class DBConnection:
    pass

class Connect(Node[DBConnection]):
    connection_string: str

class Query(Node[DataFrame]):
    connection: Node[DBConnection]  # just works
```

Unregistered types can appear in `Node[T]` but **cannot be embedded as values** in node fields.

### Registered Types

Types that need to be embedded as values in nodes require registration with encoding/decoding:

```python
TypeDef.register(
    pd.DataFrame,
    tag="dataframe",
    namespace="custom",
)
```

After registration, the type can be used both in `Node[T]` and as embedded field values.

**Current Implementation Note**: The current implementation supports registration via `TypeDef.register()` but encoding/decoding functions are not yet implemented. Registration creates a TypeDef subclass that enables schema generation and serialization of the type reference.

### Type Registration

```python
class TypeDef:
    @classmethod
    def register(
        cls,
        python_type: type | None = None,
        *,
        tag: str | None = None,
        namespace: str = "custom",
    ) -> type[TypeDef] | Any:
        """
        Register a custom type with the type system.

        - tag: identifier for schema/serialization (defaults to lowercase class name)
        - namespace: namespace for the tag (defaults to "custom")

        Can be used as decorator or function call:
            @TypeDef.register
            class MyType: ...

            TypeDef.register(pd.DataFrame, tag="df")
        """
        ...

    @classmethod
    def get_registered_type(cls, python_type: type) -> type[TypeDef] | None:
        """Get the registered TypeDef for a Python type."""
        ...
```

---

## Node System

### Core Pattern

Both `Node` and `TypeDef` use the same pattern:

- Inherit from base class
- Optionally specify `tag` and `namespace` in class definition
- Automatically becomes a frozen dataclass
- Automatically registered in a central registry

### Node Base Class

```python
@dataclass_transform(frozen_default=True)
class Node[T]:
    _tag: ClassVar[str]
    _namespace: ClassVar[str]
    _registry: ClassVar[dict[str, type[Node]]] = {}

    def __init_subclass__(cls, tag: str | None = None, namespace: str | None = None):
        dataclass(frozen=True)(cls)
        cls._namespace = namespace or ""
        base_tag = tag or cls.__name__.lower()
        cls._tag = f"{namespace}.{base_tag}" if namespace else base_tag
        Node._registry[cls._tag] = cls
```

**Key Features:**
- Generic type parameter `T` represents the node's return/value type
- `_tag` identifies the node type for serialization
- `_namespace` provides organizational grouping and prevents collisions
- `_registry` enables dynamic lookup and deserialization
- `__init_subclass__` hook automates dataclass conversion and registration
- Frozen by default ensures immutability

### Namespaces

Namespaces are a **central feature** of the tagging system. They prevent collisions and provide organizational structure:

```python
class Add(Node[float], tag="add", namespace="math"):
    left: Node[float]
    right: Node[float]
# Full tag: "math.add"

class Add(Node[str], tag="add", namespace="string"):
    parts: list[Node[str]]
# Full tag: "string.add"
```

**Standard Namespaces:**
- `std` — Standard/built-in types (IntType, FloatType, ListType, etc.)
- `custom` — Default namespace for registered user types
- User-defined — Any custom namespace for domain-specific nodes

### Node Definition

```python
# Simple node
class Literal(Node[float], tag="literal"):
    value: float

# Node with child references
class Add(Node[float], tag="add"):
    left: Child[float]
    right: Child[float]

# Generic node (unbounded)
class Map[E, R](Node[list[R]]):
    input: Node[list[E]]
    func: Node[R]

# Generic node with bounds (Python 3.12+ syntax)
class Add[T: int | float](Node[T]):
    left: Node[T]
    right: Node[T]

# Generic node with bounds (older syntax)
T = TypeVar('T', bound=int | float)

class Add(Node[T]):
    left: Node[T]
    right: Node[T]
```

### Type Aliases

```python
type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
```

**Purpose:**
- `NodeRef[T]`: Explicitly represents a reference to a node
- `Child[T]`: Convenient union type for inline nodes or references

### Field Types

| Annotation | Meaning | Schema |
|------------|---------|--------|
| `Node[T]` | Child node producing T | `NodeType(returns=...)` |
| `list[Node[T]]` | Multiple children | `ListType(element=NodeType(...))` |
| `T` (registered type) | Embedded value | `PrimitiveType`, `CustomType`, etc. |

---

## Schema Representation

Schemas are represented as **dataclasses** for type safety and clarity. Export to JSON/YAML/etc. is handled by format adapters.

### Type Schema Dataclasses

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TypeSchema:
    """Base for all type schemas."""
    pass

@dataclass(frozen=True)
class PrimitiveType(TypeSchema):
    """int, float, str, bool, None"""
    name: str  # "int", "float", "str", "bool", "none"

@dataclass(frozen=True)
class ListType(TypeSchema):
    element: TypeSchema

@dataclass(frozen=True)
class DictType(TypeSchema):
    key: TypeSchema
    value: TypeSchema

@dataclass(frozen=True)
class UnionType(TypeSchema):
    options: list[TypeSchema]

@dataclass(frozen=True)
class TypeVarRef(TypeSchema):
    """Reference to a type variable defined in scope."""
    name: str  # "T", "K", "V", etc.

@dataclass(frozen=True)
class RegisteredType(TypeSchema):
    """User-registered type."""
    name: str  # full tag like "custom.dataframe"

@dataclass(frozen=True)
class UnregisteredType(TypeSchema):
    """Type that only flows between nodes, never serialized as value."""
    name: str  # class name, e.g., "DBConnection"

@dataclass(frozen=True)
class NodeType(TypeSchema):
    """Reference to a node producing a type."""
    returns: TypeSchema
```

### Type Variable Definition

```python
@dataclass(frozen=True)
class TypeVarDef:
    """
    Definition of a type variable with optional bounds.

    Corresponds to Python's TypeVar:
    - T = TypeVar('T')                    → unbounded
    - T = TypeVar('T', bound=int | float) → upper bound

    Or Python 3.12+ syntax:
    - class Foo[T]: ...                   → unbounded
    - class Foo[T: int | float]: ...      → upper bound

    Note: For our purposes, bounds and constraints are treated equivalently.
    """
    name: str
    bound: TypeSchema | None = None  # upper bound: T must be compatible with this
```

### Field and Node Schema

```python
@dataclass(frozen=True)
class FieldSchema:
    name: str
    type: TypeSchema

@dataclass(frozen=True)
class NodeSchema:
    tag: str
    namespace: str
    type_params: list[TypeVarDef]  # type variable definitions with bounds
    returns: TypeSchema
    fields: list[FieldSchema]
```

### Schema Examples

#### Simple Types

```python
# int
PrimitiveType(name="int")

# list[int]
ListType(element=PrimitiveType(name="int"))

# int | str
UnionType(options=[
    PrimitiveType(name="int"),
    PrimitiveType(name="str")
])

# Reference to type variable T
TypeVarRef(name="T")
```

#### Node Schema Examples

**Non-generic node:**

```python
class Query(Node[DataFrame], namespace="db"):
    sql: str
    connection: Node[DBConnection]
```

Becomes:

```python
NodeSchema(
    tag="db.query",
    namespace="db",
    type_params=[],  # no type variables
    returns=RegisteredType(name="custom.dataframe"),
    fields=[
        FieldSchema(
            name="sql",
            type=PrimitiveType(name="str")
        ),
        FieldSchema(
            name="connection",
            type=NodeType(returns=UnregisteredType(name="DBConnection"))
        ),
    ]
)
```

**Bounded generic:**

```python
class Add[T: int | float](Node[T], namespace="math"):
    left: Node[T]
    right: Node[T]
```

Becomes:

```python
NodeSchema(
    tag="math.add",
    namespace="math",
    type_params=[
        TypeVarDef(
            name="T",
            bound=UnionType(options=[
                PrimitiveType(name="int"),
                PrimitiveType(name="float")
            ])
        ),
    ],
    returns=TypeVarRef(name="T"),
    fields=[
        FieldSchema(
            name="left",
            type=NodeType(returns=TypeVarRef(name="T"))
        ),
        FieldSchema(
            name="right",
            type=NodeType(returns=TypeVarRef(name="T"))
        ),
    ]
)
```

### Schema Conversion

**Current Implementation:**

```python
def extract_type(py_type: Any) -> TypeDef:
    """Convert a Python type hint to a TypeDef instance."""
    ...

def node_schema(cls: type[Node]) -> dict:
    """
    Get schema for a node class.

    Returns dict with structure:
    {
        "tag": str,
        "returns": dict,  # serialized TypeDef
        "fields": [{"name": str, "type": dict}, ...]
    }
    """
    ...

def all_schemas() -> dict:
    """Get all registered node schemas."""
    ...
```

**Future Design:**

The schema functions should be updated to return dataclass instances:

```python
def extract_type_schema(py_type: Any) -> TypeSchema:
    """Convert a Python type hint to a TypeSchema dataclass."""
    ...

def node_schema(cls: type[Node]) -> NodeSchema:
    """Extract schema from a Node subclass as dataclass."""
    ...
```

---

## Serialization

### Current Implementation

Simple, consistent serialization API. Pattern is `{"tag": cls._tag, **fields}`:

#### API

```python
to_dict(obj)   # Node | Ref | TypeDef -> dict
to_json(obj)   # Node | Ref | TypeDef -> str
from_dict(d)   # dict -> Node | Ref | TypeDef
from_json(s)   # str -> Node | Ref | TypeDef
```

#### Example

```python
node = Add(left=Literal(1.0), right=Literal(2.0))
data = to_dict(node)
# {"tag": "add", "left": {"tag": "literal", "value": 1.0}, "right": {"tag": "literal", "value": 2.0}}

restored = from_dict(data)
# Add(left=Literal(1.0), right=Literal(2.0))
```

### Format Adapters (Future Design)

Adapters will handle serialization to/from specific formats as a pluggable system. This replaces the current JSON-only focus.

#### Adapter Interface

```python
from abc import ABC, abstractmethod

class FormatAdapter(ABC):
    @abstractmethod
    def serialize_node(
        self,
        node: Node,
        type_registry: TypeRegistry,
    ) -> Any:
        """Serialize a node instance to the output format."""
        ...

    @abstractmethod
    def deserialize_node(
        self,
        data: Any,
        node_registry: dict[str, type[Node]],
        type_registry: TypeRegistry,
    ) -> Node:
        """Deserialize from the input format to a node instance."""
        ...

    @abstractmethod
    def serialize_type_schema(self, schema: TypeSchema) -> Any:
        """Serialize a type schema to the output format."""
        ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> Any:
        """Serialize a node schema to the output format."""
        ...
```

#### Usage Example

```python
# JSON adapter
json_adapter = JSONAdapter()
json_data = json_adapter.serialize_node(my_node, type_registry)

# YAML adapter
yaml_adapter = YAMLAdapter()
yaml_str = yaml_adapter.serialize_node(my_node, type_registry)

# Binary adapter
binary_adapter = BinaryAdapter()
binary_data = binary_adapter.serialize_node(my_node, type_registry)
```

---

## AST Container

Manages the complete abstract syntax tree with node storage and reference resolution.

```python
@dataclass
class AST:
    root: str
    nodes: dict[str, Node]

    def resolve[X](self, ref: Ref[X]) -> X: ...

    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...

    @classmethod
    def from_dict(cls, data: dict) -> AST: ...

    @classmethod
    def from_json(cls, s: str) -> AST: ...
```

**Responsibilities:**
- Store all nodes in a flat dictionary
- Provide reference resolution
- Maintain a single root entry point
- Enable serialization of cyclic graphs

---

## User Experience

### Defining Nodes

```python
from nanodsl import Node

# Simple node
class Constant(Node[float], tag="const", namespace="math"):
    value: float

# Generic node with bounds
class Add[T: int | float](Node[T], namespace="math"):
    left: Node[T]
    right: Node[T]
```

### Using Unregistered Types

No registration needed for types that only flow between nodes:

```python
class DBConnection:
    pass

class Connect(Node[DBConnection], namespace="db"):
    connection_string: str

class Query(Node[DataFrame], namespace="db"):
    connection: Node[DBConnection]  # unregistered type, works fine
```

### Registering Custom Types

Only when you need to embed values or want schema documentation:

```python
from nanodsl import TypeDef

@TypeDef.register
class DataFrame:
    """User-defined DataFrame type marker."""

# Or with custom tag
TypeDef.register(pd.DataFrame, tag="df")

class AnalyzeData(Node[dict[str, float]], namespace="analytics"):
    data: DataFrame  # registered type can be embedded as value
```

### Constructing ASTs

```python
ast = Add(
    left=Constant(value=5.0),
    right=Constant(value=3.0)
)
```

### Getting Schema

```python
from nanodsl import node_schema

schema = node_schema(Add)
# Returns dict with tag, namespace, returns, fields
```

---

## Design Principles

1. **Immutability**: All nodes are frozen dataclasses
2. **Type Safety**: Leverage Python 3.12+ generics for compile-time type checking
3. **Automatic Registration**: No manual registry management
4. **Uniform Pattern**: Same approach for Node and TypeDef
5. **Namespace-based Organization**: Prevent collisions and provide structure
6. **Minimal Ceremony**: Unregistered types work without registration
7. **Pluggable Serialization**: Format adapters separate concerns
8. **Reference Support**: First-class support for node references and graph structures

---

## Type Categories Summary

| Category | Registration Required? | Can be Node[T] parameter? | Can be embedded value? | Example |
|----------|------------------------|---------------------------|------------------------|---------|
| Built-in types | No | Yes | Yes | `int`, `list[str]` |
| Unregistered types | No | Yes | No | `DBConnection` |
| Registered types | Yes | Yes | Yes | `DataFrame` |

---

## Implementation Status

### Currently Implemented

- ✅ Node base class with automatic registration
- ✅ TypeDef base class with automatic registration
- ✅ Namespace support for both nodes and types
- ✅ Generic node support with type parameters
- ✅ Type registration via `TypeDef.register()`
- ✅ Schema extraction to dicts via `node_schema()` and `all_schemas()`
- ✅ Serialization via `to_dict/from_dict/to_json/from_json`
- ✅ AST container with reference resolution
- ✅ Unregistered type support (any type can be used in Node[T])

### Future Work

- ⏳ Encoding/decoding functions for registered types
- ⏳ Update schema functions to return dataclasses instead of dicts
- ⏳ Format adapter interface and implementations
- ⏳ Validation hooks for type checking
- ⏳ Custom serialization hooks
- ⏳ Node traversal utilities
- ⏳ Type inference system
- ⏳ Pretty printing and visualization
