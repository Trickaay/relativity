"""AST node definitions for the Relativity kernel DSL. Deliberately small -- see
compiler.py's module docstring for what this language is for and why it
exists; the grammar covers exactly what the BVH pipeline kernels need,
not general-purpose shading.
"""

from dataclasses import dataclass, field as dc_field


# ---- top-level declarations ----

@dataclass
class Param:
    name: str
    type: str  # "f32" | "i32" | "u32" | "vec3" | "vec4"


@dataclass
class FieldDecl:
    name: str
    elem_type: str  # element type of the array, e.g. "vec4", "u32"


@dataclass
class FuncDecl:
    name: str
    params: list[Param]
    return_type: str
    body: list  # list of statements


@dataclass
class KernelDecl:
    name: str
    params: list[Param]
    body: list
    stage: str = "compute"  # "compute" | "fragment"


@dataclass
class Module:
    fields: list[FieldDecl]
    funcs: list[FuncDecl]
    kernels: list[KernelDecl]


# ---- statements ----

@dataclass
class FoldDecl:
    """`fold name: type = <identity expr>` -- declares a parallel-reduction
    accumulator. The combine op (min/max/+) is inferred from how `name` is
    reassigned inside the enclosing for-loop (see codegen_glsl.py)."""
    name: str
    type: str
    identity: object  # an expression node


@dataclass
class Assign:
    target: object  # Name | Index | Member
    value: object


@dataclass
class AugAssign:
    target: object
    op: str  # "+=" | "-="
    value: object


@dataclass
class If:
    cond: object
    then_body: list
    else_body: list


@dataclass
class ForRange:
    var: str
    count: object  # expression for the upper bound (exclusive), lower bound is always 0
    body: list


@dataclass
class Return:
    value: object


@dataclass
class ExprStmt:
    value: object


# ---- expressions ----

@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class UnaryOp:
    op: str
    operand: object


@dataclass
class Call:
    func: str
    args: list


@dataclass
class Index:
    base: object
    index: object


@dataclass
class Member:
    base: object
    attr: str


@dataclass
class Name:
    id: str


@dataclass
class Literal:
    value: object  # int | float
    force_uint: bool = False  # True for 0x... or NNNu literals -- emitted with a GLSL 'u' suffix
