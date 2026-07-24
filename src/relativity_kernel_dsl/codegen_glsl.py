"""GLSL codegen for the Relativity kernel DSL. Walks the AST (ast_nodes.py) and
emits real, standalone GLSL compute/fragment shader source text -- see
compiler.py's module docstring for the full picture of why this targets
GLSL (via the already-proven glslang -> spirv-cross -> SDL3 GPU/MSL
pipeline) rather than an actual Taichi runtime.

Key ergonomic translation, the whole point of this module: a plain
`for i in range(n):` at the top of a `kernel` body means "run once per
element, in parallel" -- same reading as a Taichi `for i in range(n):`
inside an `@ti.kernel` -- and this file is what turns that into real
`gl_GlobalInvocationID`-based dispatch + a bounds check, so the DSL
author never writes that boilerplate by hand. Similarly, `field`
references never carry a manual `binding=N` number in the source -- this
module assigns them (first-reference order) and returns the assignment
as part of the compiled manifest, so the host driver doesn't have to
hand-track them either.

GLSL has no implicit variable declaration (unlike Python, assigning to a
fresh name doesn't create it) -- so a small type inference pass runs
alongside statement emission, tracking each name's GLSL type through a
simple scope stack, purely to decide whether an `Assign` is the first
appearance of a local (needs `<type> name = ...;`) or a later
reassignment (`name = ...;`).
"""

from dataclasses import dataclass

from . import ast_nodes as A

LOCAL_SIZE = 64

_CASTS = {"f32": "float", "i32": "int", "u32": "uint", "vec3": "vec3", "vec4": "vec4"}
_GLSL_TYPE = {"f32": "float", "i32": "int", "u32": "uint", "vec3": "vec3", "vec4": "vec4"}
_COMPARISON_OR_BOOL_OPS = {"<", "<=", ">", ">=", "==", "!=", "and", "or"}
_VEC3_COMPONENTS = ("x", "y", "z")
_SWIZZLE_LEN_TO_TYPE = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}


class CodegenError(Exception):
    pass


@dataclass
class BindingInfo:
    name: str
    kind: str          # "field" | "fold"
    binding: int        # the GLSL-source binding number -- NOT necessarily the
                         # final MSL buffer() slot; spirv-cross may reassign
                         # (confirmed empirically, see sdl3_gpu_shaders memory
                         # and this module's own smoke test) -- compiler.py
                         # re-derives the true slot from the compiled .msl by
                         # looking up `struct_name` below, and that corrected
                         # value is what the host driver must actually use.
    mode: str           # "readonly" | "readwrite"
    elem_type: str       # "vec4" | "u32" | "vec3" | "f32" | ...
    struct_name: str = ""         # exact GLSL buffer-block name emitted, e.g. "Positions"/"BoundsMinBits"
    combine_op: str = None       # fold only: "min" | "max" | "+"
    identity: object = None      # fold only: float, or tuple of 3 floats for vec3


@dataclass
class CompiledKernel:
    name: str
    stage: str  # "compute" | "fragment"
    glsl_source: str
    bindings: list           # list[BindingInfo], in binding-slot order
    uniform_params: list     # list[A.Param], in UBO order
    dispatch_param: str = None  # name of the scalar param driving ceil(n/LOCAL_SIZE) dispatch sizing
    local_size: int = LOCAL_SIZE


# ---------------------------------------------------------------------
# Fold combine-op detection
# ---------------------------------------------------------------------

def _expr_is_name(node, name):
    return isinstance(node, A.Name) and node.id == name


def _find_fold_update(body, fold_name):
    """Scans (recursively) for the statement that updates `fold_name`
    inside a loop, returning (combine_op, per_element_expr) -- e.g. for
    `bounds_min = min(bounds_min, expr)` returns ("min", expr)."""
    for stmt in body:
        if isinstance(stmt, A.Assign) and _expr_is_name(stmt.target, fold_name):
            value = stmt.value
            if isinstance(value, A.Call) and value.func in ("min", "max") and len(value.args) == 2:
                a, b = value.args
                if _expr_is_name(a, fold_name):
                    return value.func, b
                if _expr_is_name(b, fold_name):
                    return value.func, a
            if isinstance(value, A.BinOp) and value.op == "+":
                if _expr_is_name(value.left, fold_name):
                    return "+", value.right
                if _expr_is_name(value.right, fold_name):
                    return "+", value.left
            raise CodegenError(
                f"fold '{fold_name}': reassignment must be min(...)/max(...)/+ combine, got {value!r}")
        if isinstance(stmt, A.AugAssign) and _expr_is_name(stmt.target, fold_name) and stmt.op == "+=":
            return "+", stmt.value
        for sub_body in (getattr(stmt, "then_body", None), getattr(stmt, "else_body", None),
                         getattr(stmt, "body", None)):
            if sub_body:
                found = _find_fold_update(sub_body, fold_name)
                if found:
                    return found
    return None


def _literal_scalar(node) -> float:
    if isinstance(node, A.Literal):
        return float(node.value)
    if isinstance(node, A.UnaryOp) and node.op == "-":
        return -_literal_scalar(node.operand)
    raise CodegenError(f"fold identity must be a literal constant, got {node!r}")


def _fold_identity_value(fold_decl: A.FoldDecl):
    node = fold_decl.identity
    if fold_decl.type == "vec3":
        if not (isinstance(node, A.Call) and node.func == "vec3" and len(node.args) == 3):
            raise CodegenError("vec3 fold identity must be a literal vec3(a, b, c)")
        return tuple(_literal_scalar(a) for a in node.args)
    return _literal_scalar(node)


# ---------------------------------------------------------------------
# Reference collection (fields + folds referenced by a kernel body,
# in first-reference order, plus whether each is ever written).
# ---------------------------------------------------------------------

class _RefCollector:
    """Walks a kernel body AND, transitively, the bodies of any `func`s it
    calls -- a field only ever touched inside a helper function (e.g.
    Karras' `delta()` reading the `code` field) still needs a binding slot
    allocated for the kernel that calls it, or codegen would emit a
    reference to a buffer that was never declared."""

    def __init__(self, field_names, fold_names, func_by_name=None):
        self.field_names = field_names
        self.fold_names = fold_names
        self.func_by_name = func_by_name or {}
        self.order = []       # first-reference order, field/fold names only
        self.written = set()
        self._visited_funcs = set()

    def _note(self, name, is_write):
        if name in self.field_names or name in self.fold_names:
            if name not in self.order:
                self.order.append(name)
            if is_write:
                self.written.add(name)

    def visit_body(self, body):
        for stmt in body:
            self.visit_stmt(stmt)

    def visit_stmt(self, stmt):
        if isinstance(stmt, A.FoldDecl):
            self._note(stmt.name, is_write=True)
        elif isinstance(stmt, A.Assign):
            self._visit_target(stmt.target)
            self.visit_expr(stmt.value)
        elif isinstance(stmt, A.AugAssign):
            self._visit_target(stmt.target)
            self.visit_expr(stmt.value)
        elif isinstance(stmt, A.If):
            self.visit_expr(stmt.cond)
            self.visit_body(stmt.then_body)
            self.visit_body(stmt.else_body)
        elif isinstance(stmt, A.ForRange):
            self.visit_expr(stmt.count)
            self.visit_body(stmt.body)
        elif isinstance(stmt, A.Return):
            self.visit_expr(stmt.value)
        elif isinstance(stmt, A.ExprStmt):
            self.visit_expr(stmt.value)

    def _visit_target(self, target):
        base = target
        while isinstance(base, (A.Index, A.Member)):
            base = base.base
        if isinstance(base, A.Name):
            self._note(base.id, is_write=True)
        self.visit_expr(target)

    def visit_expr(self, node):
        if isinstance(node, A.Name):
            self._note(node.id, is_write=False)
        elif isinstance(node, A.BinOp):
            self.visit_expr(node.left)
            self.visit_expr(node.right)
        elif isinstance(node, A.UnaryOp):
            self.visit_expr(node.operand)
        elif isinstance(node, A.Call):
            if node.func == "atomic_add" and node.args:
                # atomic_add's first argument is a read-modify-write target
                # (maps to GLSL atomicAdd), not a plain read -- the buffer it
                # indexes into must get the read-write binding, or spirv-cross
                # marks it `readonly` and the atomic write becomes an l-value
                # error (confirmed empirically compiling radix_histogram).
                self._visit_target(node.args[0])
                for a in node.args[1:]:
                    self.visit_expr(a)
            else:
                for a in node.args:
                    self.visit_expr(a)
            if node.func in self.func_by_name and node.func not in self._visited_funcs:
                self._visited_funcs.add(node.func)
                self.visit_body(self.func_by_name[node.func].body)
        elif isinstance(node, A.Index):
            self.visit_expr(node.base)
            self.visit_expr(node.index)
        elif isinstance(node, A.Member):
            self.visit_expr(node.base)


# ---------------------------------------------------------------------
# Scoped type environment (for deciding first-declaration vs reassignment)
# ---------------------------------------------------------------------

class _Scope:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def lookup(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise KeyError(name)

    def contains(self, name):
        try:
            self.lookup(name)
            return True
        except KeyError:
            return False

    def declare(self, name, glsl_type):
        self.vars[name] = glsl_type

    def child(self):
        return _Scope(parent=self)


_FLOAT_FLIP_HELPERS = """\
uint _relativity_float_flip(float f) {
    uint bits = floatBitsToUint(f);
    uint mask = -int(bits >> 31) | 0x80000000u;
    return bits ^ mask;
}
float _relativity_float_unflip(uint bits) {
    uint mask = ((bits >> 31) - 1u) | 0x80000000u;
    return uintBitsToFloat(bits ^ mask);
}
"""


def _collect_locals(body, out_set):
    for stmt in body:
        if isinstance(stmt, (A.Assign, A.AugAssign)) and isinstance(stmt.target, A.Name):
            out_set.add(stmt.target.id)
        elif isinstance(stmt, A.ForRange):
            out_set.add(stmt.var)
            _collect_locals(stmt.body, out_set)
        elif isinstance(stmt, A.If):
            _collect_locals(stmt.then_body, out_set)
            _collect_locals(stmt.else_body, out_set)


def _rename_ast(node, local_names, suffix):
    """Renames every reference to a name in `local_names` by appending
    `suffix` -- used to inline a function's body at a call site without
    its locals colliding with either the caller's locals or another
    inlined copy of the same function at a different call site."""
    if isinstance(node, A.Name):
        return A.Name(id=node.id + suffix) if node.id in local_names else node
    if isinstance(node, A.Literal):
        return node
    if isinstance(node, A.BinOp):
        return A.BinOp(op=node.op, left=_rename_ast(node.left, local_names, suffix),
                        right=_rename_ast(node.right, local_names, suffix))
    if isinstance(node, A.UnaryOp):
        return A.UnaryOp(op=node.op, operand=_rename_ast(node.operand, local_names, suffix))
    if isinstance(node, A.Call):
        return A.Call(func=node.func, args=[_rename_ast(a, local_names, suffix) for a in node.args])
    if isinstance(node, A.Index):
        return A.Index(base=_rename_ast(node.base, local_names, suffix),
                        index=_rename_ast(node.index, local_names, suffix))
    if isinstance(node, A.Member):
        return A.Member(base=_rename_ast(node.base, local_names, suffix), attr=node.attr)
    if isinstance(node, A.Assign):
        return A.Assign(target=_rename_ast(node.target, local_names, suffix),
                         value=_rename_ast(node.value, local_names, suffix))
    if isinstance(node, A.AugAssign):
        return A.AugAssign(target=_rename_ast(node.target, local_names, suffix), op=node.op,
                            value=_rename_ast(node.value, local_names, suffix))
    if isinstance(node, A.If):
        return A.If(cond=_rename_ast(node.cond, local_names, suffix),
                    then_body=[_rename_ast(s, local_names, suffix) for s in node.then_body],
                    else_body=[_rename_ast(s, local_names, suffix) for s in node.else_body])
    if isinstance(node, A.ForRange):
        new_var = node.var + suffix if node.var in local_names else node.var
        return A.ForRange(var=new_var, count=_rename_ast(node.count, local_names, suffix),
                           body=[_rename_ast(s, local_names, suffix) for s in node.body])
    if isinstance(node, A.Return):
        return A.Return(value=_rename_ast(node.value, local_names, suffix))
    if isinstance(node, A.ExprStmt):
        return A.ExprStmt(value=_rename_ast(node.value, local_names, suffix))
    raise CodegenError(f"cannot rename node {node!r}")


class _Emitter:
    """Emits GLSL statements/expressions for one function or kernel body,
    tracking each local variable's inferred GLSL type in a scope stack so
    `Assign` can decide between a fresh `<type> name = ...;` declaration
    and a plain `name = ...;` reassignment (GLSL has no implicit
    declare-by-assignment, unlike the DSL's Python-like surface syntax).

    `inline_funcs`: functions that must be textually inlined at every call
    site rather than emitted as a real GLSL function -- specifically, any
    function that (transitively) reads a field. Confirmed empirically
    (see bvh_pipeline.rfrk / this module's own dev notes): spirv-cross
    compiles such a function into MSL by threading the buffer through as
    an explicit `device Foo&` reference parameter, and dispatching the
    resulting shader via SDL3 GPU then produces silently wrong results
    regardless of storage-buffer binding order -- inlining sidesteps that
    code path entirely by never emitting a real function with a
    buffer-reference parameter in the first place."""

    def __init__(self, field_elem_types, fold_ops, func_return_types, scalar_param_types, inline_funcs=None):
        self.field_elem_types = field_elem_types
        self.fold_ops = fold_ops
        self.func_return_types = func_return_types
        self.scalar_param_types = scalar_param_types
        self.inline_funcs = inline_funcs or {}
        self._inline_counter = 0
        self.lines: list[str] = []

    # ---- type inference ----

    def _infer(self, node, scope):
        if isinstance(node, A.Literal):
            if isinstance(node.value, float):
                return "float"
            return "uint" if node.force_uint else "int"
        if isinstance(node, A.Name):
            if scope.contains(node.id):
                return scope.lookup(node.id)
            if node.id in self.scalar_param_types:
                return self.scalar_param_types[node.id]
            raise CodegenError(f"unknown identifier '{node.id}'")
        if isinstance(node, A.UnaryOp):
            return "bool" if node.op == "not" else self._infer(node.operand, scope)
        if isinstance(node, A.BinOp):
            if node.op in _COMPARISON_OR_BOOL_OPS:
                return "bool"
            lt = self._infer(node.left, scope)
            rt = self._infer(node.right, scope)
            for t in (lt, rt):
                if t in ("vec3", "vec4"):
                    return t
            return "float" if "float" in (lt, rt) else lt
        if isinstance(node, A.Call):
            if node.func in _CASTS:
                return _CASTS[node.func]
            if node.func in ("min", "max", "clamp"):
                return self._infer(node.args[0], scope)
            if node.func in ("sqrt", "dot", "length"):
                return "float"
            if node.func in ("normalize", "cross"):
                return "vec3"
            if node.func == "atomic_add":
                return self._infer(node.args[0], scope)
            if node.func in self.func_return_types:
                return self.func_return_types[node.func]
            raise CodegenError(f"cannot infer return type of call to '{node.func}'")
        if isinstance(node, A.Index):
            base = node.base
            if isinstance(base, A.Name) and base.id in self.field_elem_types:
                return self.field_elem_types[base.id]
            raise CodegenError(f"cannot infer element type for index expression {node!r}")
        if isinstance(node, A.Member):
            base_t = self._infer(node.base, scope)
            return _SWIZZLE_LEN_TO_TYPE[len(node.attr)]
        raise CodegenError(f"cannot infer type of {node!r}")

    # ---- expression emission ----

    def emit_expr(self, node) -> str:
        if isinstance(node, A.Literal):
            if isinstance(node.value, float):
                text = repr(node.value)
                return text if ("." in text or "e" in text) else text + ".0"
            return f"{node.value}u" if node.force_uint else str(node.value)
        if isinstance(node, A.Name):
            if node.id in self.scalar_param_types and node.id not in self.fold_ops:
                return f"ubo.{node.id}"
            return node.id
        if isinstance(node, A.BinOp):
            op = "&&" if node.op == "and" else "||" if node.op == "or" else node.op
            return f"({self.emit_expr(node.left)} {op} {self.emit_expr(node.right)})"
        if isinstance(node, A.UnaryOp):
            op = "!" if node.op == "not" else node.op
            return f"({op}{self.emit_expr(node.operand)})"
        if isinstance(node, A.Index):
            return f"{self.emit_expr(node.base)}[{self.emit_expr(node.index)}]"
        if isinstance(node, A.Member):
            return f"{self.emit_expr(node.base)}.{node.attr}"
        if isinstance(node, A.Call):
            fname = _CASTS.get(node.func, "atomicAdd" if node.func == "atomic_add" else node.func)
            return f"{fname}({', '.join(self.emit_expr(a) for a in node.args)})"
        raise CodegenError(f"cannot emit expression node {node!r}")

    def _emit_expr_inline(self, node, scope, indent) -> str:
        """Like emit_expr, but expands any call to an `inline_funcs` member
        at its call site (emitting the necessary statements into
        self.lines at `indent` first) instead of emitting a real function
        call -- see the class docstring for why this exists."""
        if isinstance(node, A.Call) and node.func in self.inline_funcs:
            return self._inline_call(node, scope, indent)
        if isinstance(node, A.BinOp):
            left = self._emit_expr_inline(node.left, scope, indent)
            right = self._emit_expr_inline(node.right, scope, indent)
            op = "&&" if node.op == "and" else "||" if node.op == "or" else node.op
            return f"({left} {op} {right})"
        if isinstance(node, A.UnaryOp):
            operand = self._emit_expr_inline(node.operand, scope, indent)
            op = "!" if node.op == "not" else node.op
            return f"({op}{operand})"
        if isinstance(node, A.Index):
            base = self._emit_expr_inline(node.base, scope, indent)
            index = self._emit_expr_inline(node.index, scope, indent)
            return f"{base}[{index}]"
        if isinstance(node, A.Member):
            base = self._emit_expr_inline(node.base, scope, indent)
            return f"{base}.{node.attr}"
        if isinstance(node, A.Call):
            args = [self._emit_expr_inline(a, scope, indent) for a in node.args]
            fname = _CASTS.get(node.func, "atomicAdd" if node.func == "atomic_add" else node.func)
            return f"{fname}({', '.join(args)})"
        return self.emit_expr(node)  # Literal / Name -- no sub-expressions to recurse into

    def _inline_call(self, node, scope, indent) -> str:
        self._inline_counter += 1
        suffix = f"_inl{self._inline_counter}"
        func = self.inline_funcs[node.func]

        local_names = {p.name for p in func.params}
        _collect_locals(func.body, local_names)

        for p, arg in zip(func.params, node.args):
            arg_text = self._emit_expr_inline(arg, scope, indent)
            pname = p.name + suffix
            self.lines.append(f"{indent}{_GLSL_TYPE[p.type]} {pname} = {arg_text};")
            scope.declare(pname, _GLSL_TYPE[p.type])

        # A `_relativity_`-prefixed name, distinct from anything a renamed local
        # could produce (e.g. delta()'s own `result` local becomes
        # `result_inl1` -- naming this temp `result_inl1` too would create
        # a self-referential `int result_inl1 = result_inl1;` bug).
        result_var = f"_relativity_inline_result{suffix}"
        ret_type = _GLSL_TYPE.get(func.return_type, func.return_type)
        for stmt in (_rename_ast(s, local_names, suffix) for s in func.body):
            if isinstance(stmt, A.Return):
                self.lines.append(f"{indent}{ret_type} {result_var} = {self.emit_expr(stmt.value)};")
            else:
                self.emit_stmt(stmt, scope, indent)
        return result_var

    # ---- statement emission ----

    def emit_body(self, body, scope, indent):
        for stmt in body:
            self.emit_stmt(stmt, scope, indent)

    def emit_stmt(self, stmt, scope, indent):
        if isinstance(stmt, A.FoldDecl):
            return  # host-initialized before dispatch; nothing to emit in-shader

        if isinstance(stmt, A.Assign) and isinstance(stmt.target, A.Name) and stmt.target.id in self.fold_ops:
            self._emit_fold_update(stmt.target.id, indent)
            return
        if isinstance(stmt, A.AugAssign) and isinstance(stmt.target, A.Name) and stmt.target.id in self.fold_ops:
            self._emit_fold_update(stmt.target.id, indent)
            return

        if isinstance(stmt, A.Assign):
            value_str = self._emit_expr_inline(stmt.value, scope, indent)
            if isinstance(stmt.target, A.Name) and not scope.contains(stmt.target.id) \
                    and stmt.target.id not in self.scalar_param_types:
                glsl_type = self._infer(stmt.value, scope)
                scope.declare(stmt.target.id, glsl_type)
                self.lines.append(f"{indent}{glsl_type} {stmt.target.id} = {value_str};")
            else:
                self.lines.append(f"{indent}{self._emit_expr_inline(stmt.target, scope, indent)} = {value_str};")
        elif isinstance(stmt, A.AugAssign):
            target_str = self._emit_expr_inline(stmt.target, scope, indent)
            value_str = self._emit_expr_inline(stmt.value, scope, indent)
            self.lines.append(f"{indent}{target_str} {stmt.op} {value_str};")
        elif isinstance(stmt, A.If):
            cond_str = self._emit_expr_inline(stmt.cond, scope, indent)
            self.lines.append(f"{indent}if ({cond_str}) {{")
            self.emit_body(stmt.then_body, scope.child(), indent + "    ")
            if stmt.else_body:
                self.lines.append(f"{indent}}} else {{")
                self.emit_body(stmt.else_body, scope.child(), indent + "    ")
            self.lines.append(f"{indent}}}")
        elif isinstance(stmt, A.ForRange):
            v = stmt.var
            child = scope.child()
            child.declare(v, "int")
            count_str = self._emit_expr_inline(stmt.count, scope, indent)
            self.lines.append(f"{indent}for (int {v} = 0; {v} < int({count_str}); {v}++) {{")
            self.emit_body(stmt.body, child, indent + "    ")
            self.lines.append(f"{indent}}}")
        elif isinstance(stmt, A.Return):
            self.lines.append(f"{indent}return {self._emit_expr_inline(stmt.value, scope, indent)};")
        elif isinstance(stmt, A.ExprStmt):
            self.lines.append(f"{indent}{self._emit_expr_inline(stmt.value, scope, indent)};")
        else:
            raise CodegenError(f"cannot emit statement {stmt!r}")

    def _emit_fold_update(self, fold_name, indent):
        op, per_elem_expr, comp_type, n_comp = self.fold_ops[fold_name]
        val_expr = self.emit_expr(per_elem_expr)
        bits_name = f"{fold_name}_bits"
        if n_comp == 1:
            components = [(val_expr, 0)]
        else:
            self.lines.append(f"{indent}{comp_type} _relativity_fold_val_{fold_name} = {val_expr};")
            components = [(f"_relativity_fold_val_{fold_name}.{c}", i) for i, c in enumerate(_VEC3_COMPONENTS[:n_comp])]

        for comp_expr, slot in components:
            if op in ("min", "max"):
                glsl_fn = "atomicMin" if op == "min" else "atomicMax"
                self.lines.append(f"{indent}{glsl_fn}({bits_name}[{slot}], _relativity_float_flip({comp_expr}));")
            elif op == "+":
                self.lines.append(f"{indent}{{")
                self.lines.append(f"{indent}    uint _relativity_assumed, _relativity_old;")
                self.lines.append(f"{indent}    float _relativity_val = {comp_expr};")
                self.lines.append(f"{indent}    do {{")
                self.lines.append(f"{indent}        _relativity_assumed = {bits_name}[{slot}];")
                self.lines.append(
                    f"{indent}        _relativity_old = atomicCompSwap({bits_name}[{slot}], _relativity_assumed, "
                    f"floatBitsToUint(uintBitsToFloat(_relativity_assumed) + _relativity_val));")
                self.lines.append(f"{indent}    }} while (_relativity_assumed != _relativity_old);")
                self.lines.append(f"{indent}}}")
            else:
                raise CodegenError(f"unsupported fold combine op {op!r}")


def _emit_func(fn: A.FuncDecl, field_elem_types, func_return_types) -> str:
    ret = _GLSL_TYPE.get(fn.return_type, fn.return_type) if fn.return_type else "void"
    params = ", ".join(f"{_GLSL_TYPE[p.type]} {p.name}" for p in fn.params)
    scope = _Scope()
    for p in fn.params:
        scope.declare(p.name, _GLSL_TYPE[p.type])
    emitter = _Emitter(field_elem_types=field_elem_types, fold_ops={},
                        func_return_types=func_return_types, scalar_param_types={})
    emitter.emit_body(fn.body, scope, indent="    ")
    return f"{ret} {fn.name}({params}) {{\n" + "\n".join(emitter.lines) + "\n}"


def _funcs_needing_inline(module: A.Module, field_names, func_by_name) -> set:
    """A function needs inlining (see _Emitter's docstring for why) if its
    own body reads a field, directly or transitively through further
    calls -- reuses _RefCollector itself, rooted at the func's body
    instead of a kernel's."""
    needing_inline = set()
    for fn in module.funcs:
        collector = _RefCollector(field_names, set(), func_by_name)
        collector.visit_body(fn.body)
        if collector.order:
            needing_inline.add(fn.name)
    return needing_inline


# ---------------------------------------------------------------------
# Top-level kernel/shader compilation
# ---------------------------------------------------------------------

def compile_kernel(kernel: A.KernelDecl, module: A.Module) -> CompiledKernel:
    field_by_name = {f.name: f for f in module.fields}
    field_elem_types = {f.name: _GLSL_TYPE[f.elem_type] for f in module.fields}
    func_return_types = {fn.name: _GLSL_TYPE.get(fn.return_type, fn.return_type)
                          for fn in module.funcs if fn.return_type}

    fold_decls = [s for s in kernel.body if isinstance(s, A.FoldDecl)]
    fold_by_name = {f.name: f for f in fold_decls}
    scalar_param_types = {p.name: _GLSL_TYPE[p.type] for p in kernel.params}

    func_by_name = {fn.name: fn for fn in module.funcs}
    inline_func_names = _funcs_needing_inline(module, set(field_by_name), func_by_name)
    inline_funcs = {name: fn for name, fn in func_by_name.items() if name in inline_func_names}
    collector = _RefCollector(set(field_by_name), set(fold_by_name), func_by_name)
    collector.visit_body(kernel.body)

    bindings: list[BindingInfo] = []
    fold_ops = {}
    for name in collector.order:
        if name in field_by_name:
            fdecl = field_by_name[name]
            mode = "readwrite" if name in collector.written else "readonly"
            bindings.append(BindingInfo(name=name, kind="field", binding=len(bindings),
                                         mode=mode, elem_type=fdecl.elem_type))
        else:
            fdecl = fold_by_name[name]
            combine_op, per_elem_expr = _find_fold_update(kernel.body, name)
            n_comp = 3 if fdecl.type == "vec3" else 1
            fold_ops[name] = (combine_op, per_elem_expr, fdecl.type, n_comp)
            bindings.append(BindingInfo(name=name, kind="fold", binding=len(bindings), mode="readwrite",
                                         elem_type=fdecl.type, combine_op=combine_op,
                                         identity=_fold_identity_value(fdecl)))

    uses_float_flip = any(op in ("min", "max") for op, *_ in fold_ops.values())

    lines = ["#version 450", ""]
    if kernel.stage == "compute":
        lines.append(f"layout(local_size_x = {LOCAL_SIZE}) in;")
    lines.append("")

    for b in bindings:
        if b.kind == "field":
            glsl_t = _GLSL_TYPE[b.elem_type]
            b.struct_name = b.name[0].upper() + b.name[1:]
            # GLSL storage buffers default to fully read-write regardless of
            # actual usage (unlike storage images, whose readonly/readwrite
            # split spirv-cross infers from imageLoad/imageStore calls) -- an
            # explicit `readonly` qualifier is required for spirv-cross to
            # emit `const device` in the compiled MSL, which is what lets
            # compiler.py tell readonly and readwrite buffers apart there.
            qualifier = "readonly " if b.mode == "readonly" else ""
            lines.append(
                f"layout(std430, binding = {b.binding}) {qualifier}buffer {b.struct_name} {{ {glsl_t} {b.name}[]; }};")
        else:
            n_comp = 3 if b.elem_type == "vec3" else 1
            b.struct_name = "".join(part.capitalize() for part in b.name.split("_")) + "Bits"
            lines.append(f"layout(std430, binding = {b.binding}) buffer {b.struct_name} "
                          f"{{ uint {b.name}_bits[{n_comp}]; }};")
    lines.append("")

    if scalar_param_types:
        ubo_binding = len(bindings)
        lines.append(f"layout(binding = {ubo_binding}) uniform UBO {{")
        for p in kernel.params:
            lines.append(f"    {_GLSL_TYPE[p.type]} {p.name};")
        lines.append("} ubo;")
        lines.append("")

    if uses_float_flip:
        lines.append(_FLOAT_FLIP_HELPERS)

    # Only emit funcs actually reachable (directly or transitively) from
    # this kernel -- emitting every func in the module unconditionally
    # would leak field references from unrelated helpers (e.g. delta()'s
    # use of the `morton` field) into shaders that never call them and
    # never got that field's buffer declared. Funcs needing inlining are
    # never emitted as real functions at all (see _Emitter's docstring).
    for fn in module.funcs:
        if fn.name in collector._visited_funcs and fn.name not in inline_func_names:
            lines.append(_emit_func(fn, field_elem_types, func_return_types))
            lines.append("")

    body_stmts = [s for s in kernel.body if not isinstance(s, A.FoldDecl)]
    dispatch_param = None
    root_scope = _Scope()
    emitter = _Emitter(field_elem_types=field_elem_types, fold_ops=fold_ops,
                        func_return_types=func_return_types, scalar_param_types=scalar_param_types,
                        inline_funcs=inline_funcs)

    if kernel.stage == "compute":
        outer = body_stmts[0] if body_stmts and isinstance(body_stmts[0], A.ForRange) else None
        if outer is None:
            raise CodegenError(f"kernel '{kernel.name}': top-level body must start with a `for i in range(n):`")
        dispatch_param = outer.count.id if isinstance(outer.count, A.Name) else None
        lines.append("void main() {")
        root_scope.declare(outer.var, "uint")
        lines.append(f"    uint {outer.var} = gl_GlobalInvocationID.x;")
        bound_expr = emitter.emit_expr(outer.count)
        lines.append(f"    if ({outer.var} >= uint({bound_expr})) return;")
        emitter.emit_body(outer.body, root_scope.child(), indent="    ")
        for stmt in body_stmts[1:]:
            emitter.emit_stmt(stmt, root_scope, indent="    ")
        lines.extend(emitter.lines)
        lines.append("}")
    else:
        lines.append("layout(location = 0) out vec4 FragColor;")
        lines.append("")
        lines.append("void main() {")
        root_scope.declare("frag_coord", "vec2")
        root_scope.declare("out_color", "vec3")
        lines.append("    vec2 frag_coord = gl_FragCoord.xy;")
        lines.append("    vec3 out_color = vec3(0.0);")
        emitter.emit_body(body_stmts, root_scope, indent="    ")
        lines.extend(emitter.lines)
        lines.append("    FragColor = vec4(out_color, 1.0);")
        lines.append("}")

    glsl_source = "\n".join(lines)
    return CompiledKernel(
        name=kernel.name, stage=kernel.stage, glsl_source=glsl_source,
        bindings=bindings, uniform_params=list(kernel.params),
        dispatch_param=dispatch_param, local_size=LOCAL_SIZE,
    )
