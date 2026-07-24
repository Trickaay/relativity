"""Recursive-descent parser for the Relativity kernel DSL. Consumes the token
stream from lexer.py and produces the AST defined in ast_nodes.py.

Expression precedence (low to high), standard and unsurprising:
    or -> and -> not -> comparison -> | -> ^ -> & -> shift -> additive
    -> multiplicative -> unary(-) -> postfix (call/index/member) -> primary
"""

from .lexer import Token
from .ast_nodes import (
    Module, FieldDecl, FuncDecl, KernelDecl, Param,
    FoldDecl, Assign, AugAssign, If, ForRange, Return, ExprStmt,
    BinOp, UnaryOp, Call, Index, Member, Name, Literal,
)

_COMPARISON_OPS = {"<", "<=", ">", ">=", "==", "!="}


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # ---- token stream helpers ----

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _check(self, kind, value=None) -> bool:
        tok = self._peek()
        if tok.kind != kind:
            return False
        return value is None or tok.value == value

    def _accept(self, kind, value=None):
        if self._check(kind, value):
            return self._advance()
        return None

    def _expect(self, kind, value=None) -> Token:
        tok = self._accept(kind, value)
        if tok is None:
            got = self._peek()
            raise ParseError(
                f"line {got.line}: expected {kind}{'=' + value if value else ''}, "
                f"got {got.kind} {got.value!r}")
        return tok

    def _skip_newlines(self):
        while self._accept("NEWLINE"):
            pass

    # ---- top level ----

    def parse_module(self) -> Module:
        fields, funcs, kernels = [], [], []
        self._skip_newlines()
        while not self._check("EOF"):
            if self._check("KEYWORD", "field"):
                fields.append(self._parse_field_decl())
            elif self._check("KEYWORD", "func"):
                funcs.append(self._parse_func_or_kernel(is_func=True))
            elif self._check("KEYWORD", "kernel"):
                kernels.append(self._parse_func_or_kernel(is_func=False, stage="compute"))
            elif self._check("KEYWORD", "shader"):
                kernels.append(self._parse_func_or_kernel(is_func=False, stage="fragment"))
            else:
                tok = self._peek()
                raise ParseError(f"line {tok.line}: unexpected top-level token {tok.value!r}")
            self._skip_newlines()
        return Module(fields=fields, funcs=funcs, kernels=kernels)

    def _parse_field_decl(self) -> FieldDecl:
        self._expect("KEYWORD", "field")
        name = self._expect("IDENT").value
        self._expect("PUNCT", ":")
        elem_type = self._expect("TYPE").value
        self._expect("PUNCT", "[")
        self._expect("PUNCT", "]")
        self._expect("NEWLINE")
        return FieldDecl(name=name, elem_type=elem_type)

    def _parse_type(self) -> str:
        tok = self._accept("TYPE")
        if tok:
            return tok.value
        return self._expect("IDENT").value

    def _parse_params(self) -> list[Param]:
        params = []
        self._expect("PUNCT", "(")
        while not self._check("PUNCT", ")"):
            pname = self._expect("IDENT").value
            self._expect("PUNCT", ":")
            ptype = self._parse_type()
            params.append(Param(name=pname, type=ptype))
            if not self._accept("PUNCT", ","):
                break
        self._expect("PUNCT", ")")
        return params

    def _parse_func_or_kernel(self, is_func: bool, stage: str = "compute"):
        self._advance()  # 'func' | 'kernel' | 'shader'
        name = self._expect("IDENT").value
        params = self._parse_params()
        return_type = None
        if self._accept("OP", "->"):
            return_type = self._parse_type()
        body = self._parse_block()
        if is_func:
            return FuncDecl(name=name, params=params, return_type=return_type, body=body)
        return KernelDecl(name=name, params=params, body=body, stage=stage)

    def _parse_block(self) -> list:
        self._expect("PUNCT", ":")
        self._expect("NEWLINE")
        self._expect("INDENT")
        stmts = []
        while not self._check("DEDENT"):
            stmts.append(self._parse_statement())
        self._expect("DEDENT")
        return stmts

    # ---- statements ----

    def _parse_statement(self):
        if self._check("KEYWORD", "fold"):
            return self._parse_fold_decl()
        if self._check("KEYWORD", "if"):
            return self._parse_if()
        if self._check("KEYWORD", "for"):
            return self._parse_for()
        if self._check("KEYWORD", "return"):
            self._advance()
            value = self._parse_expr()
            self._expect("NEWLINE")
            return Return(value=value)
        return self._parse_assign_or_expr_stmt()

    def _parse_fold_decl(self) -> FoldDecl:
        self._expect("KEYWORD", "fold")
        name = self._expect("IDENT").value
        self._expect("PUNCT", ":")
        type_ = self._parse_type()
        self._expect("OP", "=")
        identity = self._parse_expr()
        self._expect("NEWLINE")
        return FoldDecl(name=name, type=type_, identity=identity)

    def _parse_if(self) -> If:
        self._expect("KEYWORD", "if")
        cond = self._parse_expr()
        then_body = self._parse_block()
        else_body = []
        if self._check("KEYWORD", "else"):
            self._advance()
            else_body = self._parse_block()
        return If(cond=cond, then_body=then_body, else_body=else_body)

    def _parse_for(self) -> ForRange:
        self._expect("KEYWORD", "for")
        var = self._expect("IDENT").value
        self._expect("KEYWORD", "in")
        self._expect("KEYWORD", "range")
        self._expect("PUNCT", "(")
        count = self._parse_expr()
        self._expect("PUNCT", ")")
        body = self._parse_block()
        return ForRange(var=var, count=count, body=body)

    def _parse_assign_or_expr_stmt(self):
        expr = self._parse_expr()
        if self._accept("OP", "="):
            value = self._parse_expr()
            self._expect("NEWLINE")
            return Assign(target=expr, value=value)
        if self._check("OP", "+=") or self._check("OP", "-="):
            op = self._advance().value
            value = self._parse_expr()
            self._expect("NEWLINE")
            return AugAssign(target=expr, op=op, value=value)
        self._expect("NEWLINE")
        return ExprStmt(value=expr)

    # ---- expressions (precedence climbing) ----

    def _parse_expr(self):
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self._check("KEYWORD", "or"):
            self._advance()
            left = BinOp(op="or", left=left, right=self._parse_and())
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._check("KEYWORD", "and"):
            self._advance()
            left = BinOp(op="and", left=left, right=self._parse_not())
        return left

    def _parse_not(self):
        if self._check("KEYWORD", "not"):
            self._advance()
            return UnaryOp(op="not", operand=self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_bitor()
        if self._check("OP") and self._peek().value in _COMPARISON_OPS:
            op = self._advance().value
            right = self._parse_bitor()
            return BinOp(op=op, left=left, right=right)
        return left

    def _parse_bitor(self):
        left = self._parse_bitxor()
        while self._check("OP", "|"):
            self._advance()
            left = BinOp(op="|", left=left, right=self._parse_bitxor())
        return left

    def _parse_bitxor(self):
        left = self._parse_bitand()
        while self._check("OP", "^"):
            self._advance()
            left = BinOp(op="^", left=left, right=self._parse_bitand())
        return left

    def _parse_bitand(self):
        left = self._parse_shift()
        while self._check("OP", "&"):
            self._advance()
            left = BinOp(op="&", left=left, right=self._parse_shift())
        return left

    def _parse_shift(self):
        left = self._parse_additive()
        while self._check("OP", "<<") or self._check("OP", ">>"):
            op = self._advance().value
            left = BinOp(op=op, left=left, right=self._parse_additive())
        return left

    def _parse_additive(self):
        left = self._parse_multiplicative()
        while self._check("OP", "+") or self._check("OP", "-"):
            op = self._advance().value
            left = BinOp(op=op, left=left, right=self._parse_multiplicative())
        return left

    def _parse_multiplicative(self):
        left = self._parse_unary()
        while self._check("OP", "*") or self._check("OP", "/") or self._check("OP", "%"):
            op = self._advance().value
            left = BinOp(op=op, left=left, right=self._parse_unary())
        return left

    def _parse_unary(self):
        if self._check("OP", "-"):
            self._advance()
            return UnaryOp(op="-", operand=self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self):
        expr = self._parse_primary()
        while True:
            if self._check("PUNCT", "["):
                self._advance()
                index = self._parse_expr()
                self._expect("PUNCT", "]")
                expr = Index(base=expr, index=index)
            elif self._check("PUNCT", "."):
                self._advance()
                attr = self._expect("IDENT").value
                expr = Member(base=expr, attr=attr)
            else:
                break
        return expr

    def _parse_primary(self):
        tok = self._peek()
        if tok.kind == "INT":
            self._advance()
            return Literal(value=int(tok.value))
        if tok.kind == "HEXNUMBER":
            self._advance()
            return Literal(value=int(tok.value, 16), force_uint=True)
        if tok.kind == "UINT":
            self._advance()
            return Literal(value=int(tok.value[:-1]), force_uint=True)
        if tok.kind == "FLOAT":
            self._advance()
            return Literal(value=float(tok.value))
        if tok.kind in ("IDENT", "TYPE"):
            self._advance()
            if self._check("PUNCT", "("):
                self._advance()
                args = []
                while not self._check("PUNCT", ")"):
                    args.append(self._parse_expr())
                    if not self._accept("PUNCT", ","):
                        break
                self._expect("PUNCT", ")")
                return Call(func=tok.value, args=args)
            return Name(id=tok.value)
        if tok.kind == "PUNCT" and tok.value == "(":
            self._advance()
            expr = self._parse_expr()
            self._expect("PUNCT", ")")
            return expr
        raise ParseError(f"line {tok.line}: unexpected token in expression: {tok.kind} {tok.value!r}")


def parse(source_text: str) -> Module:
    from .lexer import tokenize
    tokens = tokenize(source_text)
    return Parser(tokens).parse_module()
