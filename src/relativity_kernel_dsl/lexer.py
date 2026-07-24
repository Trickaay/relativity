"""
Lexer for the Relativity kernel DSL (.rfrk files) -- see the module docstring in
compiler.py for the full picture. Turns source text into a flat token
stream; the parser consumes that stream so it never has to deal with
whitespace, comments, or multi-character operators itself.

Indentation is significant (Python-like: a block is whatever's indented
under a `kernel`/`shader`/`func`/`if`/`for` header), so this lexer also
emits INDENT/DEDENT tokens, tracked via an indent-level stack -- the
same well-established technique CPython's own tokenizer uses.
"""

import re
from dataclasses import dataclass

KEYWORDS = {
    "field", "kernel", "shader", "func", "for", "in", "range", "if", "else",
    "return", "and", "or", "not", "fold",
}
TYPES = {"f32", "i32", "u32", "vec3", "vec4"}

# Longest-match-first so e.g. "<<" isn't tokenized as two "<" tokens.
_OPERATORS = [
    "<<", ">>", "<=", ">=", "==", "!=", "+=", "-=", "->",
    "+", "-", "*", "/", "%", "&", "|", "^", "<", ">", "=",
]
_PUNCTUATION = ["(", ")", "[", "]", ":", ",", "."]

_TOKEN_RE = re.compile(r"""
    (?P<COMMENT>\#[^\n]*)
  | (?P<NEWLINE>\n)
  | (?P<WHITESPACE>[ \t]+)
  | (?P<HEXNUMBER>0[xX][0-9a-fA-F]+)
  | (?P<FLOAT>\d+\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+)
  | (?P<UINT>\d+[uU])
  | (?P<INT>\d+)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<OP>""" + "|".join(re.escape(o) for o in _OPERATORS) + r""")
  | (?P<PUNCT>""" + "|".join(re.escape(p) for p in _PUNCTUATION) + r""")
""", re.VERBOSE)


@dataclass
class Token:
    kind: str   # KEYWORD, TYPE, IDENT, INT, FLOAT, HEXNUMBER, OP, PUNCT,
                # NEWLINE, INDENT, DEDENT, EOF
    value: str
    line: int


class LexError(Exception):
    pass


def tokenize(source: str) -> list[Token]:
    """Indentation is only meaningful between logical lines -- while a
    statement has unclosed `(`/`[` (e.g. a multi-line kernel parameter
    list), indent/dedent tracking and NEWLINE emission are suppressed
    entirely, the same way Python's own tokenizer treats bracketed
    continuations, so a wrapped parameter list doesn't get misread as a
    nested block."""
    tokens: list[Token] = []
    indent_stack = [0]
    lines = source.split("\n")
    paren_depth = 0

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if paren_depth == 0 and (stripped == "" or stripped.startswith("#")):
            continue  # blank/comment-only lines never affect indentation

        if paren_depth == 0:
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            if indent > indent_stack[-1]:
                indent_stack.append(indent)
                tokens.append(Token("INDENT", "", line_no))
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token("DEDENT", "", line_no))
            if indent != indent_stack[-1]:
                raise LexError(f"line {line_no}: inconsistent indentation")

        pos = 0
        while pos < len(raw_line):
            m = _TOKEN_RE.match(raw_line, pos)
            if not m:
                raise LexError(f"line {line_no}: unexpected character {raw_line[pos]!r}")
            pos = m.end()
            kind = m.lastgroup
            text = m.group()
            if kind in ("WHITESPACE", "COMMENT", "NEWLINE"):
                continue
            if kind == "IDENT":
                if text in KEYWORDS:
                    tokens.append(Token("KEYWORD", text, line_no))
                elif text in TYPES:
                    tokens.append(Token("TYPE", text, line_no))
                else:
                    tokens.append(Token("IDENT", text, line_no))
            elif kind == "OP":
                tokens.append(Token("OP", text, line_no))
            elif kind == "PUNCT":
                if text in ("(", "["):
                    paren_depth += 1
                elif text in (")", "]"):
                    paren_depth = max(0, paren_depth - 1)
                tokens.append(Token("PUNCT", text, line_no))
            else:
                tokens.append(Token(kind, text, line_no))
        if paren_depth == 0:
            tokens.append(Token("NEWLINE", "", line_no))

    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token("DEDENT", "", len(lines)))
    tokens.append(Token("EOF", "", len(lines)))
    return tokens
