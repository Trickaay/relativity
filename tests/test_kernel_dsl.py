"""Regression coverage for the Relativity kernel DSL (relativity_kernel_dsl/): a small
custom lexer/parser/codegen that compiles a Taichi-ergonomics kernel
language (automatic parallel-`for` dispatch, no manual GLSL binding-slot
bookkeeping) down to real GLSL, then through the already-proven
glslang -> spirv-cross -> SDL3 GPU/MSL toolchain -- see
pygame_sdl3_upstream_plan/agent_cognition_vision memories for why this
targets GLSL rather than an actual Taichi runtime dependency.

Lexer/parser tests are pure and always run. The round-trip and fold
tests need a real SDL3 GPU device (same caveat as
test_sdl3_gpu_bridge.py/test_gpu_fluid_pipeline.py -- SDL3's Metal
backend needs a real windowing-system connection, unavailable under the
dummy driver conftest.py forces for the rest of the suite) plus the
glslang/spirv-cross CLI tools, so they self-skip under the dummy driver.
"""

import os
import struct

import numpy as np
import pytest

from relativity_kernel_dsl.lexer import tokenize
from relativity_kernel_dsl.parser import parse
from relativity_kernel_dsl import ast_nodes as A

pytestmark_gpu = pytest.mark.skipif(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    reason="SDL3's Metal GPU backend needs a real windowing system connection, "
           "not available under the dummy driver conftest.py forces for the rest of the suite",
)


# ---------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------

def test_lexer_tokenizes_keywords_types_and_operators():
    tokens = tokenize("kernel foo(n: i32):\n    for i in range(n):\n        x = 0x1Fu\n")
    kinds_values = [(t.kind, t.value) for t in tokens if t.kind not in ("NEWLINE",)]
    assert ("KEYWORD", "kernel") in kinds_values
    assert ("IDENT", "foo") in kinds_values
    assert ("TYPE", "i32") in kinds_values
    assert ("KEYWORD", "for") in kinds_values
    assert ("KEYWORD", "range") in kinds_values


def test_lexer_emits_indent_dedent_matching_block_structure():
    tokens = tokenize("kernel foo(n: i32):\n    for i in range(n):\n        x = 1\n    y = 2\n")
    kinds = [t.kind for t in tokens]
    assert kinds.count("INDENT") == 2
    assert kinds.count("DEDENT") == 2


def test_lexer_distinguishes_hex_from_decimal_literals():
    tokens = tokenize("x = 0xFF\ny = 255\n")
    hexnums = [t for t in tokens if t.kind == "HEXNUMBER"]
    ints = [t for t in tokens if t.kind == "INT"]
    assert hexnums[0].value == "0xFF"
    assert ints[0].value == "255"


def test_lexer_longest_match_first_for_multichar_operators():
    tokens = tokenize("x = a << 2\ny = a <= 2\n")
    ops = [t.value for t in tokens if t.kind == "OP"]
    assert "<<" in ops
    assert "<=" in ops
    assert "<" not in ops  # neither should be split into a lone '<'


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------

def test_parser_builds_field_decl():
    module = parse("field positions: vec4[]\n")
    assert module.fields == [A.FieldDecl(name="positions", elem_type="vec4")]


def test_parser_builds_kernel_with_for_range_and_assign():
    module = parse(
        "field data: f32[]\n"
        "\n"
        "kernel double_it(n: i32):\n"
        "    for i in range(n):\n"
        "        data[i] = data[i] * 2.0\n"
    )
    assert len(module.kernels) == 1
    kernel = module.kernels[0]
    assert kernel.name == "double_it"
    assert kernel.stage == "compute"
    assert len(kernel.params) == 1 and kernel.params[0].name == "n"
    outer = kernel.body[0]
    assert isinstance(outer, A.ForRange) and outer.var == "i"
    assign = outer.body[0]
    assert isinstance(assign, A.Assign)
    assert isinstance(assign.target, A.Index)


def test_parser_builds_func_with_return_type():
    module = parse(
        "func expand(v: u32) -> u32:\n"
        "    return v\n"
    )
    assert len(module.funcs) == 1
    assert module.funcs[0].return_type == "u32"


def test_parser_respects_expression_precedence():
    # `and` binds looser than comparisons, which bind looser than `|`
    module = parse(
        "kernel k(n: i32):\n"
        "    for i in range(n):\n"
        "        if n > 0 and n < 10:\n"
        "            n = n\n"
    )
    if_stmt = module.kernels[0].body[0].body[0]
    assert isinstance(if_stmt, A.If)
    cond = if_stmt.cond
    assert isinstance(cond, A.BinOp) and cond.op == "and"
    assert isinstance(cond.left, A.BinOp) and cond.left.op == ">"
    assert isinstance(cond.right, A.BinOp) and cond.right.op == "<"


def test_parser_builds_shader_stage_distinct_from_kernel():
    module = parse("shader raytrace(n: i32):\n    out_color = vec3(1.0, 0.0, 0.0)\n")
    assert module.kernels[0].stage == "fragment"


def test_parser_builds_fold_decl():
    module = parse(
        "field values: f32[]\n"
        "\n"
        "kernel reduce(n: i32):\n"
        "    fold acc: f32 = 1e30\n"
        "    for i in range(n):\n"
        "        acc = min(acc, values[i])\n"
    )
    fold = module.kernels[0].body[0]
    assert isinstance(fold, A.FoldDecl) and fold.name == "acc" and fold.type == "f32"


# ---------------------------------------------------------------------
# Round-trip: real compile + real GPU dispatch
# ---------------------------------------------------------------------

@pytestmark_gpu
def test_round_trip_double_every_element_matches_numpy(tmp_path):
    from relativity_kernel_dsl.compiler import compile_module
    from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

    source = (
        "field data: f32[]\n"
        "\n"
        "kernel double_it(n: i32):\n"
        "    for i in range(n):\n"
        "        data[i] = data[i] * 2.0\n"
    )
    compiled = compile_module(source, workdir=str(tmp_path))
    shader = compiled["double_it"]
    msl_path = tmp_path / "double_it.msl"
    msl_path.write_text(shader.msl_source)

    renderer = SDL3GPUOffscreenRenderer(64, 64)
    pipeline = renderer.create_compute_pipeline(
        str(msl_path), num_readwrite_storage_buffers=len(shader.bindings),
        num_uniform_buffers=1, threadcount=(64, 1, 1))

    n = 100
    data = np.arange(n, dtype=np.float32)
    buf = renderer.create_storage_buffer(n * 4, initial_data=data)

    group_counts = ((n + 63) // 64, 1, 1)
    renderer.dispatch_compute(pipeline, [], [], group_counts,
                               uniform_bytes=struct.pack("<i", n), storage_buffers=[buf])

    result = renderer.download_buffer(buf, n * 4, dtype=np.float32)
    assert np.allclose(result, data * 2.0)


@pytestmark_gpu
def test_fold_min_max_matches_numpy_including_negative_and_subunit_values(tmp_path):
    from relativity_kernel_dsl.compiler import compile_module
    from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

    source = (
        "field values: f32[]\n"
        "\n"
        "kernel reduce_minmax(n: i32):\n"
        "    fold result_min: f32 = 1e30\n"
        "    fold result_max: f32 = -1e30\n"
        "    for i in range(n):\n"
        "        result_min = min(result_min, values[i])\n"
        "        result_max = max(result_max, values[i])\n"
    )
    compiled = compile_module(source, workdir=str(tmp_path))
    shader = compiled["reduce_minmax"]
    msl_path = tmp_path / "reduce_minmax.msl"
    msl_path.write_text(shader.msl_source)

    renderer = SDL3GPUOffscreenRenderer(64, 64)
    pipeline = renderer.create_compute_pipeline(
        str(msl_path), num_readwrite_storage_buffers=len(shader.bindings),
        num_uniform_buffers=1, threadcount=(64, 1, 1))

    rng = np.random.default_rng(0)
    n = 237
    # Deliberately includes negative and sub-1 magnitude values -- the case
    # that specifically exercises the order-preserving float<->uint bitcast
    # trick (naive unsigned comparison of raw float bits gets negative
    # numbers backwards; this is what would catch that bug).
    values = rng.uniform(-500.0, 0.75, n).astype(np.float32)

    def float_flip_encode(f):
        # Must match _relativity_float_flip's GLSL convention exactly (codegen_glsl.py):
        # sign bit set (negative) -> full complement (0xFFFFFFFF); sign bit
        # clear (positive) -> just flip the top bit (0x80000000).
        bits = np.float32(f).view(np.uint32)
        mask = np.uint32(0xFFFFFFFF) if (bits & np.uint32(0x80000000)) else np.uint32(0x80000000)
        return np.uint32(bits ^ mask)

    def float_unflip(bits):
        bits = np.uint32(bits)
        mask = np.uint32(0xFFFFFFFF) if not (bits & np.uint32(0x80000000)) else np.uint32(0x80000000)
        return np.uint32(bits ^ mask).view(np.float32)

    buffers = {}
    for b in shader.bindings:
        if b.kind == "fold":
            bits = np.array([float_flip_encode(b.identity)], dtype=np.uint32)
            buffers[b.name] = renderer.create_storage_buffer(4, initial_data=bits)
        else:
            buffers[b.name] = renderer.create_storage_buffer(4 * n, initial_data=values)

    ordered_buffers = [buffers[b.name] for b in shader.bindings]
    group_counts = ((n + 63) // 64, 1, 1)
    renderer.dispatch_compute(pipeline, [], [], group_counts,
                               uniform_bytes=struct.pack("<i", n), storage_buffers=ordered_buffers)

    gpu_min = float_unflip(renderer.download_buffer(buffers["result_min"], 4, dtype=np.uint32)[0])
    gpu_max = float_unflip(renderer.download_buffer(buffers["result_max"], 4, dtype=np.uint32)[0])

    assert np.isclose(gpu_min, values.min())
    assert np.isclose(gpu_max, values.max())
