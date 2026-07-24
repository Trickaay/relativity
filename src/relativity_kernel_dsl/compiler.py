"""
Facade for the Relativity kernel DSL (see lexer.py/parser.py/codegen_glsl.py for
the pieces): parses .rfrk source, generates GLSL, and drives it through
the same glslang -> spirv-cross -> MSL toolchain every other shader in
src/relativity_visualization/shaders/ is built with -- this module just
automates those CLI invocations instead of running them by hand each time.

Real finding from building this (empirically confirmed, not assumed):
storage TEXTURES' readonly/readwrite split is something spirv-cross
infers from actual imageLoad/imageStore usage and preserves into the
compiled MSL (already established, see the fluid-pipeline work) -- but
storage BUFFERS do NOT get this treatment: even with an explicit `readonly
buffer` qualifier in the generated GLSL (confirmed present in the SPIR-V
as a real `NonWritable` decoration via spirv-dis), spirv-cross's MSL
output uses a plain `device Foo&` for every storage buffer regardless,
never `const device`. So there is no way to recover the readonly/
readwrite split from the compiled MSL text for buffers the way there is
for textures. Given the DSL's own Python-side usage analysis (see
codegen_glsl.py's `_RefCollector`) already correctly determines which
buffers are ever written, and a "readonly by usage" buffer bound through
a mutable slot is still fully correct (the compiled shader body simply
never issues a write to it), this module sidesteps the ambiguity
entirely: every storage buffer is bound through one unified,
read-write-capable list, ordered by the compiled MSL's actual
[[buffer(N)]] assignment (still necessary -- spirv-cross does NOT
preserve GLSL source binding-number order for buffers either, confirmed
by this module's own development: a scene_bounds kernel's UBO/positions/
fold-buffers came out as MSL slots 0/1/2/3 vs GLSL-source slots 3/2/0/1).
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import ast_nodes as A
from .parser import parse
from .codegen_glsl import compile_kernel, CompiledKernel, BindingInfo

_MSL_ARG_RE = re.compile(
    r"(constant|device)\s+(\w+)&\s+\w+\s+\[\[buffer\((\d+)\)\]\]")


class CompileError(Exception):
    pass


@dataclass
class CompiledShader:
    name: str
    stage: str                 # "compute" | "fragment"
    glsl_source: str
    msl_source: str
    bindings: list             # list[BindingInfo], reordered into true MSL storage-buffer slot order
    uniform_params: list       # list[A.Param], UBO member order (matches the UBO struct in msl_source)
    dispatch_param: str = None
    local_size: int = 64


class CompiledModule:
    def __init__(self, shaders: dict):
        self.shaders = shaders  # name -> CompiledShader

    def __getitem__(self, name):
        return self.shaders[name]


def _run(cmd, error_context):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CompileError(f"{error_context} failed:\n{result.stdout}\n{result.stderr}")


def _compile_glsl_to_msl(glsl_source: str, stage: str, workdir: Path, name: str) -> str:
    ext = ".comp" if stage == "compute" else ".frag"
    src_path = workdir / f"{name}{ext}"
    spv_path = workdir / f"{name}.spv"
    msl_path = workdir / f"{name}.msl"
    src_path.write_text(glsl_source)

    _run(["glslangValidator", "-V", "--target-env", "vulkan1.1", str(src_path), "-o", str(spv_path)],
         f"glslangValidator ({name})")
    _run(["spirv-cross", "--msl", str(spv_path), "--output", str(msl_path)],
         f"spirv-cross ({name})")
    return msl_path.read_text()


def _true_storage_buffer_order(msl_source: str, bindings: list) -> list:
    """Returns `bindings` reordered to match the compiled MSL's actual
    [[buffer(N)]] assignment for storage buffers (the UBO's own slot is
    irrelevant here -- SDL3 tracks uniform buffers through a separate
    binding call/count from storage buffers, matching this bridge's
    existing dispatch_compute/create_compute_pipeline convention)."""
    sig_match = re.search(r"(?:kernel void main0|fragment .*? main0)\s*\((.*?)\)\s*\{", msl_source, re.DOTALL)
    if not sig_match:
        raise CompileError("could not locate main0 signature in compiled MSL")
    signature = sig_match.group(1)

    struct_to_slot = {}
    for qualifier, struct_name, slot in _MSL_ARG_RE.findall(signature):
        struct_to_slot[struct_name] = int(slot)

    def _slot_for(b: BindingInfo) -> int:
        if b.struct_name not in struct_to_slot:
            raise CompileError(f"binding '{b.name}' (struct {b.struct_name}) not found in compiled MSL signature")
        return struct_to_slot[b.struct_name]

    # Sort by the true MSL buffer(N) slot to get the real relative order,
    # then re-number 0..len-1 within just this storage-buffer list -- the
    # raw MSL slot numbers aren't usable directly since they share a
    # numbering space with the UBO's own slot (a separate SDL3 binding
    # category, tracked via uniform_params/dispatch_compute's own uniform
    # call, not this buffer list), which would leave gaps if used as-is.
    ordered = sorted(bindings, key=_slot_for)
    for i, b in enumerate(ordered):
        b.binding = i
    return ordered


def compile_module(source_text: str, workdir: str = None) -> CompiledModule:
    """Parses `source_text` (.rfrk), compiles every kernel/shader to GLSL,
    then to SPIR-V, then to MSL, and returns a CompiledModule with each
    shader's compiled artifacts + a binding order corrected to match what
    the compiled shader actually expects."""
    module = parse(source_text)

    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(workdir) if workdir else Path(tmp)
        wd.mkdir(parents=True, exist_ok=True)

        shaders = {}
        for kernel in module.kernels:
            compiled: CompiledKernel = compile_kernel(kernel, module)
            msl_source = _compile_glsl_to_msl(compiled.glsl_source, compiled.stage, wd, compiled.name)
            ordered_bindings = _true_storage_buffer_order(msl_source, compiled.bindings)
            shaders[kernel.name] = CompiledShader(
                name=compiled.name, stage=compiled.stage,
                glsl_source=compiled.glsl_source, msl_source=msl_source,
                bindings=ordered_bindings, uniform_params=compiled.uniform_params,
                dispatch_param=compiled.dispatch_param, local_size=compiled.local_size,
            )
        return CompiledModule(shaders)
