"""
GPUBVHPipeline -- host orchestration for the BVH build+traverse pipeline
authored in bvh_pipeline.rfrk (see that file and compiler.py's module
docstring for the full picture). Same shape as
relativity_visualization.gpu_fluid_pipeline.GPUFluidPipeline: allocates GPU
buffers, compiles the kernel module once, and exposes a small
build()/render() API that reads like plain Python -- dispatch group
counts, storage-buffer binding order, and UBO packing are all handled
here from the compiled manifest, never hand-tracked by the caller.

Node layout: leaves at global indices [0, n-1], internal nodes at
[n, 2n-2] (internal node 0 of a Karras binary-radix tree is always the
root, so the root's global index is always exactly `n`).

Scope: N<=256 spheres (single-workgroup radix sort -- see bvh_pipeline.rfrk's
own header), one static BLAS, primary-ray visibility only.

See ../../README.md for the broader picture of this repo (the DSL,
its dependency footprint, and a much smaller minimal example).

Real correctness deviation from the plan's "ceil(log2(n)) iterations" for
bounds_build_iter/escape_index_iter: a Karras tree CAN degenerate to a
near-linear chain for adversarially-ordered morton codes (worst-case
depth close to n, not O(log n)) -- ceil(log2(n)) iterations would only be
guaranteed sufficient for a balanced tree. Using n2 iterations instead
(a safe upper bound on any possible tree depth, since no tree over n
leaves can be deeper than n-1) costs a handful of extra trivial GPU
dispatches at this pass's N<=256 scale and removes the risk entirely --
worth the small extra cost for a guarantee instead of an assumption.
"""

import math
import struct
from pathlib import Path

import numpy as np

from .compiler import compile_module
from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

_KERNEL_SOURCE_PATH = Path(__file__).parent / "kernels" / "bvh_pipeline.rfrk"
_BUILD_DIR = Path(__file__).parent / "_build"
_FULLSCREEN_VERT_MSL = Path(__file__).parent.parent / "relativity_visualization" / "shaders" / "fullscreen_triangle.msl"

# std140 uniform-block layout rules, verified empirically against
# glslang's actual compiled UBO member offsets (see compiler.py/plan
# notes): scalars are 4-byte aligned, vec3 is 16-byte ALIGNED (but only
# consumes 12 bytes -- a following scalar can pack right after it).
_STD140_SIZE_ALIGN = {"f32": (4, 4), "i32": (4, 4), "u32": (4, 4), "vec3": (12, 16), "vec4": (16, 16)}


def _pack_ubo(params, values) -> bytes:
    cursor = 0
    chunks = []
    for p in params:
        size, align = _STD140_SIZE_ALIGN[p.type]
        aligned = (cursor + align - 1) // align * align
        if aligned > cursor:
            chunks.append(b"\x00" * (aligned - cursor))
        cursor = aligned
        v = values[p.name]
        if p.type == "vec3":
            chunks.append(struct.pack("<3f", *v))
        elif p.type == "vec4":
            chunks.append(struct.pack("<4f", *v))
        elif p.type == "f32":
            chunks.append(struct.pack("<f", float(v)))
        elif p.type == "i32":
            chunks.append(struct.pack("<i", int(v)))
        elif p.type == "u32":
            chunks.append(struct.pack("<I", int(v)))
        cursor += size
    total = (cursor + 15) // 16 * 16
    if total > cursor:
        chunks.append(b"\x00" * (total - cursor))
    return b"".join(chunks)


def _float_flip(f) -> int:
    """Order-preserving float->uint32 encode, matching bvh_pipeline.rfrk's
    _relativity_float_flip exactly (see codegen_glsl.py) -- used to seed a fold
    accumulator's identity value before dispatch."""
    bits = np.float32(f).view(np.uint32)
    mask = np.uint32(0xFFFFFFFF) if (bits & np.uint32(0x80000000)) else np.uint32(0x80000000)
    return int(np.uint32(bits ^ mask))


def _float_unflip(bits) -> float:
    bits = np.uint32(bits)
    mask = np.uint32(0xFFFFFFFF) if not (bits & np.uint32(0x80000000)) else np.uint32(0x80000000)
    return float(np.uint32(bits ^ mask).view(np.float32))


class GPUBVHPipeline:
    def __init__(self, n_primitives, render_width=640, render_height=480):
        assert n_primitives <= 256, "single-workgroup radix sort scope, see bvh_pipeline.rfrk's header"
        self.n = n_primitives
        self.n2 = 2 * n_primitives - 1
        self.width = render_width
        self.height = render_height

        self.compiled = compile_module(_KERNEL_SOURCE_PATH.read_text(), workdir=str(_BUILD_DIR))
        self.renderer = SDL3GPUOffscreenRenderer(render_width, render_height)

        n, n2 = self.n, self.n2
        r = self.renderer
        self.buffers = {
            "positions": r.create_storage_buffer(n * 16, graphics_readable=True),
            "morton": r.create_storage_buffer(n * 4),
            "morton_tmp": r.create_storage_buffer(n * 4),
            "prim_index": r.create_storage_buffer(n * 4),
            "prim_index_tmp": r.create_storage_buffer(n * 4),
            "hist": r.create_storage_buffer(256 * 4),
            "prefix": r.create_storage_buffer(256 * 4),
            "bvh_bmin": r.create_storage_buffer(n2 * 16, graphics_readable=True),
            "bvh_bmax": r.create_storage_buffer(n2 * 16, graphics_readable=True),
            "bvh_left": r.create_storage_buffer(n2 * 4, graphics_readable=True),
            "bvh_right": r.create_storage_buffer(n2 * 4),
            "bvh_escape": r.create_storage_buffer(n2 * 4, graphics_readable=True),
            "bounds_min": r.create_storage_buffer(12),
            "bounds_max": r.create_storage_buffer(12),
        }

        self.pipelines = {}
        for name, shader in self.compiled.shaders.items():
            if shader.stage != "compute":
                continue
            msl_path = str(_BUILD_DIR / f"{name}.msl")
            self.pipelines[name] = r.create_compute_pipeline(
                msl_path, num_readwrite_storage_buffers=len(shader.bindings),
                num_uniform_buffers=1 if shader.uniform_params else 0,
                threadcount=(64, 1, 1))

        raytrace = self.compiled["raytrace"]
        frag_msl_path = str(_BUILD_DIR / "raytrace.msl")
        r.load_shader_pair(str(_FULLSCREEN_VERT_MSL), frag_msl_path,
                            frag_uniform_buffers=1, frag_samplers=0,
                            frag_storage_buffers=len(raytrace.bindings))

    def _dispatch(self, name, buffer_overrides=None, **scalar_kwargs):
        shader = self.compiled[name]
        overrides = buffer_overrides or {}
        ordered_buffers = [overrides.get(b.name, self.buffers.get(b.name)) for b in shader.bindings]
        uniform_bytes = _pack_ubo(shader.uniform_params, scalar_kwargs) if shader.uniform_params else None
        n_val = scalar_kwargs[shader.dispatch_param]
        group_counts = (max(1, math.ceil(n_val / shader.local_size)), 1, 1)
        self.renderer.dispatch_compute(self.pipelines[name], [], [], group_counts,
                                        uniform_bytes=uniform_bytes, storage_buffers=ordered_buffers)

    def set_spheres(self, centers_xyz, radii):
        centers_xyz = np.asarray(centers_xyz, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        data = np.zeros((self.n, 4), dtype=np.float32)
        data[:, :3] = centers_xyz
        data[:, 3] = radii
        self.renderer.upload_buffer_data(self.buffers["positions"], data)

    def build(self):
        n, n2 = self.n, self.n2

        min_bits = np.full(3, _float_flip(1e30), dtype=np.uint32)
        max_bits = np.full(3, _float_flip(-1e30), dtype=np.uint32)
        self.renderer.upload_buffer_data(self.buffers["bounds_min"], min_bits)
        self.renderer.upload_buffer_data(self.buffers["bounds_max"], max_bits)
        self._dispatch("scene_bounds", n=n)

        min_raw = self.renderer.download_buffer(self.buffers["bounds_min"], 12, dtype=np.uint32)
        max_raw = self.renderer.download_buffer(self.buffers["bounds_max"], 12, dtype=np.uint32)
        scene_min = np.array([_float_unflip(b) for b in min_raw], dtype=np.float32)
        scene_max = np.array([_float_unflip(b) for b in max_raw], dtype=np.float32)
        extent = float(np.max(scene_max - scene_min))
        scene_scale = 1023.0 / extent if extent > 1e-9 else 1.0
        self.scene_min, self.scene_max = scene_min, scene_max

        self._dispatch("morton_encode", n=n, scene_min=tuple(scene_min), scene_scale=scene_scale)

        # Radix sort: 4 passes (8 bits each -- covers the 30-bit morton
        # code), ping-ponging which physical buffer plays "current" vs
        # "_tmp" without any GPU copy (see bvh_pipeline.rfrk's header).
        # Each pass is one dispatch of the fully serial radix_sort_pass
        # kernel (histogram + prefix sum + stable scatter in one thread --
        # see that kernel's header for why this must be serial, not
        # atomic-parallel, for correctness).
        code_bufs = [self.buffers["morton"], self.buffers["morton_tmp"]]
        idx_bufs = [self.buffers["prim_index"], self.buffers["prim_index_tmp"]]
        cur = 0
        for pass_i in range(4):
            shift = pass_i * 8
            self._dispatch("radix_sort_pass", buffer_overrides={
                "morton": code_bufs[cur], "prim_index": idx_bufs[cur],
                "morton_tmp": code_bufs[1 - cur], "prim_index_tmp": idx_bufs[1 - cur],
            }, one=1, n=n, shift=shift)
            cur = 1 - cur
        assert cur == 0, "4 (even) passes must leave the sorted result in the originally-named buffers"

        self._dispatch("karras_build", n=n, n_internal=max(1, n - 1))
        self._dispatch("init_leaves", n=n)

        for _ in range(n2):
            self._dispatch("bounds_build_iter", n2=n2)

        escape_init = np.zeros(n2, dtype=np.int32)
        escape_init[n] = -1  # root (internal node 0) is never anyone's child, so this seed persists
        self.renderer.upload_buffer_data(self.buffers["bvh_escape"], escape_init)
        for _ in range(n2):
            self._dispatch("escape_index_iter", n2=n2)

    def download_nodes(self):
        """Debug/test readback: returns bmin/bmax (n2,3), left/right/escape
        (n2,) int32 arrays for direct correctness checking."""
        n2 = self.n2
        bmin = self.renderer.download_buffer(self.buffers["bvh_bmin"], n2 * 16, dtype=np.float32).reshape(n2, 4)[:, :3]
        bmax = self.renderer.download_buffer(self.buffers["bvh_bmax"], n2 * 16, dtype=np.float32).reshape(n2, 4)[:, :3]
        left = self.renderer.download_buffer(self.buffers["bvh_left"], n2 * 4, dtype=np.int32)
        right = self.renderer.download_buffer(self.buffers["bvh_right"], n2 * 4, dtype=np.int32)
        escape = self.renderer.download_buffer(self.buffers["bvh_escape"], n2 * 4, dtype=np.int32)
        return bmin.copy(), bmax.copy(), left.copy(), right.copy(), escape.copy()

    def render(self, cam_pos, cam_forward, cam_right, cam_up, fov_y_degrees=60.0):
        raytrace = self.compiled["raytrace"]
        aspect = self.width / self.height
        tan_half_fov = math.tan(math.radians(fov_y_degrees) / 2.0)
        values = {
            "n2": self.n2, "root": self.n,
            "cam_pos": tuple(cam_pos), "cam_forward": tuple(cam_forward),
            "cam_right": tuple(cam_right), "cam_up": tuple(cam_up),
            "aspect": aspect, "tan_half_fov": tan_half_fov,
            "width": float(self.width), "height": float(self.height),
        }
        uniform_bytes = _pack_ubo(raytrace.uniform_params, values)
        ordered_buffers = [self.buffers[b.name] for b in raytrace.bindings]
        return self.renderer.render_frame(fragment_uniform_bytes=uniform_bytes,
                                           fragment_storage_buffers=ordered_buffers)

    def close(self):
        self.renderer.close()
