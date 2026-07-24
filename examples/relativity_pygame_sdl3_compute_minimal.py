"""
Minimal proof of concept: a real SDL3 GPU **compute** dispatch (authored
in the Relativity kernel DSL -- see relativity_kernel_dsl/compiler.py -- and compiled
through the same glslang -> spirv-cross -> SDL3/MSL pipeline every other
GPU demo in this project uses) running every frame alongside ordinary
pygame.draw calls, in the SAME pygame window, on the SAME frame. That
coexistence -- a real compute/shader pipeline and pygame's legacy 2D
drawing API both working in one process without either replacing the
other -- is the one thing this whole subsystem exists to demonstrate.

Deliberately much simpler than relativity_bvh_demo.py (a full BVH build +
raytraced-fragment-shader pipeline, where the GPU shader produces every
pixel on screen). Here the GPU does ONLY the numeric work -- a per-point
wave height, computed in parallel for every point, every frame -- and
100% of the actual pixel drawing is ordinary pygame.draw.circle calls.
That split (GPU crunches numbers in parallel; pygame still owns drawing)
is arguably the more broadly useful story: it generalizes to any
parallel numeric workload, not just raytracing.

Dependencies: numpy, pygame-ce, sdl3 (PySDL3), plus glslangValidator and
spirv-cross on PATH at import time (compiles the ~6-line kernel below
once, at startup). No Taichi, no scikit-image, nothing else from this
monorepo's broader requirements.txt -- this subsystem's real footprint.

IMPORTANT: like every SDL3-GPU demo in this project, this needs a real
windowing-system connection (SDL3's Metal backend on macOS) and will NOT
run under a headless/dummy SDL driver.

Run with:
    python examples/relativity_pygame_sdl3_compute_minimal.py

Controls: esc to quit.
"""

import math
import os
import struct
import sys
import tempfile

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativity_kernel_dsl.compiler import compile_module
from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

WIDTH, HEIGHT = 900, 500
N_POINTS = 200

# The entire GPU-side program: one field read (per-point phase), one
# field written (per-point wave height), one kernel. `n`/`time`/`speed`/
# `amplitude` all become one compiled uniform buffer automatically; `n`
# also drives the parallel dispatch size (see compiler.py's
# dispatch_param) and the auto-generated bounds check -- none of that is
# hand-written here.
KERNEL_SOURCE = """
field phase: f32[]
field wave_y: f32[]

kernel compute_wave(n: i32, time: f32, speed: f32, amplitude: f32):
    for i in range(n):
        wave_y[i] = sin(phase[i] * speed + time) * amplitude
"""

_UBO_SCALAR_PACK = {"f32": "f", "i32": "i", "u32": "I"}


def _pack_scalar_ubo(params, values):
    """std140 packing for an all-scalar (no vec3/vec4) uniform block --
    every member here is 4-byte aligned, so this is just a sequential
    pack, no padding logic needed (contrast with gpu_bvh_pipeline.py's
    _pack_ubo, which also has to handle vec3's 16-byte alignment)."""
    fmt = "<" + "".join(_UBO_SCALAR_PACK[p.type] for p in params)
    return struct.pack(fmt, *(values[p.name] for p in params))


def main():
    workdir = tempfile.mkdtemp(prefix="relativity_compute_minimal_")
    compiled = compile_module(KERNEL_SOURCE, workdir=workdir)
    shader = compiled["compute_wave"]

    # Used purely for its SDL3 GPU device + compute dispatch here -- no
    # fragment/graphics pipeline is loaded at all, unlike every other
    # demo's GPUXRenderer wrapper (this class's compute path is fully
    # independent of its graphics path, see its own docstring).
    gpu = SDL3GPUOffscreenRenderer(WIDTH, HEIGHT)
    phase_buf = gpu.create_storage_buffer(N_POINTS * 4)
    wave_buf = gpu.create_storage_buffer(N_POINTS * 4)

    phase = np.linspace(0.0, 4 * math.pi, N_POINTS, dtype=np.float32)
    gpu.upload_buffer_data(phase_buf, phase)

    msl_path = os.path.join(workdir, "compute_wave.msl")
    pipeline = gpu.create_compute_pipeline(
        msl_path, num_readwrite_storage_buffers=len(shader.bindings),
        num_uniform_buffers=1, threadcount=(64, 1, 1))
    # Binding order is resolved from the actually-compiled MSL (see
    # compiler.py's _true_storage_buffer_order), never hand-tracked --
    # look up each buffer by the field name that produced it.
    buffers_by_name = {"phase": phase_buf, "wave_y": wave_buf}
    ordered_buffers = [buffers_by_name[b.name] for b in shader.bindings]

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("pygame + SDL3 GPU compute, one frame, coexisting")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    running = True
    t = 0.0
    while running:
        dt = clock.tick(60) / 1000.0
        t += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # --- the GPU compute dispatch ---
        uniform_bytes = _pack_scalar_ubo(shader.uniform_params,
                                          {"n": N_POINTS, "time": t, "speed": 1.0, "amplitude": 80.0})
        group_counts = (max(1, math.ceil(N_POINTS / shader.local_size)), 1, 1)
        gpu.dispatch_compute(pipeline, [], [], group_counts,
                              uniform_bytes=uniform_bytes, storage_buffers=ordered_buffers)
        wave_y = gpu.download_buffer(wave_buf, N_POINTS * 4, dtype=np.float32)

        # --- 100% ordinary pygame drawing, using the GPU's numbers ---
        screen.fill((12, 14, 20))
        xs = np.linspace(40, WIDTH - 40, N_POINTS)
        for x, y_off in zip(xs, wave_y):
            pygame.draw.circle(screen, (120, 200, 255), (int(x), int(HEIGHT / 2 + y_off)), 3)

        hud = font.render(
            f"GPU compute (SDL3/MSL, via relativity_kernel_dsl) + pygame.draw, same frame -- fps={clock.get_fps():4.1f}",
            True, (225, 230, 240))
        screen.blit(hud, (10, 10))
        pygame.display.flip()

    gpu.close()
    pygame.quit()


if __name__ == "__main__":
    main()
