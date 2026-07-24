"""
Live viewer for GPUBVHPipeline (src/relativity_kernel_dsl/gpu_bvh_pipeline.py):
a real Karras binary-radix-tree BVH build + primary-ray traversal, authored
entirely in the Relativity kernel DSL (src/relativity_kernel_dsl/kernels/bvh_pipeline.rfrk)
and compiled through glslang -> spirv-cross -> SDL3 GPU/MSL -- the first
substantial workload proving that DSL end to end (see docs/bug_log.md
for the real bugs found building it: a spirv-cross/SDL3 function-
buffer interaction, and a radix-sort stability bug that looked exactly
like cross-instance GPU state corruption until properly diagnosed).

The scene (a static cluster of spheres) is built ONCE at startup -- only
the camera moves each frame. Every pixel on screen comes directly from
the GPUBVHPipeline.render() fragment shader; there is no CPU mesh or
separate compositing pass involved, since the raytrace shader already
produces the entire frame.

For a much smaller, simpler proof of the same core idea (a real SDL3
GPU compute dispatch coexisting with ordinary pygame.draw calls in one
frame), see relativity_pygame_sdl3_compute_minimal.py in this same directory,
or ../README.md for the full picture of this repo.

Scope (see bvh_pipeline.rfrk's own header): N<=256 spheres, one static
BLAS, primary-ray visibility only -- no shadows/reflections/multi-bounce
GI (that would be the next slice, a full wavefront path tracer).

IMPORTANT: like every SDL3 GPU demo in this repo, this CANNOT run under
SDL_VIDEODRIVER=dummy -- SDL3's Metal backend needs a real
windowing-system connection to create a device. Run interactively.

Requires Python 3.12+, pygame-ce + PySDL3 (see requirements.txt),
plus glslangValidator/spirv-cross on PATH (used once at startup to
compile the .rfrk kernels).

Run with:
    python examples/relativity_bvh_demo.py

Controls: left-drag orbit, scroll zoom, esc to quit.
"""

import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativity_kernel_dsl.gpu_bvh_pipeline import GPUBVHPipeline
from relativity_visualization.camera import OrbitCamera

WIDTH, HEIGHT = 1000, 700
N_SPHERES = 96


def build_scene(rng):
    """A few overlapping clusters rather than one uniform blob, so the
    BVH's spatial structure actually has something non-trivial to do."""
    clusters = [
        (np.array([-2.5, 0.0, 0.0]), 2.2),
        (np.array([2.0, 1.5, -1.0]), 1.8),
        (np.array([0.5, -2.0, 1.5]), 2.0),
    ]
    centers = np.zeros((N_SPHERES, 3), dtype=np.float32)
    per_cluster = N_SPHERES // len(clusters)
    for i, (center, spread) in enumerate(clusters):
        lo = i * per_cluster
        hi = N_SPHERES if i == len(clusters) - 1 else (i + 1) * per_cluster
        count = hi - lo
        centers[lo:hi] = center + rng.normal(0.0, spread, (count, 3))
    radii = rng.uniform(0.15, 0.5, N_SPHERES).astype(np.float32)
    return centers, radii


def main():
    rng = np.random.default_rng(7)
    centers, radii = build_scene(rng)

    pipeline = GPUBVHPipeline(N_SPHERES, render_width=WIDTH, render_height=HEIGHT)
    pipeline.set_spheres(centers, radii)
    pipeline.build()

    camera = OrbitCamera(distance=14.0, pitch=0.35, yaw=0.6, fov_deg=50.0, width=WIDTH, height=HEIGHT)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Relativity — Kernel-DSL BVH build + traverse (Karras tree, primary rays)")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(0)  # uncapped
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            camera.handle_event(event)

        eye, forward, right, up = camera.basis()
        pixels = pipeline.render(eye, forward, right, up, fov_y_degrees=camera.fov_deg)
        frame = pygame.image.frombuffer(pixels.tobytes(), (WIDTH, HEIGHT), "RGBA")
        screen.blit(frame, (0, 0))

        hud = [
            f"kernel-DSL BVH: {N_SPHERES} spheres, {pipeline.n2} nodes (Karras binary-radix tree)",
            f"real Karras split-search + stable serial radix sort + primary-ray traversal, all GLSL-compiled",
            f"fps={clock.get_fps():4.1f}   left-drag: orbit   scroll: zoom   esc: quit",
        ]
        y = 8
        for line in hud:
            surf = font.render(line, True, (225, 230, 240))
            screen.blit(surf, (10, y))
            y += 18

        pygame.display.flip()

    pipeline.close()
    pygame.quit()


if __name__ == "__main__":
    main()
