"""
Minimal proof of concept: the actual procedural-planet GPU shader work
(a biome-ladder rocky surface -- water/sand/vegetation/rock/ice -- with
a real ray-marched Rayleigh/Mie atmosphere layered on top) composited
onto an ordinary pygame window every frame. Same "SDL3 GPU pipeline and
pygame's legacy 2D API coexist in one frame" story as
relativity_pygame_sdl3_compute_minimal.py, but showing the actual visual
payoff instead of a bare sine wave.

To be accurate about what this does and doesn't demonstrate:
planet_surface.frag and atmosphere.frag are hand-written GLSL, compiled
through the same glslang -> spirv-cross -> SDL3/MSL toolchain the
kernel-DSL work uses -- they are NOT authored in the .rfrk DSL itself
(only the BVH pipeline, see relativity_bvh_demo.py / relativity_kernel_dsl, is). This
example and the DSL/BVH one are two separate achievements sharing one
toolchain, not one combined thing -- see ../README.md.

Dependencies: numpy, pygame-ce, sdl3 (PySDL3), plus glslangValidator and
spirv-cross on PATH (compiled once, lazily, the first time each shader
is used, and cached under relativity_visualization/shaders/).

IMPORTANT: like every SDL3-GPU demo in this project, this needs a real
windowing-system connection (SDL3's Metal backend on macOS) and will
NOT run under a headless/dummy SDL video driver.

Run with:
    python examples/relativity_pygame_planet_showcase_minimal.py

Controls: esc to quit.
"""

import os
import sys

import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativity_visualization.gpu_planet_renderer import GPUPlanetRenderer
from relativity_visualization.gpu_celestial_renderer import GPUCelestialRenderer
from relativity_visualization.overlay_widgets import make_nebula_starfield

WIDTH, HEIGHT = 900, 650
PLANET_SCREEN_RADIUS = 220
ROTATION_SPEED = 0.25
LIGHT_DIR = (0.4, 0.3, -1.0)


def main():
    planet_renderer = GPUPlanetRenderer()
    celestial_renderer = GPUCelestialRenderer()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Procedural planet + atmosphere (GPU shaders) + pygame, one frame")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    background = make_nebula_starfield(WIDTH, HEIGHT)

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

        # --- the GPU shader work: a real biome-ladder rocky planet plus
        # a real ray-marched Rayleigh/Mie atmosphere, each rendered to a
        # fixed 256x256 sprite. planet_radius=1.0/atmosphere_radius=1.05
        # matters here: both shaders share the same analytic-sphere ray
        # convention, and the atmosphere must use the planet's own
        # radius=1.0 or it renders smaller than the planet and vanishes
        # behind it entirely (a real bug found and fixed wiring this
        # into the solar-system demo -- see atmosphere_and_biome_wiring
        # memory / run_scene.py's render_atmosphere_layers). ---
        planet_rgba = planet_renderer.render_planet(
            light_dir_local=LIGHT_DIR, rotation_phase=t * ROTATION_SPEED,
            palette_low=(0.12, 0.32, 0.55), palette_high=(0.45, 0.42, 0.28),
            kind="rocky", seed=3.0, turbulence=0.6, terrain_strength=0.15, earthlike=True)
        atmosphere_rgba = celestial_renderer.render_atmosphere(
            sun_dir_local=LIGHT_DIR, planet_radius=1.0, atmosphere_radius=1.05)

        # --- 100% ordinary pygame compositing from here ---
        screen.blit(background, (0, 0))
        size = PLANET_SCREEN_RADIUS * 2
        cx, cy = WIDTH // 2, HEIGHT // 2
        for rgba in (planet_rgba, atmosphere_rgba):
            surf = pygame.image.frombuffer(rgba.tobytes(), (rgba.shape[1], rgba.shape[0]), "RGBA").convert_alpha()
            surf = pygame.transform.smoothscale(surf, (size, size))
            screen.blit(surf, (cx - PLANET_SCREEN_RADIUS, cy - PLANET_SCREEN_RADIUS))

        hud = font.render(
            f"biome-ladder planet + ray-marched atmosphere (GLSL/SDL3) + pygame.draw, "
            f"same frame -- fps={clock.get_fps():4.1f}",
            True, (225, 230, 240))
        screen.blit(hud, (10, 10))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
