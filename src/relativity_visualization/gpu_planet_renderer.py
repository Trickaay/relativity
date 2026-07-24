"""
Procedural planet-surface rendering via the SDL3 GPU bridge
(planet_surface.frag, see that file for the shading algorithm).

Renders each planet as a fixed 256x256 "sphere facing the camera"
sprite, independent of its actual on-screen size or position -- the
caller (run_scene.py) scales/positions the returned RGBA array to the
planet's real projected screen diameter each frame and hands it to
PygameRenderer.render(gpu_sprites=...) for depth-sorted compositing
against the rest of the scene. One GPUPlanetRenderer instance is reused
across every planet and every frame (same shader pipeline, different
uniforms per call) -- there is no per-planet GPU resource.
"""

import os
import struct

from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_SPRITE_SIZE = 256


class GPUPlanetRenderer:
    def __init__(self):
        self.renderer = SDL3GPUOffscreenRenderer(_SPRITE_SIZE, _SPRITE_SIZE)
        self.renderer.load_shader_pair(
            os.path.join(_SHADER_DIR, "fullscreen_triangle.msl"),
            os.path.join(_SHADER_DIR, "planet_surface.msl"),
            frag_uniform_buffers=1,
        )

    def render_planet(self, light_dir_local, rotation_phase, palette_low, palette_high,
                       kind="gas_giant", seed=0.0, turbulence=0.5, band_strength=1.0,
                       terrain_strength=0.0, earthlike=False, asteroid_mode=False):
        """light_dir_local: (x, y, z) light direction already expressed in
        this sprite's own local frame (see run_scene.py's camera.basis()
        projection -- the sprite is always rendered as a canonical sphere
        facing the camera, so the light direction must be transformed
        into that same local frame, not passed in world space).
        terrain_strength: 0 disables bump-mapped relief (flat shading,
        original behavior); >0 perturbs the shading normal by a
        procedural height gradient (see planet_surface.frag's heightAt).
        earthlike: gates the water/sand/vegetation/ice altitude bands in
        the rocky branch -- those bands are fixed Earth colors (blue
        ocean, green vegetation), wrong for a dry/rocky body like Mars or
        Mercury, which instead get a plain palette_low->palette_high
        altitude gradient (still keeping the latitude polar-ice mask,
        a real feature on Mars too).
        asteroid_mode: tri-planar procedural rock coloring for "hero"
        belt asteroids instead of the altitude ladder/gradient, and no
        polar ice mask -- see planet_surface.frag's asteroidRock(). Same
        cheap analytic-sphere intersection as every other planet, NOT
        the raymarched irregular-silhouette prototype (asteroid_rock.frag)
        -- see consolidation_particle_belts memory for why raymarching
        was ruled out for anything beyond a one-off preview.
        Returns a (256, 256, 4) uint8 RGBA array, alpha=0 outside the
        sphere's silhouette."""
        lx, ly, lz = light_dir_local
        plr, plg, plb = palette_low
        phr, phg, phb = palette_high
        kind_flag = 1.0 if kind == "gas_giant" else 0.0
        earthlike_flag = 1.0 if earthlike else 0.0
        asteroid_flag = 1.0 if asteroid_mode else 0.0

        uniform_bytes = struct.pack(
            "<17f",
            lx, ly, lz, rotation_phase,
            plr, plg, plb,
            phr, phg, phb,
            band_strength, turbulence, seed, kind_flag,
            terrain_strength, earthlike_flag, asteroid_flag,
        )
        return self.renderer.render_frame(fragment_uniform_bytes=uniform_bytes)
