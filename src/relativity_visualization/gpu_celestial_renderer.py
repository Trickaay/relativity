"""
GPU-shader rendering for accretion disks, star coronas, and planetary
atmospheres -- three more sprites in the same "fixed 256x256 sphere/disk
facing the camera" family as gpu_planet_renderer.py's GPUPlanetRenderer,
composited via PygameRenderer's gpu_sprites depth-sort path the same way.

Shaders adapted from salvaged/Relativity/src/relativity_rendering/shaders/ -- see each
.frag file's header comment for what was kept as-is vs repaired vs
rescoped from a 3D-mesh varying to this project's 2D sprite convention.
star_corona.frag and accretion_disk.frag both take a doppler/brightness
pair sourced from RelativisticCamera.get_shader_constants() (1.0/1.0 if
no relative camera motion -- a no-op multiply, safe default).
"""

import os
import struct

from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_SPRITE_SIZE = 256


class GPUCelestialRenderer:
    def __init__(self):
        self._accretion = SDL3GPUOffscreenRenderer(_SPRITE_SIZE, _SPRITE_SIZE)
        self._accretion.load_shader_pair(
            os.path.join(_SHADER_DIR, "fullscreen_triangle.msl"),
            os.path.join(_SHADER_DIR, "accretion_disk.msl"),
            frag_uniform_buffers=1,
        )

        self._corona = SDL3GPUOffscreenRenderer(_SPRITE_SIZE, _SPRITE_SIZE)
        self._corona.load_shader_pair(
            os.path.join(_SHADER_DIR, "fullscreen_triangle.msl"),
            os.path.join(_SHADER_DIR, "star_corona.msl"),
            frag_uniform_buffers=1,
        )

        self._atmosphere = SDL3GPUOffscreenRenderer(_SPRITE_SIZE, _SPRITE_SIZE)
        self._atmosphere.load_shader_pair(
            os.path.join(_SHADER_DIR, "fullscreen_triangle.msl"),
            os.path.join(_SHADER_DIR, "atmosphere.msl"),
            frag_uniform_buffers=1,
        )

    def render_accretion_disk(self, inner_radius, outer_radius, temperature,
                               doppler=1.0, brightness=1.0):
        """inner_radius/outer_radius: sprite-space radii (0..~1.4, since
        v_uv*2-1 spans -1..1 with sqrt(2) at the sprite corners)."""
        uniform_bytes = struct.pack("<5f", inner_radius, outer_radius, temperature, doppler, brightness)
        return self._accretion.render_frame(fragment_uniform_bytes=uniform_bytes)

    def render_star_corona(self, temperature, star_radius=0.3, doppler=1.0, brightness=1.0,
                            rotation_phase=0.0, seed=0.0):
        uniform_bytes = struct.pack("<6f", temperature, star_radius, doppler, brightness,
                                     rotation_phase, seed)
        return self._corona.render_frame(fragment_uniform_bytes=uniform_bytes)

    def render_atmosphere(self, sun_dir_local, planet_radius=0.85, atmosphere_radius=1.0):
        sx, sy, sz = sun_dir_local
        uniform_bytes = struct.pack("<5f", sx, sy, sz, planet_radius, atmosphere_radius)
        return self._atmosphere.render_frame(fragment_uniform_bytes=uniform_bytes)
