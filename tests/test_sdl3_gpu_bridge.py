"""Regression coverage for SDL3GPUOffscreenRenderer -- the thin ctypes
bridge to SDL3's GPU API every renderer in this repo (GPUPlanetRenderer,
GPUCelestialRenderer, GPUBVHPipeline) is built on top of.

IMPORTANT CAVEAT, different from every other test in this suite: this
CANNOT run under SDL_VIDEODRIVER=dummy. SDL3's Metal GPU backend
genuinely needs a real windowing system connection to create a device
-- confirmed empirically (device creation raises "No supported SDL_GPU
backend found!" under the dummy driver, works under the real one).
conftest.py forces the dummy driver for the whole suite (so every other
demo/test runs headless), which means these tests will FAIL in that
environment -- they're skipped automatically via the check below rather
than given a false failure signal. Run this file directly with a real
display attached:
    SDL_VIDEODRIVER=cocoa python -m pytest tests/test_sdl3_gpu_bridge.py
(substitute the appropriate driver for your platform; won't work if
conftest.py's dummy-driver env vars are already set process-wide before
this file's collection -- see the skip condition below).

Uses planet_surface.msl (already present, exercised more thoroughly by
test_gpu_planet_renderer.py) as a real, non-trivial fragment shader for
these bridge-level checks, rather than introducing a separate shader
just for this file.
"""

import os
import struct

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    reason="SDL3's Metal GPU backend needs a real windowing system connection, "
           "not available under the dummy driver conftest.py forces for the rest of the suite",
)

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "src", "relativity_visualization", "shaders")


def _planet_uniform_bytes(rotation_phase=0.0):
    # Matches planet_surface.frag's UBO layout (see gpu_planet_renderer.py) --
    # light dir, rotation phase, low/high palette, band/turbulence/seed,
    # kind flag, terrain strength, earthlike flag, asteroid flag.
    return struct.pack(
        "<17f",
        0.4, 0.5, -1.0, rotation_phase,
        0.3, 0.3, 0.35,
        0.9, 0.85, 0.7,
        1.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )


def test_bridge_renders_varying_non_blank_pixels():
    from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

    renderer = SDL3GPUOffscreenRenderer(320, 240)
    renderer.load_shader_pair(
        os.path.join(SHADER_DIR, "fullscreen_triangle.msl"),
        os.path.join(SHADER_DIR, "planet_surface.msl"),
        frag_uniform_buffers=1,
    )

    pixels = renderer.render_frame(fragment_uniform_bytes=_planet_uniform_bytes())
    assert pixels.shape == (240, 320, 4)
    assert pixels.dtype == np.uint8
    assert pixels.std() > 10  # a real shaded sphere, not a blank/uniform frame


def test_bridge_output_changes_with_time_uniform():
    from relativity_visualization.sdl3_gpu_bridge import SDL3GPUOffscreenRenderer

    renderer = SDL3GPUOffscreenRenderer(320, 240)
    renderer.load_shader_pair(
        os.path.join(SHADER_DIR, "fullscreen_triangle.msl"),
        os.path.join(SHADER_DIR, "planet_surface.msl"),
        frag_uniform_buffers=1,
    )

    frame_a = renderer.render_frame(fragment_uniform_bytes=_planet_uniform_bytes(rotation_phase=0.0))
    frame_b = renderer.render_frame(fragment_uniform_bytes=_planet_uniform_bytes(rotation_phase=2.0))
    assert not np.array_equal(frame_a, frame_b)  # rotation phase genuinely animates the surface
