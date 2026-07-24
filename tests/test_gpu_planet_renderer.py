"""Regression coverage for GPUPlanetRenderer / planet_surface.frag. Same
real-windowing-system caveat as test_sdl3_gpu_bridge.py / test_gpu_fluid_pipeline.py.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    reason="SDL3's Metal GPU backend needs a real windowing system connection, "
           "not available under the dummy driver conftest.py forces for the rest of the suite",
)

PALETTE_LOW = (0.2, 0.2, 0.3)
PALETTE_HIGH = (0.9, 0.8, 0.6)


def _make_renderer():
    from relativity_visualization.gpu_planet_renderer import GPUPlanetRenderer
    return GPUPlanetRenderer()


def test_sphere_silhouette_opaque_center_transparent_corners():
    renderer = _make_renderer()
    rgba = renderer.render_planet(
        light_dir_local=(0.3, 0.5, -1.0), rotation_phase=0.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH, kind="gas_giant", seed=0.0)

    h, w = rgba.shape[:2]
    assert rgba[h // 2, w // 2, 3] == 255  # center: inside the sphere's silhouette
    assert rgba[2, 2, 3] == 0  # corner: outside it -- must be transparent to composite correctly
    assert rgba[2, w - 3, 3] == 0
    assert rgba[h - 3, 2, 3] == 0


def test_gas_giant_shows_latitude_banding_rocky_does_not():
    renderer = _make_renderer()
    gas_giant = renderer.render_planet(
        light_dir_local=(0.0, 0.0, -1.0), rotation_phase=0.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH,
        kind="gas_giant", seed=0.0, turbulence=0.3, band_strength=1.5)
    rocky = renderer.render_planet(
        light_dir_local=(0.0, 0.0, -1.0), rotation_phase=0.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH,
        kind="rocky", seed=0.0, turbulence=0.3)

    def row_variance(rgba):
        h, w = rgba.shape[:2]
        opaque = rgba[:, :, 3] > 0
        row_means = np.array([
            rgba[y, opaque[y], :3].astype(np.float32).mean() if opaque[y].any() else np.nan
            for y in range(h)
        ])
        return np.nanvar(row_means)

    # Banding is a real row-to-row (latitude) color pattern; rocky's
    # blotchy noise has no such structure -- the banded shader should
    # show meaningfully more row-to-row variance.
    assert row_variance(gas_giant) > row_variance(rocky) * 1.5


def test_rotation_phase_changes_output():
    renderer = _make_renderer()
    frame_a = renderer.render_planet(
        light_dir_local=(0.3, 0.3, -1.0), rotation_phase=0.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH, kind="gas_giant", seed=0.0)
    frame_b = renderer.render_planet(
        light_dir_local=(0.3, 0.3, -1.0), rotation_phase=2.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH, kind="gas_giant", seed=0.0)

    assert not np.array_equal(frame_a, frame_b)


def test_rocky_planet_shows_polar_ice_and_more_color_variety_than_flat_blend():
    """The altitude-banded biome ladder (water/sand/vegetation/rock/ice,
    see planet_surface.frag) should produce noticeably more distinct
    colors across a rocky planet's surface than a flat 2-color blend
    would, AND the poles should end up measurably whiter/brighter than
    the equator (the latitude ice-cap mask layered on top of the
    altitude ladder)."""
    renderer = _make_renderer()
    rgba = renderer.render_planet(
        light_dir_local=(0.3, 0.3, -1.0), rotation_phase=0.0,
        palette_low=(0.05, 0.1, 0.3), palette_high=(0.35, 0.3, 0.25),
        kind="rocky", seed=2.0, turbulence=0.4, earthlike=True)

    opaque = rgba[:, :, 3] > 0
    colors = rgba[:, :, :3][opaque]
    # quantize to 8-levels-per-channel buckets so noise dither doesn't
    # inflate the count -- a flat 2-color blend would only ever produce
    # colors along one line between low/high, landing in a handful of
    # buckets; a real multi-band biome ladder spans many more.
    buckets = set(map(tuple, (colors // 24 * 24).astype(int)))
    assert len(buckets) > 15

    h, w = rgba.shape[:2]
    col = w // 2
    opaque_rows = np.where(rgba[:, col, 3] > 0)[0]
    top, bottom = opaque_rows.min(), opaque_rows.max()
    pole_region = rgba[top + 2:top + 8, col, :3].astype(np.float32)
    equator = rgba[h // 2, col, :3].astype(np.float32)
    assert pole_region.mean() > equator.mean()


def test_asteroid_mode_has_no_polar_ice_and_real_rock_color_variety():
    """asteroid_mode should show genuine tri-planar rock color variety
    (not a flat 2-color gradient) but must NOT show the earthlike/
    non-earthlike polar ice-cap mask -- real belt asteroids don't have
    polar ice, and the whole point of asteroid_mode is a plain rock look."""
    renderer = _make_renderer()
    rgba = renderer.render_planet(
        light_dir_local=(0.3, 0.3, -1.0), rotation_phase=0.0,
        palette_low=(0.15, 0.13, 0.11), palette_high=(0.55, 0.5, 0.45),
        kind="rocky", seed=5.0, turbulence=0.5, asteroid_mode=True)

    opaque = rgba[:, :, 3] > 0
    colors = rgba[:, :, :3][opaque]
    buckets = set(map(tuple, (colors // 24 * 24).astype(int)))
    assert len(buckets) > 10  # real rock variety, not a flat blend

    h, w = rgba.shape[:2]
    col = w // 2
    opaque_rows = np.where(rgba[:, col, 3] > 0)[0]
    top, bottom = opaque_rows.min(), opaque_rows.max()
    pole_region = rgba[top + 2:top + 8, col, :3].astype(np.float32)
    equator = rgba[h // 2, col, :3].astype(np.float32)
    # no ice-white pole spike: poles shouldn't be dramatically brighter
    # than the equator the way the ice-cap mask makes earthlike/rocky
    # planets -- a loose bound (mean brightness ratio), not requiring
    # them to be equal, since lighting/normal alone still varies some.
    assert pole_region.mean() < equator.mean() * 1.8


def test_asteroid_mode_differs_from_plain_altitude_gradient():
    """asteroid_mode's tri-planar coloring should produce a genuinely
    different result than the plain non-earthlike altitude gradient at
    the same seed/palette -- confirms the shader actually branches, not
    silently falling through to the old rocky path."""
    renderer = _make_renderer()
    common = dict(light_dir_local=(0.3, 0.3, -1.0), rotation_phase=0.0,
                  palette_low=(0.15, 0.13, 0.11), palette_high=(0.55, 0.5, 0.45),
                  kind="rocky", seed=5.0, turbulence=0.5)
    plain = renderer.render_planet(**common, asteroid_mode=False)
    asteroid = renderer.render_planet(**common, asteroid_mode=True)
    assert not np.array_equal(plain, asteroid)


def test_non_earthlike_rocky_planet_has_no_earth_specific_bands():
    """A Mars/Mercury-style dry rocky body (earthlike defaults to False)
    must not get the earthlike ladder's hardcoded blue-water/green-
    vegetation bands -- it should stay within its own palette_low/high
    gradient instead."""
    renderer = _make_renderer()
    palette_low = (0.4, 0.18, 0.1)
    palette_high = (0.85, 0.5, 0.3)
    rgba = renderer.render_planet(
        light_dir_local=(0.3, 0.3, -1.0), rotation_phase=0.0,
        palette_low=palette_low, palette_high=palette_high,
        kind="rocky", seed=2.0, turbulence=0.4)

    opaque = rgba[:, :, 3] > 0
    colors = rgba[:, :, :3][opaque].astype(np.float32)
    # the earthlike ladder's vegetation band (0.15, 0.35, 0.12) is green
    # (G channel clearly dominant) -- this dry palette has no green
    # anywhere in low/high/ice/rim-light, so no pixel should end up
    # green-dominant either.
    green_dominant = (colors[:, 1] > colors[:, 0] + 15) & (colors[:, 1] > colors[:, 2] + 15)
    assert not green_dominant.any()


def test_terrain_strength_changes_shading():
    renderer = _make_renderer()
    flat = renderer.render_planet(
        light_dir_local=(0.3, 0.5, -1.0), rotation_phase=0.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH,
        kind="rocky", seed=4.0, turbulence=0.5, terrain_strength=0.0)
    bumped = renderer.render_planet(
        light_dir_local=(0.3, 0.5, -1.0), rotation_phase=0.0,
        palette_low=PALETTE_LOW, palette_high=PALETTE_HIGH,
        kind="rocky", seed=4.0, turbulence=0.5, terrain_strength=0.8)

    # Bump-mapping perturbs the lighting normal, not just color banding --
    # a real, if small, per-pixel shading difference should appear even
    # though both renders share the same seed/palette/light direction.
    diff = np.abs(flat.astype(np.int16) - bumped.astype(np.int16))
    assert diff.mean() > 1.0
