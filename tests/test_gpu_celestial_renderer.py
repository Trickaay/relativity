"""Regression coverage for GPUCelestialRenderer / accretion_disk.frag,
star_corona.frag, atmosphere.frag. Same real-windowing-system caveat as
test_gpu_planet_renderer.py.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    reason="SDL3's Metal GPU backend needs a real windowing system connection, "
           "not available under the dummy driver conftest.py forces for the rest of the suite",
)


def _make_renderer():
    from relativity_visualization.gpu_celestial_renderer import GPUCelestialRenderer
    return GPUCelestialRenderer()


def test_accretion_disk_annulus_shape():
    renderer = _make_renderer()
    rgba = renderer.render_accretion_disk(inner_radius=0.3, outer_radius=0.9, temperature=8000.0)
    h, w = rgba.shape[:2]
    assert rgba[h // 2, w // 2, 3] == 0  # dead center: inside the inner radius, transparent (a hole)
    assert rgba[2, 2, 3] == 0            # corner: outside the outer radius, transparent
    assert rgba[h // 2, int(w * 0.5 + w * 0.3), 3] > 0  # between the radii: opaque disk


def test_accretion_disk_temperature_changes_color():
    renderer = _make_renderer()
    cool = renderer.render_accretion_disk(inner_radius=0.3, outer_radius=0.9, temperature=3000.0)
    hot = renderer.render_accretion_disk(inner_radius=0.3, outer_radius=0.9, temperature=15000.0)
    assert not np.array_equal(cool[..., :3], hot[..., :3])


def test_accretion_disk_doppler_asymmetry_brightens_one_side():
    renderer = _make_renderer()
    rgba = renderer.render_accretion_disk(inner_radius=0.3, outer_radius=0.9, temperature=8000.0,
                                           doppler=1.5, brightness=2.0)
    h, w = rgba.shape[:2]
    left = rgba[h // 2, : w // 3, :3].astype(np.float32).mean()
    right = rgba[h // 2, 2 * w // 3:, :3].astype(np.float32).mean()
    assert abs(left - right) > 5.0  # relativistic beaming should make one side visibly brighter


def test_star_corona_non_blank_and_varies_with_temperature():
    renderer = _make_renderer()
    cool = renderer.render_star_corona(temperature=3000.0, star_radius=0.3)
    hot = renderer.render_star_corona(temperature=30000.0, star_radius=0.3)
    assert cool.std() > 5.0
    h, w = cool.shape[:2]
    # away from the saturated core, where color (not just brightness) should differ
    assert not np.array_equal(cool[h // 2, w // 2 + 32, :3], hot[h // 2, w // 2 + 32, :3])


def test_star_corona_disk_shows_real_granulation_texture():
    """The photosphere should show real fbm brightness variation across
    its disk (convection-cell granulation), not the old flat
    length(uv)-only radial vignette -- catches a regression back to a
    perfectly smooth gradient, which a std-based check on the whole
    sprite (mixing disk + corona + transparent background) wouldn't
    reliably catch on its own."""
    renderer = _make_renderer()
    rgba = renderer.render_star_corona(temperature=5778.0, star_radius=0.3, seed=1.0)
    disk = rgba[:, :, 3] > 250  # near-fully-opaque disk interior, excludes the soft corona edge
    assert disk.sum() > 100
    luminance = rgba[:, :, :3][disk].astype(np.float32).mean(axis=1)
    assert luminance.std() > 5.0


def test_star_corona_rotation_phase_changes_granulation():
    renderer = _make_renderer()
    a = renderer.render_star_corona(temperature=5778.0, star_radius=0.3, rotation_phase=0.0, seed=1.0)
    b = renderer.render_star_corona(temperature=5778.0, star_radius=0.3, rotation_phase=2.0, seed=1.0)
    disk = a[:, :, 3] > 250
    diff = np.abs(a[:, :, :3][disk].astype(np.float32) - b[:, :, :3][disk].astype(np.float32))
    assert diff.mean() > 1.0


def test_atmosphere_renders_a_visible_shell():
    renderer = _make_renderer()
    rgba = renderer.render_atmosphere(sun_dir_local=(0.3, 0.2, -1.0),
                                       planet_radius=0.85, atmosphere_radius=1.0)
    assert rgba.std() > 1.0
    assert (rgba[..., 3] > 0).sum() > 0  # some pixels are part of the atmosphere shell


def test_atmosphere_brightens_toward_the_sun_facing_limb():
    """The real ray-marched in_scatter() integration should respond to
    sun direction (unlike a naive single-sample-per-pixel shell, which
    could plausibly look identical regardless of where the sun is) --
    rendering with two opposite sun directions must produce visibly
    different images, not just the same shell shape twice."""
    renderer = _make_renderer()
    lit_from_right = renderer.render_atmosphere(sun_dir_local=(1.0, 0.0, -0.3),
                                                 planet_radius=0.85, atmosphere_radius=1.0)
    lit_from_left = renderer.render_atmosphere(sun_dir_local=(-1.0, 0.0, -0.3),
                                                planet_radius=0.85, atmosphere_radius=1.0)
    diff = np.abs(lit_from_right[..., :3].astype(np.float32) - lit_from_left[..., :3].astype(np.float32))
    assert diff.mean() > 1.0


# A further test in the original project, test_stellar_corona_skips_cpu_mesh_and_produces_a_sprite,
# is omitted here: it exercises this repo's broader scene-graph/YAML
# pipeline (Scene, FrontendVisualizationLayer, run_scene.py), which is
# out of scope for this curated, DSL/GPU-focused subset -- see the main
# project for that integration-level coverage.
