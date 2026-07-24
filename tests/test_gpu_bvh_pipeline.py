"""Regression coverage for GPUBVHPipeline (relativity_kernel_dsl/gpu_bvh_pipeline.py):
a real Karras binary-radix-tree BVH build + primary-ray traversal, authored
in the Relativity kernel DSL and compiled through glslang -> spirv-cross -> SDL3
GPU/MSL. Fixes two real bugs an audit of the dormant GLSL BVH library found
(a degenerate `parent=i-1` build instead of a real Karras split-search, and
a radix sort that corrupted its own scatter-rank counter) -- see
bvh_pipeline.rfrk's header for the full context.

Needs a real SDL3 GPU device (same caveat as test_sdl3_gpu_bridge.py/
test_gpu_fluid_pipeline.py/test_kernel_dsl.py -- unavailable under the
dummy driver conftest.py forces for the rest of the suite) plus the
glslang/spirv-cross CLI tools, so this self-skips under the dummy driver.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SDL_VIDEODRIVER") == "dummy",
    reason="SDL3's Metal GPU backend needs a real windowing system connection, "
           "not available under the dummy driver conftest.py forces for the rest of the suite",
)


@pytest.fixture
def make_pipeline():
    """Factory fixture: each pipeline it creates is closed on teardown.
    Not closing SDL3GPUOffscreenRenderer instances between short-lived
    uses within the same process was confirmed to cause real
    cross-instance interference -- a later pipeline's compute dispatches
    produced a subtly wrong (but not crashing) result while an earlier
    instance's device/resources were still alive. See
    SDL3GPUOffscreenRenderer.close()'s docstring."""
    from relativity_kernel_dsl.gpu_bvh_pipeline import GPUBVHPipeline

    created = []

    def _make(n=64, seed=42, render_width=160, render_height=120):
        rng = np.random.default_rng(seed)
        centers = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
        radii = rng.uniform(0.2, 0.6, n).astype(np.float32)

        pipeline = GPUBVHPipeline(n, render_width=render_width, render_height=render_height)
        created.append(pipeline)
        pipeline.set_spheres(centers, radii)
        pipeline.build()
        return pipeline, centers, radii

    yield _make
    for p in created:
        p.close()


def test_leaf_bounds_match_primitive_geometry(make_pipeline):
    pipeline, centers, radii = make_pipeline()
    bmin, bmax, left, right, escape = pipeline.download_nodes()

    for i in range(pipeline.n):
        assert np.allclose(bmin[i], centers[i] - radii[i], atol=1e-4)
        assert np.allclose(bmax[i], centers[i] + radii[i], atol=1e-4)


def test_internal_bounds_enclose_children_recursively(make_pipeline):
    pipeline, centers, radii = make_pipeline()
    bmin, bmax, left, right, escape = pipeline.download_nodes()

    for i in range(pipeline.n, pipeline.n2):
        l, r = left[i], right[i]
        assert np.all(bmin[i] <= bmin[l] + 1e-4)
        assert np.all(bmin[i] <= bmin[r] + 1e-4)
        assert np.all(bmax[i] >= bmax[l] - 1e-4)
        assert np.all(bmax[i] >= bmax[r] - 1e-4)


def test_every_primitive_reachable_exactly_once_via_escape_chain(make_pipeline):
    """This is the check that would have caught the original library's
    real bug: a leaf-specific `escape[i] = i+1` formula that only makes
    sense for a depth-first-preorder node layout, not this pipeline's
    leaves-then-internals layout -- it produced a non-terminating,
    duplicate-revisiting walk before the fix (escape values coming
    purely from parent-to-child propagation, root seeded to -1)."""
    pipeline, centers, radii = make_pipeline()
    bmin, bmax, left, right, escape = pipeline.download_nodes()

    n, n2 = pipeline.n, pipeline.n2
    visited = []
    idx = n  # root: internal node 0's global index is always exactly n
    for _ in range(n2 + 1):
        if idx < 0:
            break
        if left[idx] < 0:
            visited.append(int(idx))
            idx = escape[idx]
        else:
            idx = left[idx]

    assert idx == -1, "traversal must terminate at the root's seeded escape=-1"
    assert len(visited) == n
    assert len(set(visited)) == n
    assert set(visited) == set(range(n))


def test_scene_bounds_fold_matches_numpy_min_max(make_pipeline):
    pipeline, centers, radii = make_pipeline()
    expected_min = np.min(centers - radii[:, None], axis=0)
    expected_max = np.max(centers + radii[:, None], axis=0)
    assert np.allclose(pipeline.scene_min, expected_min, atol=1e-3)
    assert np.allclose(pipeline.scene_max, expected_max, atol=1e-3)


def test_render_produces_distinct_visible_spheres_at_expected_screen_positions(make_pipeline):
    pipeline, centers, radii = make_pipeline(n=8, render_width=160, render_height=120)

    cam_pos = np.array([0.0, 0.0, 20.0], dtype=np.float32)
    cam_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    pixels = pipeline.render(cam_pos, cam_forward, cam_right, cam_up, fov_y_degrees=40.0)

    assert pixels.shape == (120, 160, 4)

    background = np.array([0.02, 0.02, 0.05]) * 255.0
    rgb = pixels[:, :, :3].astype(np.float32)
    is_sphere = np.linalg.norm(rgb - background, axis=-1) > 15.0
    assert is_sphere.sum() > 0, "render produced no visible sphere pixels at all"

    # Project each sphere's world-space center to screen space (simple pinhole
    # projection matching the shader's own camera basis) and confirm a
    # sphere-colored pixel exists near that projected location -- proves
    # positions genuinely round-trip through build+traverse+shade, not just
    # "something non-blank appeared somewhere."
    width, height = 160, 120
    aspect = width / height
    import math
    tan_half_fov = math.tan(math.radians(40.0) / 2.0)
    hits_near_projection = 0
    for c in centers:
        rel = c - cam_pos
        depth = np.dot(rel, -cam_forward) if np.dot(rel, cam_forward) < 0 else np.dot(rel, cam_forward)
        forward_dist = np.dot(rel, cam_forward)
        if forward_dist <= 0:
            continue  # behind the camera, not expected to be visible
        px = np.dot(rel, cam_right) / (forward_dist * aspect * tan_half_fov)
        py = np.dot(rel, cam_up) / (forward_dist * tan_half_fov)
        screen_x = int((px * 0.5 + 0.5) * width)
        screen_y = int((1.0 - (py * 0.5 + 0.5)) * height)
        if not (0 <= screen_x < width and 0 <= screen_y < height):
            continue
        y0, y1 = max(0, screen_y - 6), min(height, screen_y + 7)
        x0, x1 = max(0, screen_x - 6), min(width, screen_x + 7)
        if is_sphere[y0:y1, x0:x1].any():
            hits_near_projection += 1

    assert hits_near_projection >= len(centers) // 2, (
        f"expected most spheres' projected positions to land near a rendered sphere pixel, "
        f"got {hits_near_projection}/{len(centers)}")
