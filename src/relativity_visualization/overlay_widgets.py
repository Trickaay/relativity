"""
Small 2D screen-space overlay helpers for scenes that want a starfield
backdrop or a live data-inset panel (e.g. a semi-major-axis histogram)
drawn on top of the 3D view -- plain pygame drawing, no dependency on
matplotlib or the 3D scene-graph pipeline.
"""

import numpy as np
import pygame


def make_starfield(width, height, n_stars=400, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, width, n_stars)
    ys = rng.uniform(0, height, n_stars)
    brightness = rng.uniform(80, 220, n_stars).astype(int)
    return list(zip(xs.astype(int), ys.astype(int), brightness))


def draw_starfield(surface, stars):
    for x, y, b in stars:
        surface.set_at((x, y), (b, b, b))


def _upsample_noise(coarse01, width, height):
    """Bilinear-upsample a small [0,1] 2D array to (height, width) via
    pygame's smoothscale -- avoids adding a scipy dependency just for
    this one interpolation. coarse01 is (rows, cols) numpy-natural
    (y, x); pygame surfaces are indexed (width, height), so transpose on
    the way in and out."""
    small_u8 = (np.clip(coarse01, 0.0, 1.0) * 255).astype(np.uint8)
    small_surf = pygame.surfarray.make_surface(np.repeat(small_u8.T[:, :, None], 3, axis=2))
    big_surf = pygame.transform.smoothscale(small_surf, (width, height))
    big = pygame.surfarray.array3d(big_surf)[:, :, 0].T.astype(np.float32) / 255.0
    return big


def make_nebula_starfield(width, height, seed=0):
    """Procedural static nebula + multi-layer starfield background,
    baked ONCE and cached as a pygame Surface by the caller (see
    PygameRenderer) -- this renderer's starfield is deliberately
    screen-space-fixed, not camera-rotation-aware (stars are meant to
    read as infinitely distant background noise, not a real skybox), so
    there's no reason to recompute this every frame or route it through
    the SDL3 GPU shader pipeline; a one-time numpy bake gets the same
    pixels for free.

    Conceptually adapted from a raymarched Shadertoy background
    technique (several star layers of different size/density/color for
    a sense of depth, plus a soft colored nebula wash) -- reimplemented
    directly in numpy rather than transliterating the reference's GLSL
    hash-grid math, since that math's only real purpose there was cheap
    per-pixel-shader star placement; for a one-time bake, ordinary
    per-star scatter-and-falloff (like make_starfield already used, just
    with soft radii and layered color/brightness instead of one flat
    single-pixel layer) gets the same visual result more simply."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Nebula wash: a few octaves of bilinearly-upsampled value noise
    # (a cheap value-noise fbm), blended between two dim tint colors and
    # kept faint (final contribution capped well under white) so it
    # reads as background wisps, not a wash that competes with the
    # actual scene drawn on top of it.
    nebula = np.zeros((height, width), dtype=np.float32)
    amp, total_amp = 1.0, 0.0
    for octave in range(4):
        cells = 2 ** (octave + 1)
        coarse = rng.uniform(0.0, 1.0, size=(cells + 1, cells + 1))
        nebula += _upsample_noise(coarse, width, height) * amp
        total_amp += amp
        amp *= 0.5
    nebula /= total_amp

    tint_a = np.array([0.10, 0.05, 0.18])  # deep purple
    tint_b = np.array([0.03, 0.07, 0.14])  # dim blue
    nebula_color = (tint_a[None, None, :] * nebula[:, :, None]
                     + tint_b[None, None, :] * (1.0 - nebula[:, :, None]))
    # Note: no extra multiply-by-`nebula` term here (an earlier version
    # had one) -- nebula_color already scales smoothly between the two
    # dim tints across the [0, 1] range, so multiplying by nebula again
    # only squares the falloff and made the whole wash too faint to
    # actually notice against the black background at normal brightness
    # (confirmed by rendering at 12x brightness first to check the cloud
    # shapes were real fbm, not noise, before concluding the *tuning*,
    # not the technique, was the problem).
    img += nebula_color * 0.9

    # Star layers: fewer/larger/brighter stars in the "near" layers, many
    # more small dim ones in the "far" layer -- suggests depth even
    # though it's a flat static texture. Each star gets a soft Gaussian
    # falloff (a few pixels) instead of a single hard-set pixel, and a
    # small per-star color jitter so the field isn't uniformly white.
    layer_specs = [
        dict(count=250, radius=0.6, brightness=(50, 120), jitter=0.05),
        dict(count=110, radius=1.0, brightness=(110, 190), jitter=0.10),
        dict(count=30, radius=1.7, brightness=(180, 255), jitter=0.18),
    ]
    yy, xx = np.mgrid[0:height, 0:width]
    for spec in layer_specs:
        n = spec["count"]
        xs = rng.uniform(0, width, n)
        ys = rng.uniform(0, height, n)
        brightness = rng.uniform(*spec["brightness"], n)
        jitter = rng.uniform(-1.0, 1.0, (n, 3)) * spec["jitter"]
        r = spec["radius"]
        for x, y, b, j in zip(xs, ys, brightness, jitter):
            x0, x1 = max(0, int(x - r * 3)), min(width, int(x + r * 3) + 1)
            y0, y1 = max(0, int(y - r * 3)), min(height, int(y + r * 3) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            d2 = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2
            falloff = np.exp(-d2 / (2 * r * r))
            color = np.clip(1.0 + j, 0.6, 1.3) * (b / 255.0)
            img[y0:y1, x0:x1, :] += falloff[:, :, None] * color[None, None, :]

    rgb = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


def draw_histogram(surface, font, values, bounds, rect, title="", bins=50,
                    bar_color=(140, 190, 230), bg_color=(30, 30, 45, 200)):
    """values: 1D array (NaNs / out-of-range ignored). rect: (x, y, w, h)
    pixel box in screen space."""
    x, y, w, h = rect
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill(bg_color)

    valid = values[~np.isnan(values)]
    valid = valid[(valid >= bounds[0]) & (valid <= bounds[1])]
    counts, _edges = np.histogram(valid, bins=bins, range=bounds)
    max_count = counts.max() if counts.max() > 0 else 1

    plot_h = h - 28
    bar_w = w / bins
    for i, c in enumerate(counts):
        bar_h = int((c / max_count) * (plot_h - 8))
        bx = int(i * bar_w)
        pygame.draw.rect(panel, bar_color, (bx, plot_h - bar_h, max(1, int(bar_w) - 1), bar_h))

    pygame.draw.line(panel, (120, 120, 140), (0, plot_h), (w, plot_h), 1)

    if title:
        label = font.render(title, True, (210, 215, 230))
        panel.blit(label, (4, plot_h + 6))

    lo_label = font.render(f"{bounds[0]:.1f} AU", True, (170, 175, 195))
    hi_label = font.render(f"{bounds[1]:.1f} AU", True, (170, 175, 195))
    panel.blit(lo_label, (2, 2))
    panel.blit(hi_label, (w - hi_label.get_width() - 2, 2))

    surface.blit(panel, (x, y))
