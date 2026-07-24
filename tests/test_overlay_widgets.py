"""Regression coverage for overlay_widgets.py's make_nebula_starfield --
pure pygame/numpy, no GPU device needed, runs under the normal dummy
driver (see conftest.py)."""

import numpy as np
import pygame

from relativity_visualization.overlay_widgets import make_nebula_starfield


def test_make_nebula_starfield_returns_correctly_sized_surface():
    pygame.init()
    surf = make_nebula_starfield(200, 150, seed=0)
    assert surf.get_width() == 200
    assert surf.get_height() == 150


def test_nebula_starfield_has_real_content_not_flat_black():
    pygame.init()
    surf = make_nebula_starfield(200, 150, seed=0)
    arr = pygame.surfarray.array3d(surf).astype(np.float32)
    assert arr.std() > 1.0
    assert arr.max() > 100  # bright star cores should be present


def test_nebula_starfield_is_deterministic_given_a_seed():
    pygame.init()
    a = pygame.surfarray.array3d(make_nebula_starfield(200, 150, seed=42))
    b = pygame.surfarray.array3d(make_nebula_starfield(200, 150, seed=42))
    assert np.array_equal(a, b)


def test_nebula_starfield_differs_between_seeds():
    pygame.init()
    a = pygame.surfarray.array3d(make_nebula_starfield(200, 150, seed=1))
    b = pygame.surfarray.array3d(make_nebula_starfield(200, 150, seed=2))
    assert not np.array_equal(a, b)


def test_nebula_wash_covers_the_whole_background_not_just_star_points():
    """The dim colored nebula wash should leave essentially no pixel at
    pure black, since it's a continuous full-frame tint underneath the
    sparse star points -- catches the earlier tuning bug where the wash
    was computed correctly but multiplied down to the point of being
    indistinguishable from black at normal brightness (verified then by
    checking min pixel brightness was near zero across the whole image,
    not just that *some* pixels were nonzero)."""
    pygame.init()
    surf = make_nebula_starfield(300, 300, seed=0)
    arr = pygame.surfarray.array3d(surf).astype(np.float32)
    channel_sum = arr.sum(axis=2)
    assert (channel_sum > 0).mean() > 0.999  # essentially the entire frame has some tint
    assert channel_sum.min() > 10  # and it's not just a handful of stray bright pixels
