"""
Shared pytest setup for this repo's regression suite.

- SDL is forced onto the dummy video/audio drivers *before* pygame gets
  imported anywhere, so most of the suite runs headless (no window pops
  up, works over SSH/CI with no display). A subset of tests touch a
  real SDL3 GPU device and are skipped automatically under the dummy
  driver (see each test file's own module docstring) -- run those with
  SDL_VIDEODRIVER=cocoa (macOS/Metal) instead.
- src/ and examples/ are put on sys.path once, here, since this repo
  has no installed package/pyproject.toml -- everything is run directly
  against source.

Run with: python -m pytest tests/
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

for p in (SRC_DIR, EXAMPLES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
