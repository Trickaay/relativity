# Relativity — SDL3 GPU compute + a pygame-embeddable shader DSL

A small, from-scratch compiler for a tiny Python-like kernel DSL, targeting SDL3's new GPU API (compute *and* graphics shaders), designed to run alongside pygame's existing 2D drawing in the same window and the same frame — proving a real shader/compute pipeline can coexist with pygame's legacy blit API without either replacing the other.

This repo is a focused, curated subset of a larger personal project (a physics-simulation engine) — just the pieces relevant to that one idea, so it's easy to read end to end.

## The problem

pygame has no access to modern GPU compute/shader pipelines today — everything goes through its legacy 2D surface/blit API. SDL3 (which pygame-ce is migrating to) introduces a real cross-platform GPU API with compute and graphics shaders, but there's currently no path for a pygame program to use it without dropping down to raw `ctypes`/SDL3 C-API calls by hand.

## The approach

`src/relativity_kernel_dsl` parses a small `.rfrk` DSL — Taichi-style ergonomics (`for i in range(n):` inside a `kernel` automatically becomes parallel GPU dispatch; array/"field" references never need a manual binding number in source) — compiles it to real GLSL, then through `glslangValidator` → `spirv-cross` → SDL3's GPU API (Metal on macOS; SDL3 also supports Vulkan/D3D12, not yet tested here).

It deliberately does **not** depend on Taichi (or any other heavy runtime) at runtime — keeping the dependency footprint small enough to be realistic for pygame-ce to actually take on. The entire subsystem needs only `numpy`, `pygame-ce`/`PySDL3`, and the two external CLI tools above.

## Two examples, smallest first

- **`examples/relativity_pygame_sdl3_compute_minimal.py`** — the core proof point in its simplest form: a real GPU compute dispatch (a parallel per-point sine-wave kernel, authored in the DSL) running every frame, its output read back and drawn with ordinary `pygame.draw.circle` calls, in the same window, same frame.
- **`examples/relativity_bvh_demo.py`** — a substantially more ambitious real workload: a Karras binary-radix-tree BVH build and primary-ray GPU traversal, entirely authored in the DSL, interactively camera-controlled.

A third example is included too, though it's a **separate achievement, not DSL-authored**:

- **`examples/relativity_pygame_planet_showcase_minimal.py`** — a procedural rocky planet (an altitude-banded biome surface) with a real ray-marched Rayleigh/Mie atmosphere layered on top, composited onto an ordinary pygame window every frame. `planet_surface.frag`/`atmosphere.frag` are hand-written GLSL, compiled through the same glslang → spirv-cross → SDL3/MSL toolchain, but independent of the `.rfrk` compiler. Included because it's a more visually concrete demonstration of the same underlying "GPU shaders + pygame coexist" story.

## Getting started

```
pip install -r requirements.txt
```

You'll also need `glslangValidator` and `spirv-cross` on `PATH` (both are commonly available via your platform's package manager, e.g. Homebrew on macOS: `brew install glslang spirv-cross`).

```
python examples/relativity_pygame_sdl3_compute_minimal.py
python examples/relativity_bvh_demo.py
python examples/relativity_pygame_planet_showcase_minimal.py
```

**Important**: like any SDL3-GPU-backed program, these need a real windowing-system connection and will not run under a headless/dummy SDL video driver.

Controls: left-drag to orbit the camera (BVH demo only), scroll to zoom, esc to quit.

## Honest limitations

- Only tested on macOS/Metal so far — not Vulkan or D3D12.
- The BVH demo's scope is capped at ≤256 primitives (a single-workgroup radix sort, a deliberate simplification, not a hard architectural limit).
- Primary-ray visibility only — no shadows, reflections, or multi-bounce GI yet.
- Needs a real windowing-system connection; no headless/CI support.

## Running tests

```
python -m pytest tests/
```

Most tests run under a dummy SDL driver automatically. Tests that touch a real SDL3 GPU device need:
```
SDL_VIDEODRIVER=cocoa python -m pytest tests/
```
(substitute the appropriate driver for your platform)

## More context

- [`docs/bug_log.md`](docs/bug_log.md) — real bugs found building this, written up in detail: what broke, why, how it was fixed, and why the fix works.
