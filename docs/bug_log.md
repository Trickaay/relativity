# Bug log

A running record of real bugs found in this project: what broke, the
relevant code, why it happened, how it was fixed, and — the actual
point of this file — *why* the fix works, written so it's useful for
learning, not just a changelog.

Newest entries at the top. Each entry follows the same shape:

- **Where**: file(s)/function(s) involved
- **Symptom**: what was actually observed (a screenshot artifact, a
  wrong number, a crash, a performance complaint)
- **The code**: the relevant snippet, with enough surrounding context
  to follow without the rest of the codebase
- **Root cause**: the actual mechanism — not just "X was wrong" but
  *why* it was wrong
- **The fix**: what changed
- **Why the fix works**: the reasoning connecting cause to fix
- **Lesson**: the general, reusable takeaway

---

## Asteroid belt: 4 color buckets caused a ~3x FPS drop

**Where**: `examples/run_scene.py`'s `render_particle_belts()` /
`_asteroid_color_buckets()` (now removed), `src/relativity_visualization/trail_accumulator.py`

**Symptom**: user reported the solar-system demo ran at very low FPS.
Measured directly (not guessed): 18.6 fps (53.7ms/frame).

**The code** (the part that mattered): a `TrailAccumulator` is a
persistent "glow cloud" — a 2D array the size of the screen that fades
slightly every frame and has new particle positions added into it.
Turning that density into an actual image each frame is `blit_onto()`:

```python
def blit_onto(self, target_surface, color=(190, 205, 255), gain=0.01):
    mapped = 1.0 - np.exp(-self.density * gain)
    rgb = (mapped[:, :, None] * 255.0 * (np.array(color) / 255.0)).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb)
    target_surface.blit(surf, (0, 0), special_flags=pygame.BLEND_ADD)
```

`self.density` is a `(width, height)` array — for a 1100×750 window
that's 825,000 numbers. Every one of those four lines above touches
*all* of them: an `exp()` call per pixel, a multiply, a cast to
8-bit-per-channel color, then a full-screen composite.

A previous change (asteroid color variety) had split one belt into 4
separate `TrailAccumulator`s — one per rock color — so `blit_onto()`
above ran 4 times per frame instead of once.

**Root cause**: `blit_onto()` was never expensive-*looking* — it's four
short lines. But "short" and "cheap" aren't the same thing when every
line is a whole-screen numpy operation. Profiled directly (not
assumed): one call costs ~6-7ms. Four calls, every single frame,
costs ~28ms — more than the entire rest of the render pipeline
(planets, atmosphere, sun, bloom, everything) *combined*. The color
-variety feature had been checked for *correctness* (right colors, right
particle counts, a full test suite) but never checked for
*performance* before shipping — those are genuinely separate questions,
and this feature only failed the second one.

I first suspected `pygame.surfarray.make_surface()` (allocates a brand
new `Surface` every call) was the expensive part, by analogy with an
earlier, similar bloom-rendering fix in this same project. Tried
replacing it with a persistent `Surface` written into via
`pixels3d()` (a direct view, no new allocation). **This made no
measurable difference.** Breaking `blit_onto()` down line-by-line
showed the *actual* cost was the `exp()`/multiply/cast math itself
(~6ms), not the surface allocation (~0.1ms) — a different bottleneck
than the one I guessed from a superficially similar past bug.

**The fix**: reverted the 4-accumulator color-variety feature back to
one `TrailAccumulator` per belt with one flat color — the design from
before that feature existed. 53.7ms/frame → 36.2ms/frame (18.6 fps →
27.7 fps).

**Why the fix works**: it's arithmetic, not a clever optimization — if
one full-screen pass costs ~7ms, four of them cost ~28ms, because
there's no shared work between them to amortize (each accumulator has
its own density array, entirely disjoint from the others). Removing
3 of the 4 passes removes 3/4 of that cost, full stop.

**Lesson**: when a change *multiplies* an existing per-frame operation
by a small constant factor (here: 1 → 4 full-screen passes), profile it
before shipping, even if each individual piece "looks small" in the
source code. And when investigating a slowdown, don't assume the fix
for a *similar-looking* past bug (surface allocation, in this case)
applies again — measure the actual line that's slow, because the
resemblance can be misleading.

---

## Atmosphere sprite was rendering, but invisible, due to a scale mismatch

**Where**: `examples/run_scene.py`'s `render_atmosphere_layers()`,
`src/relativity_visualization/shaders/atmosphere.frag` /
`planet_surface.frag`

**Symptom**: real ray-marched atmosphere shader, verified correct in
isolation (its own tests passed, screenshots of the shader alone showed
a proper glow) — but wiring it into the actual solar-system scene
produced *no visible change at all*. A composited screenshot showed a
perfectly hard edge around Earth with zero blended pixels.

**The code**:

```python
rgba = gpu_celestial_renderer.render_atmosphere(
    sun_dir_local=light_local,
    planet_radius=0.85,
    atmosphere_radius=float(atmo.get("atmosphere_radius", 1.05)),
)
```

`planet_radius=0.85` was copied from an earlier *standalone* test of
the atmosphere shader — a value that had nothing to do with how the
real planet is drawn. The actual planet sprite (`planet_surface.frag`)
uses a hardcoded analytic sphere of radius **1.0**, in the exact same
"virtual camera" coordinate space both shaders share (`ray_origin =
(0,0,-3)`, same field-of-view math).

**Root cause**: both shaders trace a ray from a fixed virtual camera
through a sphere of some radius, and figure out how big that sphere
looks on screen from that. If you tell the atmosphere shader "the
planet's radius is 0.85" when the real planet sprite is actually radius
1.0, the atmosphere shader draws its glow around a sphere *smaller*
than the real planet — so the "glow" ends up entirely *inside* the
opaque planet disk, completely hidden behind it. Nothing was broken in
either shader; they just disagreed about how big the planet was.

**The fix**: `planet_radius=1.0` (matching `planet_surface.frag`'s real
hardcoded value), `atmosphere_radius` left as the per-planet YAML value
(e.g. `1.05`, meaning "5% bigger than the planet").

**Why the fix works**: once both shaders agree the planet is radius
1.0, the atmosphere's outer edge (radius 1.05) is genuinely *outside*
the planet's own edge (radius 1.0) — so there's real, uncovered space
for the glow to show through, instead of being swallowed by the opaque
disk drawn on top of it.

**Lesson**: for two shaders/sprites meant to composite together at the
same size, always sample actual pixel values across the seam and
compare, don't just trust "the shader compiled and one screenshot
looked plausible" — a scale-mismatch bug like this can look *exactly*
like "the effect is just subtle," which is a very different (and much
less alarming) explanation than "it's not rendering at all."

---

## Mars (and other dry planets) were rendering with Earth's oceans and forests

**Where**: `src/relativity_visualization/shaders/planet_surface.frag`

**Symptom**: after building a nice altitude-based biome ladder for
Earth (water → sand → vegetation → rock → ice), Mars — a different
planet using the *same* shader — showed the same green vegetation and
blue water patches. Wrong: Mars is dry and red.

**The code**: the biome ladder used fixed, hardcoded colors for the
middle bands:

```glsl
vec3 sand = vec3(0.82, 0.75, 0.55);
vec3 vegetation = vec3(0.15, 0.35, 0.12);
...
surface_color = mix(surface_color, vegetation, smoothstep(SAND_LEVEL, SAND_LEVEL + TRANS, alt));
```

**Root cause**: the shader is shared by every rocky planet, but
`vegetation`/`sand`/`water_surface` were plain constants baked into the
GLSL — they didn't depend on which *planet* was being rendered at all.
The per-planet customization (`palette_low`/`palette_high`) only
controlled the water/rock *endpoints*, not the fixed middle bands. So
every rocky planet got literal grass-green vegetation, whether or not
that made any sense for it.

**The fix**: added a new `earthlike` boolean uniform. When true (only
set for Earth), use the full water/sand/vegetation/rock/ice ladder.
When false (Mars, Mercury, everything else by default), skip straight
to a plain two-color gradient between the planet's own
`palette_low`/`palette_high`.

**Why the fix works**: the bug wasn't really about color *values* — it
was about a shared shader silently assuming every planet using it
*is* Earth. Adding an explicit flag makes that assumption a real,
visible choice per-planet instead of a hidden default, so Mars can
opt out of Earth-specific content entirely rather than getting a
watered-down version of it.

**Lesson**: when one shader/function serves multiple conceptually
different "kinds" of thing (here: an Earth-like world vs. a dry rocky
world), watch for constants that quietly assume it's always the *first*
kind you had in mind when you wrote it. A parameter that only adjusts
values (palette endpoints) doesn't help if the actual *content*
(vegetation existing at all) isn't optional.

---

## Radix sort looked like cross-instance GPU state corruption, but was a stability bug

**Where**: `src/relativity_kernel_dsl/kernels/bvh_pipeline.rfrk`'s
`radix_sort_pass`

**Symptom**: a from-scratch GPU BVH pipeline gave correct results for
the *first* `GPUBVHPipeline` created in a process, but wrong results
for a second one — deterministically, every time. Looked exactly like
some kind of leftover GPU device/resource state leaking between
instances.

**The code** (the shape of the bug, simplified): a parallel radix sort
needs to compute, for each element, *where it goes* in the sorted
output. That "where" was computed with `atomicAdd` — each GPU thread
atomically increments a shared counter and uses the result as its
output position:

```glsl
uint rank = atomicAdd(hist[bucket], 1u);
```

**Root cause**: `atomicAdd` guarantees each thread gets a *different*
number — but it does **not** guarantee *which* thread gets which
number. GPU threads don't run in element order; thread 47 might finish
before thread 3. So two elements with the same sort key could get their
output ranks swapped relative to their original order. For a single
sort pass that's often invisible, but LSD radix sort (the algorithm
used here) sorts one byte at a time from least-significant to
most-significant, and its correctness *depends on* each pass preserving
the relative order established by earlier passes. Breaking that
guarantee even slightly corrupts the final result — and because GPU
thread scheduling is itself somewhat run-dependent, this showed up as
"instance 2 is wrong" purely by coincidence of *when* it happened to be
tested, not because instances actually interfered with each other.

**The fix**: rewrote the entire radix pass (histogram + prefix sum +
scatter) to run serially in a single thread, using the running prefix
sum itself as an insertion cursor — removing the atomic entirely. Costs
a little extra time, irrelevant at this pipeline's small scale (≤256
elements).

**Why the fix works**: a serial scatter processes elements in a fixed,
known order, so equal-key elements land in the output in the same
relative order they started in — the exact property LSD radix sort
needs and atomics couldn't guarantee.

**Lesson**: a bug that looks *exactly* like "state is leaking between
separate objects/instances" deserves real suspicion if the objects
involved don't actually share any state — here, each `GPUBVHPipeline`
had entirely its own GPU buffers. The real tell that it wasn't a
lifecycle bug: results were deterministic *per run* but sensitive to
unrelated scheduling factors, not truly random — a hallmark of a
race/ordering bug being misread as something else.
