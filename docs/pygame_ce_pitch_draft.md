# Draft: pygame-ce issue comment (NOT POSTED — draft only)

Target: [#2760](https://github.com/pygame-community/pygame-ce/issues/2760) ("DISCUSSION: pygame-ce 3.0 API changes"), cross-referencing [#2331](https://github.com/pygame-community/pygame-ce/issues/2331) ("SDL3 -> pygame-ce 3.0?") for context.

**Before posting this anywhere**: confirm the repo (currently `github.com/Trickaay/physics_simulator_poc-`) is public and pushed with the two example files below actually present at the paths referenced -- paths below already reflect the completed RFR→Relativity rename. This is a draft for the user to review/edit, not text to be posted automatically.

---

## Draft comment text

Since SDL3's new GPU API supports real compute and graphics shaders, I wanted to share a small experiment relevant to this thread: a from-scratch compiler for a tiny Python-like kernel DSL that targets SDL3's GPU API (compute + fragment shaders) and runs alongside pygame's existing 2D drawing in the same window/frame, without needing pygame itself to expose any new API yet.

**The approach**: source in a small `.rfrk` DSL (Taichi-style ergonomics — `for i in range(n):` inside a `kernel` becomes parallel GPU dispatch automatically, array/"field" references don't need manual binding numbers) → generated GLSL → `glslangValidator` → `spirv-cross` → SDL3's GPU API (tested on Metal so far; SDL3 also supports Vulkan/D3D12, not yet tried). It deliberately avoids depending on Taichi (or any other heavy runtime) at runtime — the whole subsystem only needs `numpy`, `pygame-ce`/`PySDL3`, and the two external CLI tools above.

Two examples, smallest first:
- A ~150-line minimal demo: one real GPU compute dispatch per frame (a parallel sine-wave kernel), read back and drawn with ordinary `pygame.draw.circle` calls — the core proof point (compute + legacy 2D drawing coexisting) in isolation.
- A more substantial workload: a real Karras binary-radix-tree BVH build + primary-ray GPU traversal, entirely authored in the DSL, camera-interactive.

**Honest limitations**: only tested on macOS/Metal so far, not Vulkan/D3D12; the BVH demo caps at ≤256 primitives (a single-workgroup radix sort, a scope choice not a hard limit); primary-ray visibility only, no shadows/reflections yet; and — like anything using SDL3's GPU device — this needs a real windowing-system connection, it won't run headless.

(Separate from the DSL, but through the same GLSL → SDL3/MSL toolchain: a small procedural-planet demo — a biome-banded rocky surface plus a real ray-marched Rayleigh/Mie atmosphere, composited onto ordinary pygame drawing every frame — if a more visually concrete example of "GPU shaders + pygame coexisting" is useful alongside the more technical DSL/BVH ones.)

I'm not proposing a specific pygame-ce API shape yet — mostly asking: is exposing something like this (compute/shader access alongside the existing 2D API, without breaking it) a direction that's actually of interest for 3.0? Happy to share the actual code/repo and put more work into this if there's appetite, rather than showing up with an unsolicited large PR.

(Separately, re #3187's main-loop-structure proposal: a project of mine already separates fixed-timestep physics stepping from the render loop in a way that looks structurally similar to what a main-callbacks-compatible loop would need — happy to talk through that too if useful, though it's a separate, bigger conversation from the GPU-access question above.)

---

## Notes for the user (not part of the draft itself)

- Tone deliberately asks a question rather than presenting a finished proposal — both #2331 and #2760 show no maintainer engagement yet and no existing PRs, so a scoped, inviting comment seemed like the right first move rather than a large unsolicited PR (per the pygame_sdl3_upstream_plan memory's own reasoning).
- The #3187 paragraph is intentionally short/separate — it's a bigger, different conversation (event-loop architecture, not GPU access) and bundling too much into one comment risks diluting both asks.
- Consider posting the GPU-access point to #2760 first and only bringing up #3187 later/separately once there's a response, rather than both at once as drafted here — your call.
