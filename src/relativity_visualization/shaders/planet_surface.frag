#version 450

// Procedural planet-surface shader -- NOT adapted from the 363-file
// library (searched: nothing there does procedural planet texturing,
// only atmo_core.glsl's single 2D hash for adaptive-optics phase
// screens, an unrelated use). Written fresh, in the same style as
// raymarch_sphere.frag (analytic ray-sphere intersection, same camera
// setup) and smoke_selfshadow_adapted.frag (explicit set=3 UBO,
// all-scalar layout to avoid std140 vec3 padding, matching
// advect_adapted.comp/diffuse_adapted.comp's convention).
//
// Rendered as a fixed 256x256 "sphere facing the camera" sprite,
// independent of the actual on-screen size/position -- the host
// (gpu_planet_renderer.py) scales/positions the output per planet per
// frame instead of resizing the GPU target. Pixels outside the sphere's
// silhouette get alpha=0 so the sprite composites correctly against
// whatever pygame draws behind it.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 3, binding = 0) uniform UBO {
    float lightX, lightY, lightZ;      // light direction, already in this sprite's local frame
    float rotationPhase;               // radians, spins the noise-sampling coordinate
    float paletteLowR, paletteLowG, paletteLowB;
    float paletteHighR, paletteHighG, paletteHighB;
    float bandStrength;
    float turbulence;
    float seed;
    float kindGasGiant;                // > 0.5 -> banded/swirled, else blotchy
    float terrainStrength;             // 0 = flat shading (original behavior); >0 = bump-mapped relief
    float earthlike;                   // > 0.5 -> water/sand/vegetation/ice altitude bands; else a plain palette gradient (Mars/Mercury/etc.)
    float asteroidMode;                // > 0.5 -> tri-planar rock coloring for "hero" belt asteroids, no altitude ladder or polar ice
} ubo;

float hash3(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3d(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = hash3(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash3(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash3(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash3(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash3(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash3(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash3(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash3(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

float fbm(vec3 p) {
    float value = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        value += amp * noise3d(p * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    return value;
}

// Concept adapted from salvaged/RFR/src/rfr_autonomous_universe3/planet_surface.py
// (FBM + ridged + billow height blend) -- not that file's code, which
// doesn't compile against the real Taichi API and uses a dimensionally
// unsound hash function. Implemented fresh here reusing this shader's
// own already-verified noise3d/fbm (ridged = 1-|2n-1|, billow = |2n-1|
// are trivial derivations of the same base noise, no new primitives).
float heightAt(vec3 dir) {
    float base = fbm(dir * 3.0 + ubo.seed);
    float n = noise3d(dir * 6.0 + ubo.seed);
    float ridged = 1.0 - abs(n * 2.0 - 1.0);
    float billow = abs(n * 2.0 - 1.0);
    return base * 0.6 + ridged * 0.3 - billow * 0.1;
}

vec3 rotate_y(vec3 v, float cr, float sr) {
    return vec3(v.x * cr - v.z * sr, v.y, v.x * sr + v.z * cr);
}

// Tri-planar procedural rock coloring for "hero" belt asteroids --
// ported from a raymarched-SDF prototype (asteroid_rock.frag) that
// proved the technique looks right, but keeps THIS shader's cheap
// analytic-sphere intersection instead of raymarching (a real asteroid
// belt has thousands of members; even a handful of raymarched-per-
// dispatch bodies measured at ~3.9ms each, vs. this shader's ~0.7ms --
// see consolidation_particle_belts memory). Blends three axis-projected
// fbm samples by the normal's squared components (GPU Gems 3's
// technique, same one the prototype used) -- on a perfect sphere this
// doesn't solve a real UV-seam problem the way it would on an irregular
// mesh (direction-based sampling already has no seams), but the
// resulting mottled dark/mid/light + speckle look is what actually
// reads as "rock" rather than a smooth gradient, which is the real
// point of reusing it here.
vec3 asteroidRock(vec3 p, vec3 n) {
    vec3 w = n * n;
    w /= (w.x + w.y + w.z);

    float freq = 3.0;
    float nx = fbm(vec3(p.y, p.z, 0.0) * freq + ubo.seed + 11.0);
    float ny = fbm(vec3(p.z, p.x, 0.0) * freq + ubo.seed + 23.0);
    float nz = fbm(vec3(p.x, p.y, 0.0) * freq + ubo.seed + 37.0);
    float blended = nx * w.x + ny * w.y + nz * w.z;

    vec3 low = vec3(ubo.paletteLowR, ubo.paletteLowG, ubo.paletteLowB);
    vec3 high = vec3(ubo.paletteHighR, ubo.paletteHighG, ubo.paletteHighB);
    vec3 mid = mix(low, high, 0.5);
    vec3 color = blended < 0.5 ? mix(low, mid, blended * 2.0) : mix(mid, high, (blended - 0.5) * 2.0);

    float speckle_x = fbm(vec3(p.y, p.z, 0.0) * 14.0 + ubo.seed + 71.0);
    float speckle_y = fbm(vec3(p.z, p.x, 0.0) * 14.0 + ubo.seed + 83.0);
    float speckle_z = fbm(vec3(p.x, p.y, 0.0) * 14.0 + ubo.seed + 97.0);
    float speckle = speckle_x * w.x + speckle_y * w.y + speckle_z * w.z;
    color *= 0.85 + 0.3 * speckle;

    return color;
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    vec3 ray_origin = vec3(0.0, 0.0, -3.0);
    vec3 ray_dir = normalize(vec3(uv, 1.5));

    // analytic ray/unit-sphere intersection
    float b = dot(ray_origin, ray_dir);
    float c = dot(ray_origin, ray_origin) - 1.0;
    float disc = b * b - c;
    if (disc < 0.0) {
        frag_color = vec4(0.0);
        return;
    }
    float t = -b - sqrt(disc);
    if (t < 0.0) {
        frag_color = vec4(0.0);
        return;
    }

    vec3 hit_pos = ray_origin + ray_dir * t;
    vec3 n = normalize(hit_pos);

    float cr = cos(ubo.rotationPhase);
    float sr = sin(ubo.rotationPhase);
    vec3 rotated = vec3(n.x * cr - n.z * sr, n.y, n.x * sr + n.z * cr);

    vec3 light_dir = normalize(vec3(ubo.lightX, ubo.lightY, ubo.lightZ));

    // Bump-mapped terrain relief: perturb the shading normal by the
    // local height gradient (finite difference in a tangent frame
    // around n), no geometry displacement needed since these sprites
    // are analytic-intersection raymarches, not real meshes.
    vec3 bumped_normal = n;
    if (ubo.terrainStrength > 0.0) {
        vec3 up_hint = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent = normalize(cross(up_hint, n));
        vec3 bitangent = cross(n, tangent);

        float eps = 0.05;
        float h0 = heightAt(rotated);
        float hT = heightAt(rotate_y(normalize(n + tangent * eps), cr, sr));
        float hB = heightAt(rotate_y(normalize(n + bitangent * eps), cr, sr));

        vec3 bump = (tangent * (hT - h0) + bitangent * (hB - h0)) * (ubo.terrainStrength / eps);
        bumped_normal = normalize(n - bump);
    }
    float diffuse = clamp(dot(bumped_normal, light_dir), 0.08, 1.0);

    float turb = fbm(rotated * 3.0 + ubo.seed);

    vec3 low = vec3(ubo.paletteLowR, ubo.paletteLowG, ubo.paletteLowB);
    vec3 high = vec3(ubo.paletteHighR, ubo.paletteHighG, ubo.paletteHighB);
    vec3 surface_color;

    if (ubo.kindGasGiant > 0.5) {
        float band = rotated.y * 3.0 + (turb - 0.5) * ubo.turbulence * 4.0;
        float mix_factor = clamp(0.5 + 0.5 * sin(band * 3.14159), 0.0, 1.0);
        mix_factor = clamp(mix_factor + (turb - 0.5) * ubo.bandStrength * 0.4, 0.0, 1.0);
        surface_color = mix(low, high, mix_factor);
    } else {
        float alt = heightAt(rotated);
        vec3 ice = vec3(0.92, 0.95, 1.0);

        if (ubo.asteroidMode > 0.5) {
            surface_color = asteroidRock(rotated, bumped_normal);
        } else if (ubo.earthlike > 0.5) {
            // Altitude-banded biome ladder (water -> sand -> vegetation ->
            // rock -> ice), adapted from Julien Sulpis' "Procedural Blue
            // Planet" (https://www.shadertoy.com/view/Ds3XRl) -- same idea
            // (a sequence of smoothstep transitions on a single height
            // value), reusing this shader's OWN existing heightAt()
            // (already computed above for bump-mapping) as that height
            // value instead of introducing a second noise function, and
            // paletteLow/paletteHigh as the water/rock endpoints so
            // per-planet palette customization still works -- only the
            // intermediate sand/vegetation/ice bands are fixed constants,
            // matching the reference's own approach (it hardcodes all
            // band colors, no per-planet palette at all). Only appropriate
            // for a water/vegetation world (Earth) -- see the `else`
            // branch below for dry rocky bodies. Thresholds tuned against
            // heightAt()'s actual empirical range (~0.04-0.72 across a
            // rendered disk, verified directly, not guessed) so every
            // band is genuinely reachable.
            vec3 water_deep = low;
            vec3 water_surface = mix(low, vec3(0.05, 0.25, 0.4), 0.5);
            vec3 sand = vec3(0.82, 0.75, 0.55);
            vec3 vegetation = vec3(0.15, 0.35, 0.12);
            vec3 rock = high;

            const float WATER_LEVEL = 0.40;
            const float SAND_LEVEL = 0.44;
            const float VEG_LEVEL = 0.52;
            const float ROCK_LEVEL = 0.60;
            const float ICE_LEVEL = 0.66;
            const float TRANS = 0.03;

            surface_color = mix(water_deep, water_surface, smoothstep(0.0, WATER_LEVEL, alt));
            surface_color = mix(surface_color, sand, smoothstep(WATER_LEVEL, WATER_LEVEL + TRANS, alt));
            surface_color = mix(surface_color, vegetation, smoothstep(SAND_LEVEL, SAND_LEVEL + TRANS, alt));
            surface_color = mix(surface_color, rock, smoothstep(VEG_LEVEL, VEG_LEVEL + TRANS, alt));
            surface_color = mix(surface_color, ice, smoothstep(ROCK_LEVEL, ROCK_LEVEL + TRANS, alt));
        } else {
            // Dry rocky body (Mars, Mercury, ...): no oceans or
            // vegetation, just a plain altitude gradient across the
            // planet's own low/high palette -- still terrain variety, but
            // no Earth-specific colors forced onto it.
            surface_color = mix(low, high, smoothstep(0.1, 0.6, alt));
        }

        // Poles are cold regardless of altitude -- an additional latitude
        // mask (same rotated.y proxy the gas-giant branch above uses for
        // banding), separate from and layered on top of the altitude
        // ladder. A real feature on both Earth and Mars, so it applies
        // regardless of the earthlike flag -- but NOT for asteroids
        // (real belt asteroids don't have polar ice caps; this mask
        // would just paint a fake white cap onto an otherwise plain
        // rock). Idea from Reinder Nijhoff's "Planet" shader
        // (https://www.shadertoy.com/view/4tjGRh)'s `iceFactor =
        // abs(pow(z/EARTH_RADIUS, 13))`.
        if (ubo.asteroidMode <= 0.5) {
            float polar = pow(abs(rotated.y), 8.0);
            surface_color = mix(surface_color, ice, clamp(polar - 0.15, 0.0, 1.0) * 0.85);
        }
    }

    float rim = pow(1.0 - clamp(dot(n, -ray_dir), 0.0, 1.0), 3.0);
    vec3 color = surface_color * diffuse + rim * 0.15;

    frag_color = vec4(color, 1.0);
}
