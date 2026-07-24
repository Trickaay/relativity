#version 450

// Adapted from salvaged/RFR/src/rfr_rendering/shaders/star_corona.frag --
// read in full; confirmed CORRUPTED as delivered: a stray
// `layout(push_constant) uniform Relativity {...}` block plus loose
// `color *= rel.doppler;`/`color *= rel.brightness;` statements sat
// AFTER main()'s closing brace, at invalid global scope -- would not
// compile. Fixed here by moving that multiplication inside main(), and
// this is exactly where RelativisticCamera's real output plugs in (see
// relativistic_camera.py) instead of the disconnected placeholder
// salvaged left behind. Same worldPos->UV adaptation as
// accretion_disk.frag: a 2D radial glow sprite instead of a 3D mesh
// varying, matching this project's fullscreen-triangle sprite
// convention (planet_surface.frag).
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 3, binding = 0) uniform UBO {
    float temperature;
    float starRadius;
    float doppler;      // RelativisticCamera.doppler_shift() -- 1.0 = no relative motion
    float brightness;   // RelativisticCamera.brightness_boost() -- 1.0 = no relative motion
    float rotationPhase;
    float seed;
} ubo;

// See accretion_disk.frag's blackbody() comment -- same fix, same reason
// (the original formula saturated to solid white across this range,
// confirmed empirically, not just a style preference).
vec3 blackbody(float T) {
    float t = clamp(T / 40000.0, 0.0, 1.0);
    vec3 cool = vec3(1.0, 0.35, 0.1);
    vec3 mid = vec3(1.0, 0.95, 0.85);
    vec3 hot = vec3(0.65, 0.75, 1.0);
    return t < 0.5 ? mix(cool, mid, t * 2.0) : mix(mid, hot, (t - 0.5) * 2.0);
}

// Same noise/fbm primitives as planet_surface.frag -- reused verbatim
// rather than reinvented, since it's already verified there.
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
    for (int i = 0; i < 4; i++) {
        value += amp * noise3d(p * freq);
        amp *= 0.5;
        freq *= 2.0;
    }
    return value;
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float r = max(ubo.starRadius, 0.001);
    vec3 base_color = blackbody(ubo.temperature);

    // Real analytic ray/sphere intersection (same camera setup as
    // planet_surface.frag: ray_origin at z=-3, FOV factor 1.5) instead
    // of the old pure length(uv) radial vignette -- gives the star an
    // actual rotating textured disk instead of a flat glow blob.
    vec3 ray_origin = vec3(0.0, 0.0, -3.0);
    vec3 ray_dir = normalize(vec3(uv, 1.5));
    float b = dot(ray_origin, ray_dir);
    float c = dot(ray_origin, ray_origin) - r * r;
    float disc = b * b - c;

    vec3 color = vec3(0.0);
    float disk_alpha = 0.0;

    if (disc >= 0.0) {
        float t = -b - sqrt(disc);
        if (t >= 0.0) {
            vec3 hit_pos = ray_origin + ray_dir * t;
            vec3 n = normalize(hit_pos);
            float cr = cos(ubo.rotationPhase);
            float sr = sin(ubo.rotationPhase);
            vec3 rotated = vec3(n.x * cr - n.z * sr, n.y, n.x * sr + n.z * cr);

            // Photosphere granulation: real granulation is a BRIGHTNESS
            // texture (convection cells), not a hue variation, so fbm
            // only modulates a scalar multiplier on the blackbody color
            // -- one higher-frequency layer for small granules, one
            // lower-frequency layer for larger active-region-style
            // patches.
            float granules = fbm(rotated * 8.0 + ubo.seed);
            float patches = fbm(rotated * 2.5 + ubo.seed + 5.0);
            float surface_brightness = 0.75 + granules * 0.35 + patches * 0.25;

            color = base_color * surface_brightness;
            disk_alpha = 1.0;
        }
    }

    // Outer corona glow (unchanged falloff shape from the original
    // flat-vignette version, so existing tuning/visual balance carries
    // over), plus a cheap angular flicker so the rim isn't perfectly
    // smooth -- adapted from a mercator-projected sun reference shader's
    // `noise(atan(uv.y, uv.x) * k)` ring modulation.
    float dist = length(uv) / r;
    float glow = exp(-dist * 3.0);
    float angle_noise = fract(sin(atan(uv.y, uv.x) * 12.9898 + ubo.seed) * 43758.5453);
    float rim_flicker = 0.75 + 0.25 * angle_noise;

    color += base_color * pow(glow, 4.0) * 2.0;
    color += base_color * glow * rim_flicker * (1.0 - disk_alpha);

    color *= ubo.doppler;
    color *= ubo.brightness;

    float alpha = max(disk_alpha, glow);
    frag_color = vec4(color, alpha);
}
