#version 450

// Adapted from salvaged/RFR/src/rfr_rendering/shaders/accretion_disk.frag
// -- read in full, confirmed clean (no corruption, unlike its sibling
// star_corona.frag). Physics kept as-is: blackbody temperature-color
// approximation, radius-based temperature gradient, exponential glow
// falloff. What changed: the original expected worldPos/normal varyings
// from a vertex shader driving a real annulus mesh. This project renders
// celestial-body shaders as fixed-resolution "sprite facing the camera"
// fullscreen-triangle renders (same pattern as planet_surface.frag), so
// the disk is reconstructed directly from the 2D sprite UV (r = radial
// distance from sprite center) instead of an intersected 3D worldPos.
// The single view-dependent Doppler dot product became a full angular
// asymmetry (brighter/blueshifted on the approaching side, dimmer/
// redshifted on the receding side) since a flat 2D sprite has no real
// "view direction into the disk" -- this is the same visual signature
// real accretion-disk renders show (relativistic beaming brightens one
// side of the ring more than the other).
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 3, binding = 0) uniform UBO {
    float innerRadius;
    float outerRadius;
    float temperature;
    float doppler;      // RelativisticCamera.doppler_shift() -- 1.0 = no relative motion
    float brightness;   // RelativisticCamera.brightness_boost() -- 1.0 = no relative motion
} ubo;

// The original salvaged formula (1.2929*pow(t,-0.133) etc, t=T/10000)
// was empirically confirmed to saturate to solid white
// across the entire realistic temperature range for both accretion
// disks and stars (a few thousand to tens of thousands of Kelvin) --
// two renders at 3000K and 15000K produced byte-identical mean color.
// Replaced with a simple, verified, monotonic red -> white -> blue-white
// ramp (not the rigorous CIE blackbody locus, just a robust visual
// approximation that actually varies across this range).
vec3 blackbody(float T) {
    float t = clamp(T / 40000.0, 0.0, 1.0);
    vec3 cool = vec3(1.0, 0.35, 0.1);
    vec3 mid = vec3(1.0, 0.95, 0.85);
    vec3 hot = vec3(0.65, 0.75, 1.0);
    return t < 0.5 ? mix(cool, mid, t * 2.0) : mix(mid, hot, (t - 0.5) * 2.0);
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float r = length(uv);

    if (r < ubo.innerRadius || r > ubo.outerRadius) {
        frag_color = vec4(0.0);
        return;
    }

    float t = mix(ubo.temperature * 2.0, ubo.temperature * 0.2,
                  (r - ubo.innerRadius) / (ubo.outerRadius - ubo.innerRadius));
    vec3 color = blackbody(t);

    // phase in [-1, 1]: which side of the disk this pixel is on.
    float phase = uv.x / max(r, 0.0001);
    float side_doppler = mix(1.0 / ubo.doppler, ubo.doppler, (phase + 1.0) * 0.5);
    float side_brightness = mix(1.0 / ubo.brightness, ubo.brightness, (phase + 1.0) * 0.5);

    color.r *= side_doppler;
    color.b /= side_doppler;
    color *= side_brightness;

    float glow = exp(-abs(r - ubo.innerRadius) * 5.0);
    frag_color = vec4(color * glow, glow);
}
