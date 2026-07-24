#version 450

// Real ray-marched Rayleigh + Mie atmospheric scattering, replacing this
// file's earlier single-sample approximation (prompted by two reference
// shaders the user found -- David A Roberts' "Humanity" and Reinder
// Nijhoff's "Planet" -- both credit the exact same GLtracy scattering
// technique this file's own header already traced its lineage to,
// confirming it as the right one to finish porting rather than the
// single-sample stand-in with an EXPOSURE_SCALE hack that was here
// before). density()/optic()/in_scatter()/phase_ray()/phase_mie() are
// GLtracy's own technique (https://www.shadertoy.com/view/lslXDr),
// generalized from that reference's hardcoded R_INNER=1.0/
// ATMOSPHERE_THICKNESS=0.2 constants to this shader's existing
// ubo.planetRadius/ubo.atmosphereRadius (so the UBO layout and
// render_atmosphere() call signature are UNCHANGED -- gpu_celestial_renderer.py
// needs no changes at all). The reference's own k_ray/k_mie constants
// (tuned for a unit-scale sphere, used WITH real integration) replace
// the old real-world SI-scale Rayleigh/Mie coefficients this file
// previously carried -- those were correct in principle but written for
// a real km-scale atmosphere integrated along a ray, and only rendered
// visibly here via an exposure hack because there was no integration to
// naturally produce visible brightness; with real integration at this
// shader's actual unit scale, GLtracy's own constants are the correct
// fit and the exposure hack is no longer needed.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

const float PI = 3.14159265359;
const int NUM_OUT_SCATTER = 8;
const int NUM_IN_SCATTER = 80;

layout(set = 3, binding = 0) uniform UBO {
    float sunDirX, sunDirY, sunDirZ;
    float planetRadius;
    float atmosphereRadius;
} ubo;

// ray vs sphere centered at the origin -- returns (tmin, tmax), or
// (1e4, -1e4) (tmin > tmax, an easy "no hit" sentinel) on a miss.
vec2 ray_vs_sphere(vec3 p, vec3 dir, float r) {
    float b = dot(p, dir);
    float c = dot(p, p) - r * r;
    float d = b * b - c;
    if (d < 0.0) {
        return vec2(1e4, -1e4);
    }
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}

float density(vec3 p, float ph) {
    float thickness = max(ubo.atmosphereRadius - ubo.planetRadius, 1e-4);
    return exp(-max(length(p) - ubo.planetRadius, 0.0) / thickness / ph);
}

// optical depth (integrated density) along the segment p->q
float optic(vec3 p, vec3 q, float ph) {
    vec3 s = (q - p) / float(NUM_OUT_SCATTER);
    vec3 v = p + s * 0.5;

    float sum = 0.0;
    for (int i = 0; i < NUM_OUT_SCATTER; i++) {
        sum += density(v, ph);
        v += s;
    }
    sum *= length(s);
    return sum;
}

float phase_ray(float cc) {
    return (3.0 / (16.0 * PI)) * (1.0 + cc);
}

float phase_mie(float g, float c, float cc) {
    float gg = g * g;
    float a = (1.0 - gg) * (1.0 + cc);
    float b = 1.0 + gg - 2.0 * g * c;
    b *= sqrt(b);
    b *= 2.0 + gg;
    return (3.0 / (8.0 * PI)) * a / b;
}

// walks the primary view ray through the atmosphere shell (e.x..e.y),
// accumulating Rayleigh/Mie in-scatter with proper out-scattering
// attenuation via a secondary ray toward the sun at each sample point.
vec3 in_scatter(vec3 o, vec3 dir, vec2 e, vec3 l) {
    const float ph_ray = 0.05;
    const float ph_mie = 0.02;
    const vec3 k_ray = vec3(3.8, 13.5, 33.1);
    const float k_mie = 21.0;
    const float k_mie_ex = 1.1;

    vec3 sum_ray = vec3(0.0);
    vec3 sum_mie = vec3(0.0);

    float n_ray0 = 0.0;
    float n_mie0 = 0.0;

    float len = (e.y - e.x) / float(NUM_IN_SCATTER);
    vec3 s = dir * len;
    vec3 v = o + dir * (e.x + len * 0.5);

    for (int i = 0; i < NUM_IN_SCATTER; i++) {
        float d_ray = density(v, ph_ray) * len;
        float d_mie = density(v, ph_mie) * len;

        n_ray0 += d_ray;
        n_mie0 += d_mie;

        vec2 f = ray_vs_sphere(v, l, ubo.atmosphereRadius);
        vec3 u = v + l * f.y;

        float n_ray1 = optic(v, u, ph_ray);
        float n_mie1 = optic(v, u, ph_mie);

        vec3 att = exp(-(n_ray0 + n_ray1) * k_ray - (n_mie0 + n_mie1) * k_mie * k_mie_ex);

        sum_ray += d_ray * att;
        sum_mie += d_mie * att;

        v += s;
    }

    float c = dot(dir, -l);
    float cc = c * c;
    vec3 scatter = sum_ray * k_ray * phase_ray(cc) + sum_mie * k_mie * phase_mie(-0.78, c, cc);

    return 10.0 * scatter;
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    vec3 ray_origin = vec3(0.0, 0.0, -3.0);
    vec3 ray_dir = normalize(vec3(uv, 1.5));

    vec2 e = ray_vs_sphere(ray_origin, ray_dir, ubo.atmosphereRadius);
    if (e.x > e.y) {
        frag_color = vec4(0.0);
        return;
    }
    e.x = max(e.x, 0.0);

    // clip the far end to the opaque planet's own surface, if hit, so
    // scattering isn't (incorrectly) integrated through solid ground
    vec2 f = ray_vs_sphere(ray_origin, ray_dir, ubo.planetRadius);
    if (f.x > 0.0) {
        e.y = min(e.y, f.x);
    }

    vec3 sunDir = normalize(vec3(ubo.sunDirX, ubo.sunDirY, ubo.sunDirZ));
    vec3 scatter = in_scatter(ray_origin, ray_dir, e, sunDir);

    float alpha = clamp(length(scatter) * 1.5, 0.0, 1.0);
    frag_color = vec4(scatter, alpha);
}
