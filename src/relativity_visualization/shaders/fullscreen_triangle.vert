#version 450

// Standard "fullscreen triangle" trick: 3 hardcoded NDC vertices derived
// purely from gl_VertexIndex, no vertex buffer needed at all. One
// oversized triangle covers the whole viewport (clipped), cheaper than
// a textbook quad (4 verts / 2 tris / a shared diagonal edge).
layout(location = 0) out vec2 v_uv;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
