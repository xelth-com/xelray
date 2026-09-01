// Shaders for the organ-mesh viewer: one opaque pass, then weighted blended
// order-independent transparency (Meshkin / McGuire & Bavoil) for the
// translucent shells, then a fullscreen composite.
//
// Why OIT at all: a segmentation is a dozen nested closed surfaces — skin
// around fat around muscle around bone. Sorting them per triangle is hopeless
// and sorting them per organ is wrong wherever two organs interpenetrate,
// which is exactly where marching cubes leaves slivers. WBOIT is
// order-independent and costs one extra pass.
//
// Coordinates: vertices arrive in **LPS millimetres** and `view_proj` already
// folds in the LPS -> display flip `S = diag(1, -1, 1)` (see camera.rs). The
// vertex shader applies the same flip by hand to the normal and to the world
// position it hands the fragment shader, which both live in *display* space —
// the space `globals.eye` is in.

struct Globals {
    // proj * look_at * S. Feed it raw LPS positions.
    view_proj: mat4x4<f32>,
    // xyz: eye position, display space. w unused.
    eye: vec4<f32>,
    // x: 1.0 when the surface format is not sRGB and the shader has to encode
    // the transfer function itself. Rest unused.
    params: vec4<f32>,
};

struct Material {
    // Linear RGB with straight (non-premultiplied) alpha. The sRGB -> linear
    // conversion happens once on the CPU when the buffer is filled.
    color: vec4<f32>,
};

// Group 0 carries the per-frame globals plus, for the composite pipeline only,
// the two OIT targets. The mesh pipelines bind a layout holding binding 0
// alone; a binding no entry point of theirs reads is not part of their
// interface, so the three can share one module.
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var t_accum: texture_2d<f32>;
@group(0) @binding(2) var t_reveal: texture_2d<f32>;

// Group 1 is one dynamic-offset slice of the per-group material buffer.
@group(1) @binding(0) var<uniform> material: Material;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
};

@vertex
fn vs_mesh(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(pos, 1.0);
    // `S` by hand. For a pure axis flip the normal matrix (transpose of the
    // inverse) is `S` itself, so this is the same expression twice.
    out.world = vec3<f32>(pos.x, -pos.y, pos.z);
    out.normal = vec3<f32>(nrm.x, -nrm.y, nrm.z);
    return out;
}

// Hemisphere ambient, +z up in display space (superior). Warm-neutral sky
// over a dim ground keeps the underside of an organ readable instead of
// black, which a pure headlight would leave it.
const SKY: vec3<f32> = vec3<f32>(0.45);
const GROUND: vec3<f32> = vec3<f32>(0.15);

fn shade(normal: vec3<f32>, world: vec3<f32>, front_facing: bool, base: vec3<f32>) -> vec3<f32> {
    // Marching-cubes shells are drawn unculled — `S` flips the winding and the
    // slivers would drop out otherwise — so a back-facing fragment's normal
    // points away from us and has to be turned around before it is lit.
    var n = normalize(normal);
    n = select(n, -n, !front_facing);

    let v = normalize(globals.eye.xyz - world);
    // Headlight: the light sits at the eye, so L == V and the Blinn-Phong
    // half vector collapses to V as well.
    let ndl = max(dot(n, v), 0.0);
    let ambient = mix(GROUND, SKY, n.z * 0.5 + 0.5);
    let spec = pow(ndl, 32.0) * 0.25;
    return base * (ambient + ndl * 0.85) + vec3<f32>(spec);
}

// Only used when the surface came back in a non-sRGB format (some WebGL2
// configurations); otherwise the hardware does this after blending, which is
// both cheaper and more correct.
fn encode(c: vec3<f32>) -> vec3<f32> {
    if globals.params.x < 0.5 {
        return c;
    }
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// ---------------------------------------------------------------------------
// Pass 1 — opaque
// ---------------------------------------------------------------------------

@fragment
fn fs_opaque(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(encode(shade(in.normal, in.world, front, material.color.rgb)), 1.0);
}

// ---------------------------------------------------------------------------
// Pass 2 — WBOIT accumulation
// ---------------------------------------------------------------------------

struct WboitOut {
    // Rgba16Float, blended ONE/ONE: sum of premultiplied colour * weight.
    @location(0) accum: vec4<f32>,
    // R16Float, blended ZERO/ONE_MINUS_SRC: the running product of (1 - a).
    @location(1) reveal: f32,
};

@fragment
fn fs_wboit(in: VsOut, @builtin(front_facing) front: bool) -> WboitOut {
    let a = material.color.a;
    let lit = shade(in.normal, in.world, front, material.color.rgb);

    // McGuire & Bavoil's depth weight. Their `z` is view-space depth; the
    // distance from the eye is the same thing to within the cosine of the
    // field of view, and saves shipping a second matrix. The 200 mm scale is
    // roughly a torso, so the whole body lands in the useful part of the
    // curve.
    let z = length(globals.eye.xyz - in.world);
    let w = a * clamp(10.0 / (1e-5 + pow(z / 200.0, 4.0)), 1e-2, 3e3);

    var out: WboitOut;
    out.accum = vec4<f32>(lit * a, a) * w;
    out.reveal = a;
    return out;
}

// ---------------------------------------------------------------------------
// Pass 3 — composite
// ---------------------------------------------------------------------------

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covering the viewport, addressed by index alone —
    // no vertex buffer, and no seam down the middle of a two-triangle quad.
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_composite(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let at = vec2<i32>(frag.xy);
    let accum = textureLoad(t_accum, at, 0);
    let reveal = textureLoad(t_reveal, at, 0).r;

    // Weight-averaged colour, then the canonical WBOIT resolve: the pipeline
    // blends with src = ONE_MINUS_SRC_ALPHA and dst = SRC_ALPHA, so emitting
    // `reveal` as alpha gives dst = avg * (1 - reveal) + dst * reveal — the
    // translucent layers over the opaque image behind them.
    let avg = accum.rgb / max(accum.a, 1e-4);
    return vec4<f32>(encode(avg), reveal);
}

// ---------------------------------------------------------------------------
// Fallback — plain sorted alpha, for adapters that cannot blend into a float
// colour attachment (WebGL2 without EXT_float_blend).
// ---------------------------------------------------------------------------

@fragment
fn fs_blend(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let lit = shade(in.normal, in.world, front, material.color.rgb);
    return vec4<f32>(encode(lit), material.color.a);
}
