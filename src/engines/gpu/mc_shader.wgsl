// Monte Carlo European option pricing compute shader.
// Each workgroup thread simulates one path and computes the payoff.

struct Params {
    spot: f32,
    strike: f32,
    rate: f32,
    vol: f32,
    expiry: f32,
    dt_drift: f32,    // (r - 0.5*vol^2) * dt
    dt_vol: f32,      // vol * sqrt(dt)
    discount: f32,
    num_steps: u32,
    num_paths: u32,
    seed: u32,
    is_call: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> payoffs: array<f32>;

// Xoshiro128++ PRNG state per thread (4x u32).
var<private> rng_s0: u32;
var<private> rng_s1: u32;
var<private> rng_s2: u32;
var<private> rng_s3: u32;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn xoshiro128pp_next() -> u32 {
    let result = rotl(rng_s0 + rng_s3, 7u) + rng_s0;
    let t = rng_s1 << 9u;
    rng_s2 ^= rng_s0;
    rng_s3 ^= rng_s1;
    rng_s1 ^= rng_s2;
    rng_s0 ^= rng_s3;
    rng_s2 ^= t;
    rng_s3 = rotl(rng_s3, 11u);
    return result;
}

fn xoshiro128pp_f32() -> f32 {
    // Map u32 -> (0, 1) float.
    return f32(xoshiro128pp_next() >> 8u) * 5.9604644775390625e-8;
}

fn seed_rng(global_id: u32) {
    // SplitMix32 to derive initial state from global_id + params.seed.
    var z = global_id + params.seed;

    z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u;
    rng_s0 = z;
    z += 0x9e3779b9u;
    z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u;
    rng_s1 = z;
    z += 0x9e3779b9u;
    z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u;
    rng_s2 = z;
    z += 0x9e3779b9u;
    z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u; z *= 0x45d9f3bu; z ^= z >> 16u;
    rng_s3 = z;
}

// Box-Muller transform: generate standard normal from two uniforms.
fn box_muller() -> vec2<f32> {
    let u1 = max(xoshiro128pp_f32(), 1.0e-10);
    let u2 = xoshiro128pp_f32();
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let path_id = global_id.x;
    if path_id >= params.num_paths {
        return;
    }

    seed_rng(path_id);

    var spot = params.spot;
    let dt_drift = params.dt_drift;
    let dt_vol = params.dt_vol;
    let num_steps = params.num_steps;

    // Simulate GBM path, consuming normals from Box-Muller pairs.
    var step = 0u;
    // Process two steps at a time using both Box-Muller outputs.
    let pairs = num_steps / 2u;
    for (var p = 0u; p < pairs; p++) {
        let z = box_muller();
        spot *= exp(fma(dt_vol, z.x, dt_drift));
        spot *= exp(fma(dt_vol, z.y, dt_drift));
    }
    // Handle odd remaining step.
    if (num_steps & 1u) != 0u {
        let z = box_muller();
        spot *= exp(fma(dt_vol, z.x, dt_drift));
    }

    // Compute payoff.
    var payoff: f32;
    if params.is_call != 0u {
        payoff = max(spot - params.strike, 0.0);
    } else {
        payoff = max(params.strike - spot, 0.0);
    }

    payoffs[path_id] = payoff;
}
