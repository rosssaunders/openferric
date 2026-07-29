// Monte Carlo European option pricing compute shader with on-device reduction.
// Each invocation uses both normals from one Box-Muller pair to price two
// exact-terminal GBM samples. A tree reduction in shared memory produces
// per-workgroup partial sums, so only those summaries are read back.

struct Params {
    spot: f32,
    strike: f32,
    terminal_drift: f32, // (r - 0.5*vol^2) * expiry
    terminal_vol: f32,   // vol * sqrt(expiry)
    num_paths: u32,
    seed: u32,
    is_call: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<f32>;

// Workgroup shared memory for hierarchical tree reduction.
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sum_sq: array<f32, 256>;

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

fn payoff_for_normal(z: f32) -> f32 {
    let terminal_spot = params.spot * exp(fma(params.terminal_vol, z, params.terminal_drift));
    if params.is_call != 0u {
        return max(terminal_spot - params.strike, 0.0);
    }
    return max(params.strike - terminal_spot, 0.0);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let first_path_id = global_id.x * 2u;
    let lid = local_id.x;

    // Invocations beyond num_paths contribute zero to the reduction. For an
    // odd path count, the final invocation uses only the first normal.
    var payoff_sum: f32 = 0.0;
    var payoff_sum_sq: f32 = 0.0;
    if first_path_id < params.num_paths {
        seed_rng(global_id.x);
        let normals = box_muller();

        let first_payoff = payoff_for_normal(normals.x);
        payoff_sum = first_payoff;
        payoff_sum_sq = first_payoff * first_payoff;

        if first_path_id + 1u < params.num_paths {
            let second_payoff = payoff_for_normal(normals.y);
            payoff_sum += second_payoff;
            payoff_sum_sq += second_payoff * second_payoff;
        }
    }

    // ── Hierarchical tree reduction in shared memory ──
    shared_sum[lid] = payoff_sum;
    shared_sum_sq[lid] = payoff_sum_sq;
    workgroupBarrier();

    // 8 reduction steps for workgroup_size 256 (log2(256) = 8).
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lid < stride {
            shared_sum[lid] += shared_sum[lid + stride];
            shared_sum_sq[lid] += shared_sum_sq[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes workgroup partial sums to the output buffer.
    // Layout: [sum_0, sum_sq_0, sum_1, sum_sq_1, ...]
    if lid == 0u {
        let wg_idx = wg_id.x;
        partial_sums[wg_idx * 2u] = shared_sum[0];
        partial_sums[wg_idx * 2u + 1u] = shared_sum_sq[0];
    }
}
