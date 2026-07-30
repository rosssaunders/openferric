// Monte Carlo European option pricing compute shader with on-device reduction.
// Each invocation uses both normals from one Box-Muller pair to price two
// exact-terminal GBM samples. A tree reduction in shared memory produces
// per-workgroup centered-moment summaries, so only those are read back.

struct Params {
    spot: f32,
    strike: f32,
    terminal_drift: f32, // (r - 0.5*vol^2) * expiry
    terminal_vol: f32,   // vol * sqrt(expiry)
    num_paths: u32,
    seed_low: u32,
    seed_high: u32,
    is_call: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> partial_moments: array<f32>;

// Workgroup shared memory for a Chan centered-moment reduction.
var<workgroup> shared_count: array<f32, 256>;
var<workgroup> shared_mean: array<f32, 256>;
var<workgroup> shared_m2: array<f32, 256>;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

// Threefry2x32-20 counter-based generator. The invocation id is a counter,
// while both halves of the public u64 seed form the key. Unlike seeding a
// stateful generator from `path_id + seed`, adjacent seeds do not shift and
// reuse almost the entire set of per-path streams.
fn threefry2x32(counter: vec2<u32>) -> vec2<u32> {
    let parity = 0x1bd11bdau;
    let keys = array<u32, 3>(
        params.seed_low,
        params.seed_high,
        parity ^ params.seed_low ^ params.seed_high,
    );
    let rotations = array<u32, 8>(13u, 15u, 26u, 6u, 17u, 29u, 16u, 24u);

    var value = vec2<u32>(counter.x + keys[0], counter.y + keys[1]);
    for (var round = 0u; round < 20u; round += 1u) {
        value.x += value.y;
        value.y = rotl(value.y, rotations[round & 7u]) ^ value.x;

        let completed = round + 1u;
        if completed % 4u == 0u {
            let injection = completed / 4u;
            value.x += keys[injection % 3u];
            value.y += keys[(injection + 1u) % 3u] + injection;
        }
    }
    return value;
}

fn open_unit(word: u32) -> f32 {
    // Midpoints of 2^23 bins are exactly representable and strictly in (0, 1).
    return (f32(word >> 9u) + 0.5) * 1.1920928955078125e-7;
}

// Box-Muller transform: generate standard normal from two uniforms.
fn box_muller(random_words: vec2<u32>) -> vec2<f32> {
    let u1 = open_unit(random_words.x);
    let u2 = open_unit(random_words.y);
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
    var sample_count: f32 = 0.0;
    var sample_mean: f32 = 0.0;
    var sample_m2: f32 = 0.0;
    if first_path_id < params.num_paths {
        let random_words = threefry2x32(vec2<u32>(global_id.x, 0xd2b74407u));
        let normals = box_muller(random_words);

        let first_payoff = payoff_for_normal(normals.x);
        sample_count = 1.0;
        sample_mean = first_payoff;

        if first_path_id + 1u < params.num_paths {
            let second_payoff = payoff_for_normal(normals.y);
            let delta = second_payoff - sample_mean;
            sample_count = 2.0;
            sample_mean += delta * 0.5;
            sample_m2 = delta * (second_payoff - sample_mean);
        }
    }

    // ── Hierarchical Chan reduction in shared memory ──
    shared_count[lid] = sample_count;
    shared_mean[lid] = sample_mean;
    shared_m2[lid] = sample_m2;
    workgroupBarrier();

    // 8 reduction steps for workgroup_size 256 (log2(256) = 8).
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lid < stride {
            let left_count = shared_count[lid];
            let right_count = shared_count[lid + stride];
            if right_count > 0.0 {
                if left_count == 0.0 {
                    shared_count[lid] = right_count;
                    shared_mean[lid] = shared_mean[lid + stride];
                    shared_m2[lid] = shared_m2[lid + stride];
                } else {
                    let total_count = left_count + right_count;
                    let delta = shared_mean[lid + stride] - shared_mean[lid];
                    shared_mean[lid] += delta * (right_count / total_count);
                    shared_m2[lid] += shared_m2[lid + stride]
                        + delta * delta * (left_count * right_count / total_count);
                    shared_count[lid] = total_count;
                }
            }
        }
        workgroupBarrier();
    }

    // Thread 0 writes a centered-moment summary for its workgroup.
    // Layout: [count_0, mean_0, M2_0, count_1, mean_1, M2_1, ...]
    if lid == 0u {
        let wg_idx = wg_id.x;
        partial_moments[wg_idx * 3u] = shared_count[0];
        partial_moments[wg_idx * 3u + 1u] = shared_mean[0];
        partial_moments[wg_idx * 3u + 2u] = shared_m2[0];
    }
}
