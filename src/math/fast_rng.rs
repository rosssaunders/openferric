//! Module `math::fast_rng`.
//!
//! Implements fast rng workflows with concrete routines such as `fill_normals`, `stream_seed`, `resolve_stream_seed`, `uniform_open01`.
//!
//! References: Glasserman (2004) Ch. 5, Salmon et al. (2011) "Parallel Random Numbers: As Easy as 1, 2, 3".
//!
//! Key types and purpose: `Philox4x32`, `Xoshiro256Rng`, `Pcg64Rng`, `FastRngKind` define the core data contracts for this module.
//!
//! Numerical considerations: approximation regions, branch choices, and machine-precision cancellation near boundaries should be validated with high-precision references.
//!
//! When to use: use these low-level routines in performance-sensitive calibration/pricing loops; use higher-level modules when model semantics matter more than raw numerics.
use rand::rngs::{StdRng, ThreadRng};
use rand::{RngExt, SeedableRng};

use crate::math::fast_norm::beasley_springer_moro_inv_cdf;

pub type Xoshiro256Rng = Xoshiro256PlusPlus;
pub type Pcg64Rng = Pcg64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FastRngKind {
    #[default]
    Philox4x32,
    Xoshiro256PlusPlus,
    Pcg64,
    ThreadRng,
    StdRng,
}

// ---------------------------------------------------------------------------
// Philox-4x32-10 counter-based RNG (Salmon et al. 2011)
// ---------------------------------------------------------------------------

const PHILOX_M4X32_0: u32 = 0xD251_1F53;
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;
const PHILOX_W32_0: u32 = 0x9E37_79B9;
const PHILOX_W32_1: u32 = 0xBB67_AE85;

#[inline(always)]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = a as u64 * b as u64;
    ((product >> 32) as u32, product as u32)
}

/// Philox-4x32-10 bijection: maps (counter, key) to 4 pseudorandom u32s.
/// Pure function with no mutable state.
#[inline]
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;
    // 10 rounds: round then bump key (final bump is discarded).
    for _ in 0..10 {
        let (hi0, lo0) = mulhilo32(PHILOX_M4X32_0, ctr[0]);
        let (hi1, lo1) = mulhilo32(PHILOX_M4X32_1, ctr[2]);
        ctr = [hi1 ^ ctr[1] ^ k[0], lo1, hi0 ^ ctr[3] ^ k[1], lo0];
        k[0] = k[0].wrapping_add(PHILOX_W32_0);
        k[1] = k[1].wrapping_add(PHILOX_W32_1);
    }
    ctr
}

/// Stateful counter-based RNG wrapping Philox-4x32-10.
///
/// Each invocation of `philox4x32_10` produces 4 u32 outputs. The internal
/// 128-bit counter auto-increments after each batch is consumed, giving a
/// period of 2^130 u32 values.
#[derive(Debug, Clone)]
pub struct Philox4x32 {
    counter: [u32; 4],
    key: [u32; 2],
    buffer: [u32; 4],
    buffer_idx: u8,
}

impl Philox4x32 {
    #[inline]
    pub fn seed_from_u64(seed: u64) -> Self {
        let key = [seed as u32, (seed >> 32) as u32];
        let counter = [0, 0, 0, 0];
        let buffer = philox4x32_10(counter, key);
        Self {
            counter,
            key,
            buffer,
            buffer_idx: 0,
        }
    }

    #[inline(always)]
    fn refill(&mut self) {
        // 128-bit counter increment.
        self.counter[0] = self.counter[0].wrapping_add(1);
        if self.counter[0] == 0 {
            self.counter[1] = self.counter[1].wrapping_add(1);
            if self.counter[1] == 0 {
                self.counter[2] = self.counter[2].wrapping_add(1);
                if self.counter[2] == 0 {
                    self.counter[3] = self.counter[3].wrapping_add(1);
                }
            }
        }
        self.buffer = philox4x32_10(self.counter, self.key);
        self.buffer_idx = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.buffer_idx >= 4 {
            self.refill();
        }
        let val = self.buffer[self.buffer_idx as usize];
        self.buffer_idx += 1;
        val
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        x as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

/// Stateless counter-based uniform random in [0, 1).
/// Same (path_id, step_id, seed) always produces the same result,
/// regardless of thread assignment or evaluation order.
#[inline]
pub fn philox_random(path_id: u64, step_id: u64, seed: u64) -> f64 {
    let counter = [
        path_id as u32,
        (path_id >> 32) as u32,
        step_id as u32,
        (step_id >> 32) as u32,
    ];
    let key = [seed as u32, (seed >> 32) as u32];
    let out = philox4x32_10(counter, key);
    let u = ((out[0] as u64) | ((out[1] as u64) << 32)) >> 11;
    u as f64 * (1.0 / ((1_u64 << 53) as f64))
}

/// Stateless counter-based standard normal via inverse CDF.
#[inline]
pub fn philox_normal(path_id: u64, step_id: u64, seed: u64) -> f64 {
    beasley_springer_moro_inv_cdf(uniform_open01(philox_random(path_id, step_id, seed)))
}

// ---------------------------------------------------------------------------
// Xoshiro256++ (retained as a FastRngKind variant)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Xoshiro256PlusPlus {
    state: [u64; 4],
}

impl Xoshiro256PlusPlus {
    #[inline]
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut state = [0_u64; 4];
        for item in &mut state {
            *item = sm.next_u64();
        }

        if state.iter().all(|&x| x == 0) {
            state[0] = 1;
        }

        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[0].wrapping_add(self.state[3]))
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        x as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    #[inline]
    pub fn next_f64_pair(&mut self) -> (f64, f64) {
        let a = self.next_f64();
        let b = self.next_f64();
        (a, b)
    }
}

/// Generate `n` standard normal samples directly into `out` buffer using Philox4x32.
#[inline]
pub fn fill_normals(rng: &mut Philox4x32, out: &mut [f64]) {
    for v in out.iter_mut() {
        let u = rng.next_f64();
        *v = beasley_springer_moro_inv_cdf(uniform_open01(u));
    }
}

// ---------------------------------------------------------------------------
// PCG64 (retained as a FastRngKind variant)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULTIPLIER: u128 = 47026247687942121848144207491837523525;

    #[inline]
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let state_hi = sm.next_u64() as u128;
        let state_lo = sm.next_u64() as u128;
        let stream = sm.next_u64() as u128;

        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };

        rng.state = (state_hi << 64) | state_lo;
        let _ = rng.next_u64();
        rng
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);

        // PCG XSL-RR 128/64 output permutation.
        let xorshifted = ((oldstate >> 64) ^ oldstate) as u64;
        let rot = (oldstate >> 122) as u32;
        xorshifted.rotate_right(rot)
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        x as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

// ---------------------------------------------------------------------------
// FastRng enum dispatch
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum FastRng {
    Philox4x32(Philox4x32),
    Xoshiro256PlusPlus(Xoshiro256Rng),
    Pcg64(Pcg64Rng),
    ThreadRng(ThreadRng),
    StdRng(Box<StdRng>),
}

impl FastRng {
    #[inline]
    pub fn from_seed(kind: FastRngKind, seed: u64) -> Self {
        match kind {
            FastRngKind::Philox4x32 => Self::Philox4x32(Philox4x32::seed_from_u64(seed)),
            FastRngKind::Xoshiro256PlusPlus => {
                Self::Xoshiro256PlusPlus(Xoshiro256Rng::seed_from_u64(seed))
            }
            FastRngKind::Pcg64 => Self::Pcg64(Pcg64Rng::seed_from_u64(seed)),
            FastRngKind::StdRng => Self::StdRng(Box::new(StdRng::seed_from_u64(seed))),
            FastRngKind::ThreadRng => Self::ThreadRng(rand::rng()),
        }
    }

    #[inline]
    pub fn random_f64(&mut self) -> f64 {
        match self {
            Self::Philox4x32(rng) => rng.next_f64(),
            Self::Xoshiro256PlusPlus(rng) => rng.next_f64(),
            Self::Pcg64(rng) => rng.next_f64(),
            Self::ThreadRng(rng) => rng.random::<f64>(),
            Self::StdRng(rng) => rng.random::<f64>(),
        }
    }

    #[inline]
    pub fn random_u64(&mut self) -> u64 {
        match self {
            Self::Philox4x32(rng) => rng.next_u64(),
            Self::Xoshiro256PlusPlus(rng) => rng.next_u64(),
            Self::Pcg64(rng) => rng.next_u64(),
            Self::ThreadRng(rng) => rng.random::<u64>(),
            Self::StdRng(rng) => rng.random::<u64>(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[inline]
pub fn stream_seed(base_seed: u64, stream_index: usize) -> u64 {
    base_seed.wrapping_add((stream_index as u64).wrapping_mul(7_919))
}

#[inline]
pub fn resolve_stream_seed(base_seed: u64, stream_index: usize, reproducible: bool) -> u64 {
    if reproducible {
        stream_seed(base_seed, stream_index)
    } else {
        rand::rng().random::<u64>()
    }
}

/// Maps [0, 1) -> (eps, 1-eps) for safe inverse-CDF transformation.
/// Uses `f64::max/min` which compiles to branchless `maxsd/minsd` on x86.
#[inline(always)]
pub fn uniform_open01(u: f64) -> f64 {
    u.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
}

#[inline(always)]
pub fn sample_standard_normal(rng: &mut FastRng) -> f64 {
    beasley_springer_moro_inv_cdf(uniform_open01(rng.random_f64()))
}

/// Bulk fill a slice with standard normal samples, avoiding per-element function call overhead.
#[inline]
pub fn fill_standard_normals(rng: &mut FastRng, out: &mut [f64]) {
    match rng {
        FastRng::Philox4x32(prng) => {
            for v in out.iter_mut() {
                let u = prng.next_f64();
                *v = beasley_springer_moro_inv_cdf(uniform_open01(u));
            }
        }
        FastRng::Xoshiro256PlusPlus(xrng) => {
            for v in out.iter_mut() {
                let u = xrng.next_f64();
                *v = beasley_springer_moro_inv_cdf(uniform_open01(u));
            }
        }
        _ => {
            for v in out.iter_mut() {
                *v = sample_standard_normal(rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Philox4x32 tests ---

    #[test]
    fn philox_same_seed_reproduces_sequence() {
        let mut a = Philox4x32::seed_from_u64(42);
        let mut b = Philox4x32::seed_from_u64(42);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn philox_different_seeds_diverge() {
        let mut a = Philox4x32::seed_from_u64(42);
        let mut b = Philox4x32::seed_from_u64(43);
        let same = (0..16).all(|_| a.next_u64() == b.next_u64());
        assert!(!same, "different seeds must produce different sequences");
    }

    #[test]
    fn philox_produces_unit_interval() {
        let mut rng = Philox4x32::seed_from_u64(1);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "out of range: {u}");
        }
    }

    #[test]
    fn philox_stateless_determinism() {
        let a = philox_random(100, 50, 0xDEAD_BEEF);
        let b = philox_random(100, 50, 0xDEAD_BEEF);
        assert_eq!(a, b, "same inputs must produce same output");

        let c = philox_random(101, 50, 0xDEAD_BEEF);
        assert_ne!(a, c, "different path_id must produce different output");

        let d = philox_random(100, 51, 0xDEAD_BEEF);
        assert_ne!(a, d, "different step_id must produce different output");
    }

    #[test]
    fn philox_stateless_uniformity() {
        let n = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let x = philox_random(i as u64, 0, 42);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "uniform mean should be ~0.5, got {mean}"
        );
        assert!(
            (var - 1.0 / 12.0).abs() < 0.005,
            "uniform variance should be ~1/12, got {var}"
        );
    }

    #[test]
    fn philox_normal_mean_and_variance() {
        let n = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let z = philox_normal(i as u64, 0, 99);
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "normal mean should be ~0, got {mean}");
        assert!(
            (var - 1.0).abs() < 0.03,
            "normal variance should be ~1, got {var}"
        );
    }

    #[test]
    fn philox_cross_thread_determinism() {
        let seed = 12345_u64;
        let results: Vec<f64> = (0..1000).map(|i| philox_random(i, 0, seed)).collect();
        // Reverse order must give identical per-element values.
        for i in (0..1000).rev() {
            assert_eq!(
                results[i as usize],
                philox_random(i, 0, seed),
                "path {i} mismatch"
            );
        }
    }

    #[test]
    fn philox_via_fastrng_reproduces() {
        let mut a = FastRng::from_seed(FastRngKind::Philox4x32, 77);
        let mut b = FastRng::from_seed(FastRngKind::Philox4x32, 77);
        for _ in 0..128 {
            assert_eq!(a.random_u64(), b.random_u64());
        }
    }

    // --- Existing RNG tests ---

    #[test]
    fn xoshiro_same_seed_reproduces_sequence() {
        let mut a = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 42);
        let mut b = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 42);
        for _ in 0..128 {
            assert_eq!(a.random_u64(), b.random_u64());
        }
    }

    #[test]
    fn pcg_same_seed_reproduces_sequence() {
        let mut a = FastRng::from_seed(FastRngKind::Pcg64, 7);
        let mut b = FastRng::from_seed(FastRngKind::Pcg64, 7);
        for _ in 0..128 {
            assert_eq!(a.random_u64(), b.random_u64());
        }
    }

    #[test]
    fn xoshiro_produces_unit_interval() {
        let mut rng = Xoshiro256Rng::seed_from_u64(1);
        for _ in 0..1000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
