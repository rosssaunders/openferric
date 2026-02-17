use rand::rngs::{StdRng, ThreadRng};
use rand::{RngExt, SeedableRng};

use crate::math::fast_norm::beasley_springer_moro_inv_cdf;

pub type Xoshiro256Rng = Xoshiro256PlusPlus;
pub type Pcg64Rng = Pcg64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastRngKind {
    Xoshiro256PlusPlus,
    Pcg64,
    ThreadRng,
    StdRng,
}

impl Default for FastRngKind {
    fn default() -> Self {
        Self::Xoshiro256PlusPlus
    }
}

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
}

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

#[derive(Debug)]
pub enum FastRng {
    Xoshiro256PlusPlus(Xoshiro256Rng),
    Pcg64(Pcg64Rng),
    ThreadRng(ThreadRng),
    StdRng(StdRng),
}

impl FastRng {
    #[inline]
    pub fn from_seed(kind: FastRngKind, seed: u64) -> Self {
        match kind {
            FastRngKind::Xoshiro256PlusPlus => {
                Self::Xoshiro256PlusPlus(Xoshiro256Rng::seed_from_u64(seed))
            }
            FastRngKind::Pcg64 => Self::Pcg64(Pcg64Rng::seed_from_u64(seed)),
            FastRngKind::StdRng => Self::StdRng(StdRng::seed_from_u64(seed)),
            FastRngKind::ThreadRng => Self::ThreadRng(rand::rng()),
        }
    }

    #[inline]
    pub fn random_f64(&mut self) -> f64 {
        match self {
            Self::Xoshiro256PlusPlus(rng) => rng.next_f64(),
            Self::Pcg64(rng) => rng.next_f64(),
            Self::ThreadRng(rng) => rng.random::<f64>(),
            Self::StdRng(rng) => rng.random::<f64>(),
        }
    }

    #[inline]
    pub fn random_u64(&mut self) -> u64 {
        match self {
            Self::Xoshiro256PlusPlus(rng) => rng.next_u64(),
            Self::Pcg64(rng) => rng.next_u64(),
            Self::ThreadRng(rng) => rng.random::<u64>(),
            Self::StdRng(rng) => rng.random::<u64>(),
        }
    }
}

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

/// Maps [0, 1) → (ε, 1−ε) for safe inverse-CDF transformation.
/// Uses `f64::max/min` which compiles to branchless `maxsd/minsd` on x86.
#[inline(always)]
pub fn uniform_open01(u: f64) -> f64 {
    u.max(f64::EPSILON).min(1.0 - f64::EPSILON)
}

#[inline(always)]
pub fn sample_standard_normal(rng: &mut FastRng) -> f64 {
    beasley_springer_moro_inv_cdf(uniform_open01(rng.random_f64()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
