const INV_U64_RANGE: f64 = 1.0 / 18_446_744_073_709_551_616.0;
const HALF_INV_U64: f64 = 0.5 * INV_U64_RANGE;
pub const SOBOL_MAX_DIMENSIONS: usize = 21_201;

#[derive(Debug, Clone)]
pub struct SobolSequence {
    dimensions: usize,
    index: u64,
    x: Vec<u64>,
    directions: Vec<[u64; 64]>,
    scramblers: Vec<u64>,
}

impl SobolSequence {
    pub fn new(dimensions: usize, seed: u64) -> Self {
        assert!(
            (1..=SOBOL_MAX_DIMENSIONS).contains(&dimensions),
            "Sobol dimensions must be in [1, {SOBOL_MAX_DIMENSIONS}]"
        );

        let mut directions = Vec::with_capacity(dimensions);
        let mut scramblers = Vec::with_capacity(dimensions);

        for dim in 0..dimensions {
            directions.push(build_direction_numbers(dim as u64, seed));
            scramblers.push(splitmix64(seed ^ ((dim as u64 + 1) << 32)));
        }

        Self {
            dimensions,
            index: 0,
            x: vec![0_u64; dimensions],
            directions,
            scramblers,
        }
    }

    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Advances the sequence and writes the next point into `out`.
    ///
    /// Returns `false` if the sequence is exhausted (after 2^64 - 1 points).
    /// This avoids the heap allocation that the `Iterator` impl requires.
    #[inline]
    pub fn next_into(&mut self, out: &mut [f64]) -> bool {
        let next_index = self.index.wrapping_add(1);
        if next_index == 0 {
            return false;
        }
        let c = next_index.trailing_zeros() as usize;
        self.index = next_index;

        for dim in 0..self.dimensions {
            self.x[dim] ^= self.directions[dim][c];
            let scrambled = self.x[dim] ^ self.scramblers[dim];
            out[dim] = (scrambled as f64).mul_add(INV_U64_RANGE, HALF_INV_U64);
        }

        true
    }

    /// Generate `n` points in bulk, writing sequentially into a flat buffer.
    /// `out` must have length >= `n * self.dimensions`.
    /// Returns the number of points actually written.
    #[inline]
    pub fn fill_points(&mut self, out: &mut [f64], n: usize) -> usize {
        let dims = self.dimensions;
        let mut count = 0;
        let mut offset = 0;
        while count < n && offset + dims <= out.len() {
            if !self.next_into(&mut out[offset..offset + dims]) {
                break;
            }
            offset += dims;
            count += 1;
        }
        count
    }
}

impl Iterator for SobolSequence {
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut point = vec![0.0_f64; self.dimensions];
        if self.next_into(&mut point) {
            Some(point)
        } else {
            None
        }
    }
}

#[inline]
fn build_direction_numbers(dim: u64, seed: u64) -> [u64; 64] {
    // Dimension 1 uses canonical Sobol direction numbers.
    if dim == 0 {
        let mut v = [0_u64; 64];
        for (j, item) in v.iter_mut().enumerate() {
            *item = 1_u64 << (63 - j);
        }
        return v;
    }

    let mut v = [0_u64; 64];
    for (j, item) in v.iter_mut().enumerate() {
        let hash = splitmix64(seed ^ ((dim + 1) << 40) ^ j as u64);
        let mask = if j == 63 {
            u64::MAX
        } else {
            (1_u64 << (j + 1)) - 1
        };
        let m = (hash | 1) & mask;
        *item = m << (63 - j);
    }
    v
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fast_rng::Xoshiro256Rng;

    #[test]
    fn sobol_points_are_in_unit_interval() {
        let mut seq = SobolSequence::new(8, 42);
        for _ in 0..1_000 {
            let p = seq.next().expect("sequence should continue");
            for u in p {
                assert!((0.0..1.0).contains(&u), "u={u}");
            }
        }
    }

    #[test]
    fn sobol_reproducible_for_same_seed() {
        let mut a = SobolSequence::new(5, 99);
        let mut b = SobolSequence::new(5, 99);

        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn sobol_first_dimension_mean_is_closer_to_half_than_prng() {
        let n = 1_000;

        let sobol_mean = SobolSequence::new(1, 7).take(n).map(|v| v[0]).sum::<f64>() / n as f64;

        let mut rng = Xoshiro256Rng::seed_from_u64(7);
        let prng_mean = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;

        assert!(
            (sobol_mean - 0.5).abs() <= (prng_mean - 0.5).abs(),
            "sobol_mean={sobol_mean} prng_mean={prng_mean}"
        );
    }

    #[test]
    fn supports_large_dimension_count() {
        let mut seq = SobolSequence::new(SOBOL_MAX_DIMENSIONS, 123);
        let point = seq.next().expect("first point should exist");
        assert_eq!(point.len(), SOBOL_MAX_DIMENSIONS);
    }
}
