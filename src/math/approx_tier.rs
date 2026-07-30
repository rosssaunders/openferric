//! Adaptive approximation tiering for SIMD math functions.
//!
//! Selects between accuracy levels for exp() and inverse CDF approximants.
//! The default is accuracy-first because approximation error is systematic:
//! antithetic variates, control variates, and nearly deterministic payoffs can
//! reduce sampling noise below the fast approximant's bias.
//!
//! ## Tiers
//!
//! - **High** (risk/PnL reporting): degree-11 exp, full-precision inverse CDF.
//!   Dense differential tests bound exp relative error by `2e-14` over the
//!   practical finite range `[-700, 700]`.
//!
//! - **Fast** (explicit throughput opt-in): degree-7 exp with relative
//!   error bounded by `8e-9` over the same range. It saves four polynomial
//!   FMAs and is appropriate only when a caller has an explicit numerical
//!   error budget showing MC noise dominates approximation error.
//!
//! ## Selection logic
//!
//! MC standard error = O(σ / √N), but σ can be arbitrarily small and variance
//! reduction can remove most stochastic error. Path count alone therefore
//! cannot justify `Fast`; automatic selection uses `High`.

/// Accuracy tier for SIMD approximants (exp, inverse CDF).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AccuracyTier {
    /// High-accuracy approximants: degree-11 exp (at most `2e-14` relative
    /// error over `[-700, 700]`) and full-precision inverse CDF.
    #[default]
    High,
    /// Fast approximants: degree-7 exp (at most `8e-9` relative error over
    /// `[-700, 700]`). Saves four polynomial FMAs; callers must establish that
    /// its systematic error fits their application-specific error budget.
    Fast,
}

impl AccuracyTier {
    /// Select the appropriate accuracy tier for Monte Carlo simulation
    /// based on the number of paths and steps.
    ///
    /// Path count and time-step count do not bound payoff variance or the bias
    /// introduced by an approximation. Automatic selection therefore uses the
    /// high-accuracy tier. Throughput-sensitive callers may explicitly request
    /// [`AccuracyTier::Fast`] after establishing an application-specific error
    /// budget.
    #[inline]
    pub fn for_mc(_num_paths: usize, _num_steps: usize) -> Self {
        AccuracyTier::High
    }

    /// Select tier for analytic (closed-form) computations.
    /// Always returns `High` since there is no stochastic noise to mask
    /// approximation error.
    #[inline]
    pub fn for_analytic() -> Self {
        AccuracyTier::High
    }

    /// Returns true if this tier uses the fast (degree-7) exp approximant.
    #[inline]
    pub fn uses_fast_exp(self) -> bool {
        self == AccuracyTier::Fast
    }
}

/// Scalar exp() dispatched by accuracy tier.
#[inline]
pub fn tiered_exp(x: f64, tier: AccuracyTier) -> f64 {
    match tier {
        // Both tiers use std::f64::exp for scalar — the tiering only
        // matters for SIMD batch operations where the polynomial degree
        // determines throughput. Scalar exp() is always full precision.
        AccuracyTier::High | AccuracyTier::Fast => x.exp(),
    }
}

/// Dispatch a batch exp operation using the appropriate SIMD tier.
///
/// On x86_64 with AVX2+FMA:
/// - `High`: degree-11 polynomial (11 FMA ops)
/// - `Fast`: degree-7 polynomial (7 FMA ops)
///
/// # Safety
/// Caller must ensure the target SIMD features are available.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tiered_exp_f64x4(
    x: std::arch::x86_64::__m256d,
    tier: AccuracyTier,
) -> std::arch::x86_64::__m256d {
    match tier {
        AccuracyTier::High => unsafe { crate::math::simd_math::exp_f64x4(x) },
        AccuracyTier::Fast => unsafe { crate::math::simd_math::fast_exp_f64x4(x) },
    }
}

/// Dispatch a two-lane NEON exp operation using the selected accuracy tier.
///
/// # Safety
/// The caller must ensure NEON is available on the executing CPU.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn tiered_exp_f64x2(
    x: std::arch::aarch64::float64x2_t,
    tier: AccuracyTier,
) -> std::arch::aarch64::float64x2_t {
    match tier {
        AccuracyTier::High => unsafe { crate::math::simd_neon::simd_exp_f64x2(x) },
        AccuracyTier::Fast => unsafe { crate::math::simd_neon::fast_exp_f64x2(x) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_selection() {
        assert_eq!(AccuracyTier::for_mc(1_000, 100), AccuracyTier::High);
        assert_eq!(AccuracyTier::for_mc(1_000_000, 252), AccuracyTier::High);
        assert_eq!(AccuracyTier::for_analytic(), AccuracyTier::High);
    }

    #[test]
    fn test_default_is_high() {
        assert_eq!(AccuracyTier::default(), AccuracyTier::High);
    }

    #[test]
    fn test_uses_fast_exp() {
        assert!(AccuracyTier::Fast.uses_fast_exp());
        assert!(!AccuracyTier::High.uses_fast_exp());
    }

    #[test]
    fn test_tiered_scalar_exp() {
        let x = 1.5_f64;
        let high = tiered_exp(x, AccuracyTier::High);
        let fast = tiered_exp(x, AccuracyTier::Fast);
        // Scalar path is always full precision
        assert_eq!(high, fast);
        assert!((high - x.exp()).abs() < 1e-15);
    }
}
