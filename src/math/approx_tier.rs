//! Adaptive approximation tiering for SIMD math functions.
//!
//! Selects between accuracy levels for exp() and inverse CDF approximants
//! based on the target Monte Carlo standard error budget. When stochastic
//! MC error dominates approximation error by 10x+, cheaper (lower-degree)
//! minimax approximants are used automatically.
//!
//! ## Tiers
//!
//! - **High** (risk/PnL reporting): degree-11 exp, full-precision inverse CDF.
//!   Max relative error ~1 ULP. Use for analytic pricing, Greeks, risk reports.
//!
//! - **Fast** (scenario sweeps, MC simulation): degree-7 minimax exp (~2e-10
//!   relative error). ~2x faster exp(). Appropriate when MC noise is orders
//!   of magnitude larger than approximation error.
//!
//! ## Selection logic
//!
//! MC standard error = O(σ / √N). Even at N = 10^8 the MC error (~10^-4)
//! dwarfs the fast-exp error (~2e-10). The tier selection is therefore
//! conservative: `Fast` is used for any MC simulation, `High` for analytic
//! or precision-critical computations.

/// Accuracy tier for SIMD approximants (exp, inverse CDF).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AccuracyTier {
    /// High-accuracy approximants: degree-11 exp (~1 ULP), full-precision
    /// inverse CDF. Use for analytic pricing, Greeks, and risk reporting.
    High,
    /// Fast approximants: degree-7 minimax exp (~2e-10 relative error).
    /// ~2x faster for exp(). Sufficient for Monte Carlo path generation
    /// where stochastic noise dominates approximation error.
    #[default]
    Fast,
}

impl AccuracyTier {
    /// Select the appropriate accuracy tier for Monte Carlo simulation
    /// based on the number of paths and steps.
    ///
    /// The MC standard error is O(1/√N). The `Fast` tier exp() error of
    /// ~2e-10 is negligible compared to MC noise at any practical path
    /// count (even 10^8 paths has ~10^-4 MC error). This function always
    /// returns `Fast` for MC use cases, making the selection explicit
    /// rather than implicit.
    ///
    /// Override with `AccuracyTier::High` if the results feed directly
    /// into risk reports or regulatory calculations where every ULP matters.
    #[inline]
    pub fn for_mc(num_paths: usize, _num_steps: usize) -> Self {
        // MC standard error at N paths ≈ σ/√N.
        // Even at N=10^8 this is ~10^-4, far above fast-exp error of 2e-10.
        // Only a vanishingly small path count would make High worthwhile,
        // but at that point the MC estimate itself is meaningless.
        let _ = num_paths; // All MC use cases can safely use Fast tier.
        AccuracyTier::Fast
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
/// - `Fast`: degree-7 minimax polynomial (7 FMA ops, ~2x faster)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_selection() {
        assert_eq!(AccuracyTier::for_mc(1_000, 100), AccuracyTier::Fast);
        assert_eq!(AccuracyTier::for_mc(1_000_000, 252), AccuracyTier::Fast);
        assert_eq!(AccuracyTier::for_analytic(), AccuracyTier::High);
    }

    #[test]
    fn test_default_is_fast() {
        assert_eq!(AccuracyTier::default(), AccuracyTier::Fast);
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
