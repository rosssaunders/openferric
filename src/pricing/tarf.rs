//! Module `pricing::tarf`.
//!
//! Implements tarf workflows with concrete routines such as `tarf_mc_price`, `tarf_delta`, `tarf_vega`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `TarfPricingResult` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::instruments::tarf::{Tarf, TarfType};

/// Result of TARF Monte Carlo pricing.
#[derive(Debug, Clone)]
pub struct TarfPricingResult {
    /// Present value of the TARF to the buyer.
    pub price: f64,
    /// Standard error of the MC estimate.
    pub std_error: f64,
    /// Average number of fixings before termination.
    pub avg_fixings: f64,
    /// Probability of hitting target profit (early termination).
    pub prob_target_hit: f64,
    /// Probability of hitting KO barrier.
    pub prob_ko_hit: f64,
}

/// Price a TARF via Monte Carlo simulation under GBM dynamics.
///
/// # Conventions
/// * KO direction is type-aware: a `Standard` TARF knocks out when spot fixes
///   at or above `ko_barrier` (upside barrier); a `Decumulator` knocks out
///   when spot fixes at or below it (downside barrier). `f64::INFINITY` means
///   "no KO barrier" for both types.
/// * The KO check runs before the fixing P&L: the breaching fixing pays
///   nothing and all remaining fixings are cancelled.
/// * Target termination uses the full-gain convention: the fixing that
///   reaches `target_profit` pays its full uncapped gain before termination.
///
/// # Arguments
/// * `tarf` - TARF instrument definition
/// * `spot` - Current spot price
/// * `rate` - Risk-free rate (continuous)
/// * `dividend_yield` - Continuous dividend yield
/// * `vol` - Black-Scholes volatility
/// * `num_paths` - Number of MC paths
/// * `seed` - RNG seed
pub fn tarf_mc_price(
    tarf: &Tarf,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    num_paths: usize,
    seed: u64,
) -> Result<TarfPricingResult, String> {
    tarf.validate()?;
    if !spot.is_finite() || spot <= 0.0 {
        return Err("spot must be finite and > 0".to_string());
    }
    if !vol.is_finite() || vol <= 0.0 {
        return Err("vol must be finite and > 0".to_string());
    }
    if num_paths == 0 {
        return Err("num_paths must be > 0".to_string());
    }

    let n_fix = tarf.fixing_times.len();
    let mut rng = StdRng::seed_from_u64(seed);

    // `f64::INFINITY` is the universal "no KO barrier" sentinel for both TARF
    // types. The explicit finiteness guard matters on the decumulator side:
    // without it, `s <= INFINITY` would knock out every path at fixing 1.
    let ko_active = tarf.ko_barrier.is_finite();

    let mut sum_pv = 0.0;
    let mut sum_pv2 = 0.0;
    let mut total_fixings = 0.0;
    let mut target_hits = 0u64;
    let mut ko_hits = 0u64;

    // Pre-compute time intervals
    let mut dts = Vec::with_capacity(n_fix);
    let mut prev_t = 0.0;
    for &t in &tarf.fixing_times {
        dts.push(t - prev_t);
        prev_t = t;
    }

    for _ in 0..num_paths {
        let mut s = spot;
        let mut accumulated_profit = 0.0;
        let mut path_pv = 0.0;
        let mut terminated = false;
        let mut fixings_used = 0usize;

        for (i, &dt) in dts.iter().enumerate() {
            let t = tarf.fixing_times[i];
            let z: f64 = StandardNormal.sample(&mut rng);
            s *= ((rate - dividend_yield - 0.5 * vol * vol) * dt + vol * dt.sqrt() * z).exp();

            // Check KO barrier before the fixing P&L (the breaching fixing
            // pays nothing). Direction is type-aware: Standard knocks out on
            // the upside, Decumulator on the downside.
            let knocked_out = ko_active
                && match tarf.tarf_type {
                    TarfType::Standard => s >= tarf.ko_barrier,
                    TarfType::Decumulator => s <= tarf.ko_barrier,
                };
            if knocked_out {
                ko_hits += 1;
                terminated = true;
                fixings_used = i + 1;
                break;
            }

            // Compute fixing P&L
            let pnl = match tarf.tarf_type {
                TarfType::Standard => {
                    if s >= tarf.strike {
                        // Upside: buyer gains (S - K) * notional
                        (s - tarf.strike) * tarf.notional_per_fixing
                    } else {
                        // Downside: leveraged loss
                        (s - tarf.strike) * tarf.notional_per_fixing * tarf.downside_leverage
                    }
                }
                TarfType::Decumulator => {
                    if s <= tarf.strike {
                        // Decumulator: seller gains (K - S) * notional
                        (tarf.strike - s) * tarf.notional_per_fixing
                    } else {
                        (tarf.strike - s) * tarf.notional_per_fixing * tarf.downside_leverage
                    }
                }
            };

            // Discount and accumulate
            let df = (-rate * t).exp();
            path_pv += pnl * df;

            // Track accumulated profit for target
            if pnl > 0.0 {
                accumulated_profit += pnl;
            }

            // Check target profit (full-gain convention: the full uncapped
            // pnl of this fixing was already added to path_pv above).
            if accumulated_profit >= tarf.target_profit {
                target_hits += 1;
                terminated = true;
                fixings_used = i + 1;
                break;
            }
        }

        if !terminated {
            fixings_used = n_fix;
        }

        sum_pv += path_pv;
        sum_pv2 += path_pv * path_pv;
        total_fixings += fixings_used as f64;
    }

    let n = num_paths as f64;
    let mean = sum_pv / n;
    let variance = (sum_pv2 / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    Ok(TarfPricingResult {
        price: mean,
        std_error,
        avg_fixings: total_fixings / n,
        prob_target_hit: target_hits as f64 / n,
        prob_ko_hit: ko_hits as f64 / n,
    })
}

/// Compute TARF delta via bump-and-reprice.
pub fn tarf_delta(
    tarf: &Tarf,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    num_paths: usize,
    seed: u64,
    bump: f64,
) -> Result<f64, String> {
    let p_up = tarf_mc_price(
        tarf,
        spot * (1.0 + bump),
        rate,
        dividend_yield,
        vol,
        num_paths,
        seed,
    )?;
    let p_down = tarf_mc_price(
        tarf,
        spot * (1.0 - bump),
        rate,
        dividend_yield,
        vol,
        num_paths,
        seed,
    )?;
    Ok((p_up.price - p_down.price) / (2.0 * spot * bump))
}

/// Compute TARF vega via bump-and-reprice.
pub fn tarf_vega(
    tarf: &Tarf,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    num_paths: usize,
    seed: u64,
    bump: f64,
) -> Result<f64, String> {
    let p_up = tarf_mc_price(
        tarf,
        spot,
        rate,
        dividend_yield,
        vol + bump,
        num_paths,
        seed,
    )?;
    let p_down = tarf_mc_price(
        tarf,
        spot,
        rate,
        dividend_yield,
        vol - bump,
        num_paths,
        seed,
    )?;
    Ok((p_up.price - p_down.price) / (2.0 * bump))
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{ContinuousCDF, Normal};

    fn make_standard_tarf() -> Tarf {
        // Weekly fixings for 1 year
        let fixing_times: Vec<f64> = (1..=52).map(|w| w as f64 / 52.0).collect();
        Tarf::standard(
            100.0,   // strike
            1000.0,  // notional per fixing
            120.0,   // KO barrier
            50000.0, // target profit
            2.0,     // 2x downside leverage
            fixing_times,
        )
    }

    fn one_fixing_standard_value(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        maturity: f64,
        notional: f64,
        downside_leverage: f64,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sigma_sqrt_t = volatility * maturity.sqrt();
        let d1 = ((spot / strike).ln()
            + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity)
            / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        let spot_pv = spot * (-dividend_yield * maturity).exp();
        let strike_pv = strike * (-rate * maturity).exp();
        let call = spot_pv * normal.cdf(d1) - strike_pv * normal.cdf(d2);
        let put = strike_pv * normal.cdf(-d2) - spot_pv * normal.cdf(-d1);
        notional * (call - downside_leverage * put)
    }

    #[test]
    fn tarf_price_is_finite() {
        let tarf = make_standard_tarf();
        let result = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 5000, 42).unwrap();
        assert!(result.price.is_finite());
        assert!(result.std_error.is_finite());
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn tarf_avg_fixings_bounded() {
        let tarf = make_standard_tarf();
        let result = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 5000, 42).unwrap();
        assert!(result.avg_fixings > 0.0);
        assert!(result.avg_fixings <= 52.0);
    }

    #[test]
    fn tarf_probabilities_are_valid() {
        let tarf = make_standard_tarf();
        let result = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 5000, 42).unwrap();
        assert!(result.prob_target_hit >= 0.0 && result.prob_target_hit <= 1.0);
        assert!(result.prob_ko_hit >= 0.0 && result.prob_ko_hit <= 1.0);
    }

    #[test]
    fn tarf_higher_vol_changes_price() {
        let tarf = make_standard_tarf();
        let low_vol = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.10, 5000, 42).unwrap();
        let high_vol = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.30, 5000, 42).unwrap();
        // Higher vol should lead to more KO events and different pricing
        assert!((low_vol.price - high_vol.price).abs() > 0.01);
    }

    #[test]
    fn tarf_delta_and_vega_match_one_fixing_black_scholes_decomposition() {
        // With one fixing, no KO and the full-gain target convention, the
        // standard TARF payoff is N * (call - L * put). Compare the public
        // bump APIs against that independent Black-Scholes decomposition at
        // the exact same spot and volatility bump points.
        const N_PATHS: usize = 1_000_000;
        const FOUR_SE_DELTA_BUDGET: f64 = 1.21e-3;
        const FOUR_SE_VEGA_BUDGET: f64 = 5.67e-1;
        let tarf = Tarf::standard(100.0, 1.0, f64::INFINITY, 1.0, 2.0, vec![1.0]);
        let spot = 100.0;
        let rate = 0.03;
        let dividend_yield = 0.01;
        let vol = 0.20;
        let spot_bump = 0.01;
        let vol_bump = 0.01;

        let actual_delta = tarf_delta(
            &tarf,
            spot,
            rate,
            dividend_yield,
            vol,
            N_PATHS,
            42,
            spot_bump,
        )
        .unwrap();
        let actual_vega = tarf_vega(
            &tarf,
            spot,
            rate,
            dividend_yield,
            vol,
            N_PATHS,
            42,
            vol_bump,
        )
        .unwrap();
        let value = |s, sigma| {
            one_fixing_standard_value(
                s,
                tarf.strike,
                rate,
                dividend_yield,
                sigma,
                tarf.fixing_times[0],
                tarf.notional_per_fixing,
                tarf.downside_leverage,
            )
        };
        let delta_reference = (value(spot * (1.0 + spot_bump), vol)
            - value(spot * (1.0 - spot_bump), vol))
            / (2.0 * spot * spot_bump);
        let vega_reference =
            (value(spot, vol + vol_bump) - value(spot, vol - vol_bump)) / (2.0 * vol_bump);

        // Independent SciPy 1.17.1 integration of the paired common-random-
        // number estimators gives SE_delta=3.00206971e-4 and
        // SE_vega=0.141699319 at one million paths.
        assert!(
            (actual_delta - delta_reference).abs() <= FOUR_SE_DELTA_BUDGET,
            "TARF delta={actual_delta} BS decomposition={delta_reference}"
        );
        assert!(
            (actual_vega - vega_reference).abs() <= FOUR_SE_VEGA_BUDGET,
            "TARF vega={actual_vega} BS decomposition={vega_reference}"
        );
    }

    #[test]
    fn path_dependent_tarf_greeks_match_independent_scipy_sobol_references() {
        const N_REPLICATES: usize = 12;
        const PATHS_PER_REPLICATE: usize = 40_000;
        let fixing_times = (1..=12).map(|month| month as f64 / 12.0).collect();
        let tarf = Tarf::standard(100.0, 1.0, 120.0, 20.0, 2.0, fixing_times);
        let mut deltas = [0.0_f64; N_REPLICATES];
        let mut vegas = [0.0_f64; N_REPLICATES];

        for replicate in 0..N_REPLICATES {
            let seed = 41_003 + 65_537 * replicate as u64;
            deltas[replicate] = tarf_delta(
                &tarf,
                100.0,
                0.03,
                0.01,
                0.20,
                PATHS_PER_REPLICATE,
                seed,
                0.01,
            )
            .unwrap();
            vegas[replicate] = tarf_vega(
                &tarf,
                100.0,
                0.03,
                0.01,
                0.20,
                PATHS_PER_REPLICATE,
                seed,
                0.01,
            )
            .unwrap();
        }

        fn mean_and_se<const N: usize>(samples: &[f64; N]) -> (f64, f64) {
            let mean = samples.iter().sum::<f64>() / N as f64;
            let variance = samples
                .iter()
                .map(|sample| (sample - mean).powi(2))
                .sum::<f64>()
                / (N - 1) as f64;
            (mean, (variance / N as f64).sqrt())
        }

        // Independent SciPy 1.17.1 inverse-normal Sobol integration with 24
        // Owen scrambles x 2^17 paths.  The target and KO are both active;
        // central bumps and common paths exactly match the public Greek APIs.
        for (label, samples, reference, reference_se) in [
            (
                "delta",
                &deltas,
                12.410_356_325_863_939,
                6.753_763_687_173_809e-3,
            ),
            (
                "vega",
                &vegas,
                -565.296_753_545_253_2,
                2.600_048_016_445_336e-1,
            ),
        ] {
            let (actual, implementation_se) = mean_and_se(samples);
            let combined_se = implementation_se.hypot(reference_se);
            assert!(
                (actual - reference).abs() <= 4.0 * combined_se,
                "path-dependent TARF {label}: actual={actual} reference={reference} implementation_se={implementation_se} reference_se={reference_se}"
            );
        }
    }

    fn make_decumulator_tarf() -> Tarf {
        // Weekly fixings for 1 year; downside KO barrier below the strike.
        let fixing_times: Vec<f64> = (1..=52).map(|w| w as f64 / 52.0).collect();
        Tarf::decumulator(
            100.0,   // strike
            1000.0,  // notional per fixing
            80.0,    // downside KO barrier
            50000.0, // target profit
            2.0,     // 2x leverage on the losing side
            fixing_times,
        )
    }

    #[test]
    fn tarf_decumulator_differs_from_standard() {
        let std_price = tarf_mc_price(&make_standard_tarf(), 100.0, 0.03, 0.0, 0.15, 5000, 42)
            .unwrap()
            .price;
        let dec_price = tarf_mc_price(&make_decumulator_tarf(), 100.0, 0.03, 0.0, 0.15, 5000, 42)
            .unwrap()
            .price;
        // Standard and decumulator should give different values
        assert!((std_price - dec_price).abs() > 0.01);
    }

    #[test]
    fn tarf_decumulator_does_not_knock_out_immediately() {
        // Regression for the inverted KO direction: the old code applied the
        // upside check `s >= ko_barrier` to decumulators, so a downside
        // barrier at 80 with spot 100 knocked out every path at fixing 1
        // (prob_ko = 1, avg_fixings = 1, price = 0).
        let result =
            tarf_mc_price(&make_decumulator_tarf(), 100.0, 0.03, 0.0, 0.15, 5000, 42).unwrap();
        assert!(result.prob_ko_hit < 0.9);
        assert!(result.avg_fixings > 2.0);
        assert!(result.price.abs() > 1.0);
    }

    #[test]
    fn tarf_rejects_barrier_on_wrong_side() {
        let mut tarf = make_standard_tarf();
        tarf.tarf_type = TarfType::Decumulator; // upside barrier 120 now invalid
        assert!(tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 100, 42).is_err());
    }

    #[test]
    fn one_fixing_tarf_matches_black_scholes_decomposition() {
        // With one fixing and no knock-out, the standard payoff is exactly
        // N * (call - L * put); the decumulator is N * (put - L * call).
        // The cached values were evaluated independently with SciPy 1.17.1's
        // normal CDF for S=K=100, r=3%, q=0, sigma=15%, and T=1.
        let mut tarf = Tarf::standard(100.0, 1_000.0, f64::INFINITY, 1.0, 2.0, vec![1.0]);

        let standard = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 250_000, 91).unwrap();
        let expected_standard = -1_574.194_303_614_27;
        assert!(
            (standard.price - expected_standard).abs() <= 4.0 * standard.std_error,
            "standard one-fixing TARF: price={} expected={} stderr={}",
            standard.price,
            expected_standard,
            standard.std_error
        );

        tarf.tarf_type = TarfType::Decumulator;
        let decumulator = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 250_000, 91).unwrap();
        let expected_decumulator = -10_440.534_239_061_77;
        assert!(
            (decumulator.price - expected_decumulator).abs() <= 4.0 * decumulator.std_error,
            "one-fixing decumulator: price={} expected={} stderr={}",
            decumulator.price,
            expected_decumulator,
            decumulator.std_error
        );
    }
}
