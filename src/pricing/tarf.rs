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

            // Check KO barrier
            if s >= tarf.ko_barrier {
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

            // Check target profit
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
    let p_up = tarf_mc_price(tarf, spot * (1.0 + bump), rate, dividend_yield, vol, num_paths, seed)?;
    let p_down = tarf_mc_price(tarf, spot * (1.0 - bump), rate, dividend_yield, vol, num_paths, seed)?;
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
    let p_up = tarf_mc_price(tarf, spot, rate, dividend_yield, vol + bump, num_paths, seed)?;
    let p_down = tarf_mc_price(tarf, spot, rate, dividend_yield, vol - bump, num_paths, seed)?;
    Ok((p_up.price - p_down.price) / (2.0 * bump))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_standard_tarf() -> Tarf {
        // Weekly fixings for 1 year
        let fixing_times: Vec<f64> = (1..=52).map(|w| w as f64 / 52.0).collect();
        Tarf::standard(
            100.0,    // strike
            1000.0,   // notional per fixing
            120.0,    // KO barrier
            50000.0,  // target profit
            2.0,      // 2x downside leverage
            fixing_times,
        )
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
    fn tarf_delta_is_finite() {
        let tarf = make_standard_tarf();
        let delta = tarf_delta(&tarf, 100.0, 0.03, 0.0, 0.15, 5000, 42, 0.01).unwrap();
        assert!(delta.is_finite());
    }

    #[test]
    fn tarf_decumulator_differs_from_standard() {
        let mut tarf = make_standard_tarf();
        let std_price = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 5000, 42)
            .unwrap()
            .price;
        tarf.tarf_type = TarfType::Decumulator;
        let dec_price = tarf_mc_price(&tarf, 100.0, 0.03, 0.0, 0.15, 5000, 42)
            .unwrap()
            .price;
        // Standard and decumulator should give different values
        assert!((std_price - dec_price).abs() > 0.01);
    }
}
