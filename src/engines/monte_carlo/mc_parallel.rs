use rayon::prelude::*;

use crate::core::{ExerciseStyle, OptionType, PricingResult};
use crate::engines::analytic::black_scholes::{bs_delta, bs_gamma, bs_vega};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};
use crate::math::normal_inv_cdf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreeksGridPoint {
    pub spot: f64,
    pub vol: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

#[inline]
fn uniform_open01(u: f64) -> f64 {
    u.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
}

#[inline]
fn payoff(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[inline]
fn split_paths(n_paths: usize, n_chunks: usize) -> Vec<usize> {
    let chunks = n_chunks.max(1);
    let base = n_paths / chunks;
    let rem = n_paths % chunks;
    (0..chunks)
        .map(|i| if i < rem { base + 1 } else { base })
        .filter(|&n| n > 0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn simulate_chunk(
    option_type: OptionType,
    strike: f64,
    spot0: f64,
    dt_drift: f64,
    dt_vol: f64,
    n_steps: usize,
    n_paths: usize,
) -> (f64, f64, usize) {
    // Use thread-id-based seed for reproducibility per chunk
    let chunk_seed = std::thread::current().id().as_u64().get();
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, chunk_seed);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for _ in 0..n_paths {
        let mut spot = spot0;
        for _ in 0..n_steps {
            let z = sample_standard_normal(&mut rng);
            spot *= (dt_drift + dt_vol * z).exp();
            spot = spot.max(1.0e-12);
        }
        let px = payoff(option_type, spot, strike);
        sum += px;
        sum_sq += px * px;
    }

    (sum, sum_sq, n_paths)
}

/// Parallel Monte Carlo pricer for European vanilla options.
///
/// Work is explicitly split into thread-sized chunks and reduced in parallel.
pub fn mc_european_parallel(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    if n_paths == 0 || n_steps == 0 {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if !matches!(instrument.exercise, ExerciseStyle::European) {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if instrument.expiry <= 0.0 {
        return PricingResult {
            price: payoff(instrument.option_type, market.spot, instrument.strike),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let vol = market.vol_for(instrument.strike, instrument.expiry);
    if vol <= 0.0 || !vol.is_finite() {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let dt = instrument.expiry / n_steps as f64;
    let dt_drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * dt;
    let dt_vol = vol * dt.sqrt();
    let discount = (-market.rate * instrument.expiry).exp();

    let chunks = split_paths(n_paths, rayon::current_num_threads());
    let (sum, sum_sq, total_paths) = chunks
        .par_iter()
        .map(|&chunk| {
            simulate_chunk(
                instrument.option_type,
                instrument.strike,
                market.spot,
                dt_drift,
                dt_vol,
                n_steps,
                chunk,
            )
        })
        .reduce(
            || (0.0_f64, 0.0_f64, 0_usize),
            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
        );

    let n = total_paths as f64;
    let mean = sum / n;
    let variance = if total_paths > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", n_steps as f64);
    diagnostics.insert("num_threads", rayon::current_num_threads() as f64);
    diagnostics.insert("vol", vol);

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * (variance / n).sqrt()),
        greeks: None,
        diagnostics,
    }
}

/// Sequential baseline using the same path generation kernel as the parallel pricer.
pub fn mc_european_sequential(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    if n_paths == 0 || n_steps == 0 {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }
    if instrument.expiry <= 0.0 {
        return PricingResult {
            price: payoff(instrument.option_type, market.spot, instrument.strike),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }
    if !matches!(instrument.exercise, ExerciseStyle::European) {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let vol = market.vol_for(instrument.strike, instrument.expiry);
    if vol <= 0.0 || !vol.is_finite() {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }
    let dt = instrument.expiry / n_steps as f64;
    let dt_drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * dt;
    let dt_vol = vol * dt.sqrt();
    let discount = (-market.rate * instrument.expiry).exp();

    let (sum, sum_sq, total_paths) = simulate_chunk(
        instrument.option_type,
        instrument.strike,
        market.spot,
        dt_drift,
        dt_vol,
        n_steps,
        n_paths,
    );
    let n = total_paths as f64;
    let mean = sum / n;
    let variance = if total_paths > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", n_steps as f64);
    diagnostics.insert("vol", vol);

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * (variance / n).sqrt()),
        greeks: None,
        diagnostics,
    }
}

#[inline]
fn greeks_grid_point(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> GreeksGridPoint {
    GreeksGridPoint {
        spot,
        vol,
        delta: bs_delta(option_type, spot, strike, rate, dividend_yield, vol, expiry),
        gamma: bs_gamma(spot, strike, rate, dividend_yield, vol, expiry),
        vega: bs_vega(spot, strike, rate, dividend_yield, vol, expiry),
    }
}

#[inline]
fn build_grid(spots: &[f64], vols: &[f64]) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(spots.len() * vols.len());
    for &spot in spots {
        for &vol in vols {
            points.push((spot, vol));
        }
    }
    points
}

/// Sequential delta/gamma/vega grid for spot-vol pairs.
#[allow(clippy::too_many_arguments)]
pub fn mc_greeks_grid_sequential(
    option_type: OptionType,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    spots: &[f64],
    vols: &[f64],
) -> Vec<GreeksGridPoint> {
    let points = build_grid(spots, vols);
    points
        .iter()
        .map(|&(spot, vol)| {
            greeks_grid_point(option_type, spot, strike, rate, dividend_yield, vol, expiry)
        })
        .collect()
}

/// Parallel delta/gamma/vega grid for spot-vol pairs using Rayon `par_iter`.
#[allow(clippy::too_many_arguments)]
pub fn mc_greeks_grid_parallel(
    option_type: OptionType,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    spots: &[f64],
    vols: &[f64],
) -> Vec<GreeksGridPoint> {
    let points = build_grid(spots, vols);
    points
        .par_iter()
        .map(|&(spot, vol)| {
            greeks_grid_point(option_type, spot, strike, rate, dividend_yield, vol, expiry)
        })
        .collect()
}
