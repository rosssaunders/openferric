//! Stochastic model implementation for Slv dynamics.
//!
//! Module openferric::models::slv provides model equations and related calibration/simulation helpers.

use crate::core::{Diagnostics, PricingError, PricingResult};
use crate::engines::monte_carlo::MonteCarloInstrument;
use crate::market::Market;
use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};
use crate::vol::local_vol::{ImpliedVolSurface, dupire_local_vol};

const MIN_SPOT: f64 = 1e-12;
const MIN_VARIANCE: f64 = 1e-12;
const MIN_LEVERAGE: f64 = 1e-4;
const MAX_LEVERAGE: f64 = 10.0;
const CALIBRATION_REFINEMENT_ITERS: usize = 2;

/// Stochastic local volatility parameters with Heston variance dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlvParams {
    /// Initial variance.
    pub v0: f64,
    /// Mean reversion speed.
    pub kappa: f64,
    /// Long-run variance.
    pub theta: f64,
    /// Volatility of variance.
    pub xi: f64,
    /// Correlation between spot and variance Brownian motions.
    pub rho: f64,
}

impl SlvParams {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.v0.is_finite() || self.v0 < 0.0 {
            return Err(PricingError::InvalidInput(
                "slv v0 must be finite and >= 0".to_string(),
            ));
        }
        if !self.kappa.is_finite() || self.kappa < 0.0 {
            return Err(PricingError::InvalidInput(
                "slv kappa must be finite and >= 0".to_string(),
            ));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(PricingError::InvalidInput(
                "slv theta must be finite and >= 0".to_string(),
            ));
        }
        if !self.xi.is_finite() || self.xi < 0.0 {
            return Err(PricingError::InvalidInput(
                "slv xi must be finite and >= 0".to_string(),
            ));
        }
        if !self.rho.is_finite() || self.rho <= -1.0 || self.rho >= 1.0 {
            return Err(PricingError::InvalidInput(
                "slv rho must be finite and in (-1, 1)".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct MarketImpliedVol<'a> {
    market: &'a Market,
}

impl ImpliedVolSurface for MarketImpliedVol<'_> {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.market.vol_for(strike, expiry)
    }
}

/// Time slice of a leverage function sampled on spot points.
#[derive(Debug, Clone, PartialEq)]
pub struct LeverageSlice {
    /// Slice time in years.
    pub time: f64,
    /// Spot grid.
    pub spots: Vec<f64>,
    /// Leverage values on the corresponding spot grid.
    pub leverage: Vec<f64>,
}

impl LeverageSlice {
    fn from_single_point(time: f64, spot: f64, leverage: f64) -> Self {
        Self {
            time,
            spots: vec![spot],
            leverage: vec![leverage.clamp(MIN_LEVERAGE, MAX_LEVERAGE)],
        }
    }

    /// Interpolates leverage at a spot.
    pub fn value(&self, spot: f64) -> f64 {
        if self.spots.is_empty() || self.leverage.is_empty() {
            return 1.0;
        }

        if self.spots.len() == 1 {
            return self.leverage[0].clamp(MIN_LEVERAGE, MAX_LEVERAGE);
        }

        if spot <= self.spots[0] {
            return self.leverage[0].clamp(MIN_LEVERAGE, MAX_LEVERAGE);
        }
        let last = self.spots.len() - 1;
        if spot >= self.spots[last] {
            return self.leverage[last].clamp(MIN_LEVERAGE, MAX_LEVERAGE);
        }

        let mut lo = 0;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.spots[mid] <= spot {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let s0 = self.spots[lo];
        let s1 = self.spots[hi];
        let l0 = self.leverage[lo];
        let l1 = self.leverage[hi];
        let w = if (s1 - s0).abs() > 1e-14 {
            (spot - s0) / (s1 - s0)
        } else {
            0.0
        };

        (l0 + (l1 - l0) * w).clamp(MIN_LEVERAGE, MAX_LEVERAGE)
    }
}

/// Leverage function surface sampled on simulation time steps.
#[derive(Debug, Clone, PartialEq)]
pub struct LeverageSurface {
    /// Total maturity used to build the surface.
    pub maturity: f64,
    /// Number of simulation steps.
    pub n_steps: usize,
    /// Per-step leverage slices (size `n_steps`).
    pub slices: Vec<LeverageSlice>,
}

impl LeverageSurface {
    /// Interpolates leverage in spot and nearest-neighbor in time.
    pub fn value(&self, spot: f64, time: f64) -> f64 {
        if self.slices.is_empty() || self.n_steps == 0 {
            return 1.0;
        }
        if self.maturity <= 0.0 {
            return self.slices[0].value(spot);
        }

        let dt = self.maturity / self.n_steps as f64;
        let mut idx = (time.max(0.0) / dt).floor() as usize;
        if idx >= self.slices.len() {
            idx = self.slices.len() - 1;
        }
        self.slices[idx].value(spot)
    }

    fn value_at_step(&self, step: usize, spot: f64) -> f64 {
        if self.slices.is_empty() {
            return 1.0;
        }
        let idx = step.min(self.slices.len() - 1);
        self.slices[idx].value(spot)
    }
}

fn gaussian_kernel(u: f64) -> f64 {
    (-0.5 * u * u).exp()
}

/// Nadaraya-Watson conditional mean estimator `E[Y|X=x]`.
pub fn nadaraya_watson_conditional_mean(
    x: f64,
    sample_x: &[f64],
    sample_y: &[f64],
    bandwidth: f64,
    fallback_mean: f64,
) -> f64 {
    if sample_x.is_empty() || sample_x.len() != sample_y.len() {
        return fallback_mean;
    }

    let h = bandwidth.max(1e-8);
    let mut numer = 0.0;
    let mut denom = 0.0;
    for (&sx, &sy) in sample_x.iter().zip(sample_y.iter()) {
        let u = (x - sx) / h;
        if u.abs() > 6.0 {
            continue;
        }
        let w = gaussian_kernel(u);
        numer += w * sy.max(0.0);
        denom += w;
    }

    if denom > 1e-14 {
        numer / denom
    } else {
        fallback_mean
    }
}

fn update_variance(v: f64, params: SlvParams, dt: f64, sqrt_dt: f64, z_v: f64) -> f64 {
    let v_pos = v.max(0.0);
    (v + params.kappa * (params.theta - v_pos) * dt + params.xi * v_pos.sqrt() * sqrt_dt * z_v)
        .max(0.0)
}

fn silverman_bandwidth(samples: &[f64], s_min: f64, s_max: f64) -> f64 {
    if samples.len() <= 1 {
        return 1e-3 * samples.first().copied().unwrap_or(100.0).abs().max(1.0);
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let std = var.max(0.0).sqrt();
    let scale = mean.abs().max(1.0);
    let span = (s_max - s_min).abs();

    let raw = 1.06 * std * n.powf(-0.2);
    let floor = (0.02 * scale).max(span / 200.0).max(1e-6);
    let cap = (0.5 * span).max(floor);

    raw.clamp(floor, cap)
}

fn calibrate_slice(
    spots: &[f64],
    variances: &[f64],
    time: f64,
    market: &Market,
    forward: f64,
) -> LeverageSlice {
    let n = spots.len();
    if n == 0 || n != variances.len() {
        return LeverageSlice::from_single_point(time, forward, 1.0);
    }

    let mean_var = (variances.iter().map(|v| v.max(0.0)).sum::<f64>() / n as f64).max(MIN_VARIANCE);
    let (mut s_min, mut s_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &s in spots {
        s_min = s_min.min(s);
        s_max = s_max.max(s);
    }
    if !s_min.is_finite() || !s_max.is_finite() || (s_max - s_min).abs() < 1e-10 {
        let lv = dupire_local_vol(
            MarketImpliedVol { market },
            forward,
            forward,
            time.max(1e-6),
        );
        let leverage = (lv / mean_var.sqrt()).clamp(MIN_LEVERAGE, MAX_LEVERAGE);
        return LeverageSlice::from_single_point(time, forward, leverage);
    }

    let n_bins = ((n as f64).sqrt().round() as usize).clamp(25, 81);
    let span = s_max - s_min;
    let ds = span / (n_bins.saturating_sub(1).max(1) as f64);
    let bandwidth = silverman_bandwidth(spots, s_min, s_max);

    let mut grid_spots = Vec::with_capacity(n_bins);
    let mut grid_leverage = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let s = s_min + i as f64 * ds;
        let cond_var = nadaraya_watson_conditional_mean(s, spots, variances, bandwidth, mean_var)
            .max(MIN_VARIANCE);
        let lv =
            dupire_local_vol(MarketImpliedVol { market }, forward, s, time.max(1e-6)).max(1e-8);
        let lev = (lv / cond_var.sqrt()).clamp(MIN_LEVERAGE, MAX_LEVERAGE);
        grid_spots.push(s);
        grid_leverage.push(lev);
    }

    LeverageSlice {
        time,
        spots: grid_spots,
        leverage: grid_leverage,
    }
}

/// Calibrates the leverage function surface using a particle method.
pub fn calibrate_leverage_surface(
    market: &Market,
    params: SlvParams,
    maturity: f64,
    n_particles: usize,
    n_steps: usize,
) -> Result<LeverageSurface, PricingError> {
    params.validate()?;
    if maturity < 0.0 || !maturity.is_finite() {
        return Err(PricingError::InvalidInput(
            "slv maturity must be finite and >= 0".to_string(),
        ));
    }
    if n_particles == 0 {
        return Err(PricingError::InvalidInput(
            "slv n_particles must be > 0".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(PricingError::InvalidInput(
            "slv n_steps must be > 0".to_string(),
        ));
    }

    let dt = if maturity > 0.0 {
        maturity / n_steps as f64
    } else {
        0.0
    };
    let sqrt_dt = dt.sqrt();
    let mu = market.rate - market.dividend_yield;
    let rho_perp = (1.0 - params.rho * params.rho).max(0.0).sqrt();
    let forward = market.spot.max(MIN_SPOT);

    let simulate_and_calibrate = |surface: Option<&LeverageSurface>, seed: u64| {
        let mut slices = Vec::with_capacity(n_steps);
        let mut spots = vec![market.spot.max(MIN_SPOT); n_particles];
        let mut variances = vec![params.v0.max(0.0); n_particles];
        slices.push(calibrate_slice(&spots, &variances, 0.0, market, forward));
        if n_steps == 1 {
            return slices;
        }

        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
        for step in 1..n_steps {
            for i in 0..n_particles {
                let z1 = sample_standard_normal(&mut rng);
                let z2 = sample_standard_normal(&mut rng);
                let z_v = z1;
                let z_s = params.rho * z1 + rho_perp * z2;

                let v_pos = variances[i].max(0.0);
                variances[i] = update_variance(variances[i], params, dt, sqrt_dt, z_v);
                let sigma = if let Some(ls) = surface {
                    ls.value_at_step(step - 1, spots[i]) * v_pos.sqrt()
                } else {
                    v_pos.sqrt()
                };

                spots[i] *= ((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z_s).exp();
                spots[i] = spots[i].max(MIN_SPOT);
            }

            let t = step as f64 * dt;
            slices.push(calibrate_slice(&spots, &variances, t, market, forward));
        }

        slices
    };

    // First pass: pure stochastic volatility particles.
    let mut surface = LeverageSurface {
        maturity,
        n_steps,
        slices: simulate_and_calibrate(None, 0x5A17_0C4A_u64),
    };

    // Refinement: fixed-point updates under current SLV leverage.
    for iter in 0..CALIBRATION_REFINEMENT_ITERS {
        let seed = 0x6B5F_11E9_u64.wrapping_add(iter as u64 * 17);
        surface.slices = simulate_and_calibrate(Some(&surface), seed);
    }

    Ok(surface)
}

fn price_with_leverage_surface<I: MonteCarloInstrument>(
    instrument: &I,
    market: &Market,
    params: SlvParams,
    leverage_surface: &LeverageSurface,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    let maturity = instrument.maturity();
    let dt = maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let discount = (-market.rate * maturity).exp();
    let mu = market.rate - market.dividend_yield;
    let rho_perp = (1.0 - params.rho * params.rho).max(0.0).sqrt();

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 0xC0DE_0013_u64);
    let mut path = vec![0.0; n_steps + 1];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let mut s = market.spot.max(MIN_SPOT);
        let mut v = params.v0.max(0.0);
        path[0] = s;

        for step in 0..n_steps {
            let z1 = sample_standard_normal(&mut rng);
            let z2 = sample_standard_normal(&mut rng);
            let z_v = z1;
            let z_s = params.rho * z1 + rho_perp * z2;

            let v_pos = v.max(0.0);
            let leverage = leverage_surface.value_at_step(step, s);
            let sigma = leverage * v_pos.sqrt();
            let inst_var = sigma * sigma;

            s *= ((mu - 0.5 * inst_var) * dt + sigma * sqrt_dt * z_s).exp();
            s = s.max(MIN_SPOT);
            v = update_variance(v, params, dt, sqrt_dt, z_v);
            path[step + 1] = s;
        }

        let pv = discount * instrument.payoff_from_path(&path);
        sum += pv;
        sum_sq += pv * pv;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let var = if n_paths > 1 {
        ((sum_sq - n * mean * mean) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };
    let stderr = (var / n).sqrt();

    let atm_local_vol = dupire_local_vol(
        MarketImpliedVol { market },
        market.spot,
        market.spot,
        dt.max(1e-6),
    );
    let atm_leverage = leverage_surface.value(market.spot, 0.0);
    let mut diagnostics = Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", n_steps as f64);
    diagnostics.insert("vol", atm_local_vol);
    diagnostics.insert("vol_adj", atm_leverage);
    diagnostics.insert("var_of_var", params.xi * params.xi);

    PricingResult {
        price: mean,
        stderr: Some(stderr),
        greeks: None,
        diagnostics,
    }
}

/// Monte Carlo SLV pricing with explicit validation errors.
pub fn slv_mc_price_checked<I: MonteCarloInstrument>(
    instrument: &I,
    market: &Market,
    params: SlvParams,
    n_particles: usize,
    n_steps: usize,
) -> Result<PricingResult, PricingError> {
    instrument.validate_for_mc()?;
    params.validate()?;

    if n_particles == 0 {
        return Err(PricingError::InvalidInput(
            "slv n_particles must be > 0".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(PricingError::InvalidInput(
            "slv n_steps must be > 0".to_string(),
        ));
    }

    let maturity = instrument.maturity();
    if !maturity.is_finite() || maturity < 0.0 {
        return Err(PricingError::InvalidInput(
            "instrument maturity must be finite and >= 0".to_string(),
        ));
    }
    if maturity == 0.0 {
        return Ok(PricingResult {
            price: instrument.payoff_from_path(&[market.spot]),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: Diagnostics::new(),
        });
    }

    let leverage_surface =
        calibrate_leverage_surface(market, params, maturity, n_particles, n_steps)?;
    Ok(price_with_leverage_surface(
        instrument,
        market,
        params,
        &leverage_surface,
        n_particles,
        n_steps,
    ))
}

/// Monte Carlo SLV pricing.
pub fn slv_mc_price<I: MonteCarloInstrument>(
    instrument: &I,
    market: &Market,
    params: SlvParams,
    n_particles: usize,
    n_steps: usize,
) -> PricingResult {
    match slv_mc_price_checked(instrument, market, params, n_particles, n_steps) {
        Ok(result) => result,
        Err(_) => PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: Diagnostics::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptionType;
    use crate::core::{BarrierDirection, BarrierStyle};
    use crate::instruments::{BarrierOption, VanillaOption};
    use crate::market::VolSurface;
    use crate::pricing::european::black_scholes_price;
    use crate::vol::implied::implied_vol;

    #[derive(Debug, Clone)]
    struct SmileSurface {
        base: f64,
        spot: f64,
        skew: f64,
    }

    impl VolSurface for SmileSurface {
        fn vol(&self, strike: f64, expiry: f64) -> f64 {
            let x = (strike / self.spot).ln();
            let t = expiry.max(1e-6);
            (self.base + self.skew * x * x + 0.01 * (t - 1.0)).max(0.05)
        }
    }

    fn heston_mc_price<I: MonteCarloInstrument>(
        instrument: &I,
        market: &Market,
        params: SlvParams,
        n_paths: usize,
        n_steps: usize,
        seed: u64,
    ) -> f64 {
        let maturity = instrument.maturity();
        let dt = maturity / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let discount = (-market.rate * maturity).exp();
        let mu = market.rate - market.dividend_yield;
        let rho_perp = (1.0 - params.rho * params.rho).max(0.0).sqrt();

        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
        let mut path = vec![0.0; n_steps + 1];
        let mut sum = 0.0;
        for _ in 0..n_paths {
            let mut s = market.spot;
            let mut v = params.v0.max(0.0);
            path[0] = s;
            for step in 0..n_steps {
                let z1 = sample_standard_normal(&mut rng);
                let z2 = sample_standard_normal(&mut rng);
                let z_v = z1;
                let z_s = params.rho * z1 + rho_perp * z2;

                let v_pos = v.max(0.0);
                let sigma = v_pos.sqrt();
                s *= ((mu - 0.5 * v_pos) * dt + sigma * sqrt_dt * z_s).exp();
                s = s.max(MIN_SPOT);
                v = update_variance(v, params, dt, sqrt_dt, z_v);
                path[step + 1] = s;
            }
            sum += discount * instrument.payoff_from_path(&path);
        }
        sum / n_paths as f64
    }

    fn local_vol_mc_price<I: MonteCarloInstrument>(
        instrument: &I,
        market: &Market,
        n_paths: usize,
        n_steps: usize,
        seed: u64,
    ) -> f64 {
        let maturity = instrument.maturity();
        let dt = maturity / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let discount = (-market.rate * maturity).exp();
        let mu = market.rate - market.dividend_yield;

        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
        let mut path = vec![0.0; n_steps + 1];
        let mut sum = 0.0;
        for _ in 0..n_paths {
            let mut s = market.spot;
            path[0] = s;
            for step in 0..n_steps {
                let z = sample_standard_normal(&mut rng);
                let t = (step as f64 * dt).max(1e-6);
                let sigma = dupire_local_vol(MarketImpliedVol { market }, market.spot, s, t);
                let var = sigma * sigma;
                s *= ((mu - 0.5 * var) * dt + sigma * sqrt_dt * z).exp();
                s = s.max(MIN_SPOT);
                path[step + 1] = s;
            }
            sum += discount * instrument.payoff_from_path(&path);
        }
        sum / n_paths as f64
    }

    #[test]
    fn slv_zero_vol_of_vol_reduces_to_pure_local_vol() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.01)
            .dividend_yield(0.0)
            .vol_surface(Box::new(SmileSurface {
                base: 0.2,
                spot: 100.0,
                skew: 0.15,
            }))
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let params = SlvParams {
            v0: 1.0,
            kappa: 1.5,
            theta: 1.0,
            xi: 0.0,
            rho: -0.4,
        };

        let slv = slv_mc_price(&option, &market, params, 4_000, 48).price;
        let lv = local_vol_mc_price(&option, &market, 4_000, 48, 19);
        let rel = ((slv - lv) / lv.max(1e-8)).abs();
        assert!(rel < 0.08, "slv={slv}, lv={lv}, rel={rel}");
    }

    #[test]
    fn slv_flat_local_vol_reduces_to_heston() {
        let sigma = 0.2;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.02)
            .dividend_yield(0.0)
            .flat_vol(sigma)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let params = SlvParams {
            v0: sigma * sigma,
            kappa: 2.2,
            theta: sigma * sigma,
            xi: 0.45,
            rho: -0.6,
        };

        let slv = slv_mc_price(&option, &market, params, 6_000, 64).price;
        let heston = heston_mc_price(&option, &market, params, 6_000, 64, 77);
        let rel = ((slv - heston) / heston.max(1e-8)).abs();
        assert!(rel < 0.08, "slv={slv}, heston={heston}, rel={rel}");
    }

    #[test]
    fn slv_european_calibration_quality_within_half_percent() {
        let sigma = 0.24;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.01)
            .dividend_yield(0.0)
            .flat_vol(sigma)
            .build()
            .expect("valid market");
        let params = SlvParams {
            v0: sigma * sigma,
            kappa: 1.8,
            theta: sigma * sigma,
            xi: 0.7,
            rho: -0.5,
        };

        let maturity = 1.0;
        let n_paths = 8_000;
        let n_steps = 64;
        let leverage = calibrate_leverage_surface(&market, params, maturity, n_paths, n_steps)
            .expect("calibration succeeds");

        for &k in &[80.0, 100.0, 120.0] {
            let option = VanillaOption::european_call(k, maturity);
            let result =
                price_with_leverage_surface(&option, &market, params, &leverage, n_paths, n_steps);
            let iv = implied_vol(
                OptionType::Call,
                market.spot,
                k,
                market.rate,
                maturity,
                result.price,
                1e-10,
                128,
            )
            .expect("implied vol solve");

            let target = market.vol_for(k, maturity);
            let rel = ((iv - target) / target).abs();
            assert!(
                rel <= 0.005,
                "strike={k}, iv={iv}, target={target}, rel={rel}"
            );
        }
    }

    #[test]
    fn leverage_is_near_one_when_local_and_stochastic_vol_match_atm() {
        let sigma = 0.2;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.0)
            .dividend_yield(0.0)
            .flat_vol(sigma)
            .build()
            .expect("valid market");
        let params = SlvParams {
            v0: sigma * sigma,
            kappa: 2.0,
            theta: sigma * sigma,
            xi: 0.5,
            rho: -0.3,
        };

        let surface =
            calibrate_leverage_surface(&market, params, 1.0, 4_000, 40).expect("calibration");
        let lev = surface.value(100.0, 0.25);
        assert!(
            (lev - 1.0).abs() <= 0.15,
            "expected leverage near 1, got {lev}"
        );
    }

    #[test]
    fn barrier_price_under_slv_differs_from_lv_and_sv() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.01)
            .dividend_yield(0.0)
            .vol_surface(Box::new(SmileSurface {
                base: 0.2,
                spot: 100.0,
                skew: 0.25,
            }))
            .build()
            .expect("valid market");
        let option = BarrierOption::builder()
            .call()
            .strike(100.0)
            .expiry(1.0)
            .up_and_out(112.0)
            .rebate(0.0)
            .build()
            .expect("valid barrier");
        let params = SlvParams {
            v0: 0.04,
            kappa: 1.7,
            theta: 0.04,
            xi: 0.9,
            rho: -0.7,
        };

        let n_paths = 6_000;
        let n_steps = 64;
        let slv = slv_mc_price(&option, &market, params, n_paths, n_steps).price;
        let lv = local_vol_mc_price(&option, &market, n_paths, n_steps, 2);
        let sv = heston_mc_price(&option, &market, params, n_paths, n_steps, 3);

        assert!(
            (slv - lv).abs() > 0.05,
            "expected SLV to differ from LV: slv={slv}, lv={lv}"
        );
        assert!(
            (slv - sv).abs() > 0.05,
            "expected SLV to differ from SV: slv={slv}, sv={sv}"
        );
    }

    #[test]
    fn slv_prices_flat_surface_close_to_black_scholes() {
        let sigma = 0.2;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.0)
            .flat_vol(sigma)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let params = SlvParams {
            v0: sigma * sigma,
            kappa: 1.6,
            theta: sigma * sigma,
            xi: 0.6,
            rho: -0.4,
        };

        let result = slv_mc_price(&option, &market, params, 6_000, 64);
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.03, sigma, 1.0);
        let rel = ((result.price - bs) / bs).abs();
        assert!(rel <= 0.08, "slv={}, bs={}, rel={rel}", result.price, bs);
    }

    #[test]
    fn barrier_payoff_path_logic_is_consistent() {
        let barrier = BarrierOption::builder()
            .put()
            .strike(100.0)
            .expiry(1.0)
            .down_and_in(85.0)
            .rebate(0.0)
            .build()
            .expect("valid barrier option");

        let hit_path = [100.0, 92.0, 84.0, 90.0];
        let no_hit_path = [100.0, 98.0, 95.0, 94.0];

        let hit_payoff = barrier.payoff_from_path(&hit_path);
        let no_hit_payoff = barrier.payoff_from_path(&no_hit_path);
        assert!(hit_payoff >= 0.0);
        assert_eq!(no_hit_payoff, 0.0);

        let out_barrier = BarrierOption {
            option_type: OptionType::Put,
            strike: 100.0,
            expiry: 1.0,
            barrier: crate::core::BarrierSpec {
                direction: BarrierDirection::Down,
                style: BarrierStyle::Out,
                level: 85.0,
                rebate: 1.0,
            },
        };
        let out_hit = out_barrier.payoff_from_path(&hit_path);
        let out_no_hit = out_barrier.payoff_from_path(&no_hit_path);
        assert_eq!(out_hit, 1.0);
        assert!(out_no_hit >= 0.0);
    }
}
