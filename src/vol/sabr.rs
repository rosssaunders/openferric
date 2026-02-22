//! Module `vol::sabr`.
//!
//! Implements sabr workflows with concrete routines such as `fit_sabr`.
//!
//! References: Hagan et al. (2002), Hull (11th ed.) Ch. 18, SABR asymptotic volatility formula around Eq. (A.69).
//!
//! Key types and purpose: `SabrParams` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
}

impl SabrParams {
    pub fn implied_vol(&self, f: f64, k: f64, t: f64) -> f64 {
        if t <= 0.0 || f <= 0.0 || k <= 0.0 {
            return 0.0;
        }

        let alpha = self.alpha.max(1e-12);
        let beta = self.beta.clamp(0.0, 1.0);
        let rho = self.rho.clamp(-0.999, 0.999);
        let nu = self.nu.max(0.0);

        let one_minus_beta = 1.0 - beta;

        let time_factor = |fk_pow: f64| {
            1.0 + (((one_minus_beta * one_minus_beta) / 24.0) * (alpha * alpha) / fk_pow
                + (rho * beta * nu * alpha) / (4.0 * fk_pow.sqrt())
                + ((2.0 - 3.0 * rho * rho) / 24.0) * (nu * nu))
                * t
        };

        if (f - k).abs() <= 1e-14 {
            let f_pow = f.powf(one_minus_beta);
            let fk_pow = f.powf(2.0 * one_minus_beta);
            return (alpha / f_pow) * time_factor(fk_pow);
        }

        let fk = f * k;
        let fk_pow_half = fk.powf(0.5 * one_minus_beta);
        let fk_pow = fk.powf(one_minus_beta);
        let log_fk = (f / k).ln();
        let log_fk2 = log_fk * log_fk;
        let log_fk4 = log_fk2 * log_fk2;

        let z = if alpha > 1e-14 {
            (nu / alpha) * fk_pow_half * log_fk
        } else {
            0.0
        };
        let z_over_xz = z_over_xz(z, rho);

        let denominator = fk_pow_half
            * (1.0
                + (one_minus_beta * one_minus_beta / 24.0) * log_fk2
                + (one_minus_beta.powi(4) / 1920.0) * log_fk4);

        let vol = (alpha / denominator) * z_over_xz * time_factor(fk_pow);
        if vol.is_finite() { vol.max(0.0) } else { 0.0 }
    }
}

fn z_over_xz(z: f64, rho: f64) -> f64 {
    if z.abs() < 1e-8 {
        // Series expansion around z=0.
        1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho * rho) / 12.0) * z * z
    } else {
        let num = (1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho;
        let den = 1.0 - rho;
        let xz = (num / den).ln();
        if xz.abs() > 1e-14 { z / xz } else { 1.0 }
    }
}

fn sabr_objective(
    params: SabrParams,
    forward: f64,
    strikes: &[f64],
    market_vols: &[f64],
    t: f64,
) -> f64 {
    let n = strikes.len().min(market_vols.len());
    if n == 0 {
        return f64::INFINITY;
    }

    let mut mse = 0.0;
    for i in 0..n {
        let model = params.implied_vol(forward, strikes[i], t);
        if !model.is_finite() {
            return f64::INFINITY;
        }
        let err = model - market_vols[i];
        mse += err * err;
    }
    mse / n as f64
}

fn project(mut p: SabrParams) -> SabrParams {
    p.alpha = p.alpha.max(1e-8);
    p.beta = p.beta.clamp(0.0, 1.0);
    p.rho = p.rho.clamp(-0.999, 0.999);
    p.nu = p.nu.max(1e-8);
    p
}

fn lm_refine(
    start: SabrParams,
    forward: f64,
    strikes: &[f64],
    market_vols: &[f64],
    t: f64,
) -> (SabrParams, f64) {
    let n = strikes.len();
    let mut p = project(start);
    let mut obj = sabr_objective(p, forward, strikes, market_vols, t);
    let mut lambda = 1e-3;

    for _ in 0..250 {
        let mut j = DMatrix::<f64>::zeros(n, 3);
        let mut r = DVector::<f64>::zeros(n);

        let eps_alpha = (p.alpha * 1e-4).max(1e-6);
        let eps_rho = 1e-5;
        let eps_nu = (p.nu * 1e-4).max(1e-6);

        for i in 0..n {
            let k = strikes[i];
            let model = p.implied_vol(forward, k, t);
            r[i] = model - market_vols[i];

            let deriv = |pp: SabrParams, pm: SabrParams| {
                let v_plus = project(pp).implied_vol(forward, k, t);
                let v_minus = project(pm).implied_vol(forward, k, t);
                v_plus - v_minus
            };

            let mut pp = p;
            pp.alpha += eps_alpha;
            let mut pm = p;
            pm.alpha -= eps_alpha;
            j[(i, 0)] = deriv(pp, pm) / (2.0 * eps_alpha);

            pp = p;
            pp.rho += eps_rho;
            pm = p;
            pm.rho -= eps_rho;
            j[(i, 1)] = deriv(pp, pm) / (2.0 * eps_rho);

            pp = p;
            pp.nu += eps_nu;
            pm = p;
            pm.nu -= eps_nu;
            j[(i, 2)] = deriv(pp, pm) / (2.0 * eps_nu);
        }

        let jt = j.transpose();
        let mut a = &jt * &j;
        for d in 0..3 {
            a[(d, d)] += lambda;
        }
        let g = &jt * &r;

        let Some(delta) = a.lu().solve(&g) else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
            continue;
        };

        let candidate = project(SabrParams {
            alpha: p.alpha - delta[0],
            beta: p.beta,
            rho: p.rho - delta[1],
            nu: p.nu - delta[2],
        });
        let cand_obj = sabr_objective(candidate, forward, strikes, market_vols, t);

        if cand_obj < obj {
            let improvement = (obj - cand_obj).abs();
            p = candidate;
            obj = cand_obj;
            lambda = (lambda * 0.5).max(1e-8);
            if improvement < 1e-14 || delta.norm() < 1e-10 {
                break;
            }
        } else {
            lambda = (lambda * 2.0).min(1e8);
        }
    }

    (p, obj)
}

pub fn fit_sabr(
    forward: f64,
    strikes: &[f64],
    market_vols: &[f64],
    t: f64,
    beta: f64,
) -> SabrParams {
    let beta = beta.clamp(0.0, 1.0);
    if forward <= 0.0 || t <= 0.0 || strikes.is_empty() || strikes.len() != market_vols.len() {
        return project(SabrParams {
            alpha: 0.2,
            beta,
            rho: 0.0,
            nu: 0.5,
        });
    }

    let atm_idx = strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - forward).abs().total_cmp(&(*b - forward).abs()))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let atm_vol = market_vols[atm_idx].max(1e-4);
    let alpha_guess = atm_vol * forward.powf(1.0 - beta);

    let mut starts = Vec::new();
    for alpha_mult in [0.5, 0.75, 1.0, 1.25, 1.5] {
        for rho in [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5] {
            for nu in [0.15, 0.3, 0.5, 0.8, 1.2] {
                starts.push(project(SabrParams {
                    alpha: alpha_guess * alpha_mult,
                    beta,
                    rho,
                    nu,
                }));
            }
        }
    }
    starts.sort_by(|a, b| {
        sabr_objective(*a, forward, strikes, market_vols, t).total_cmp(&sabr_objective(
            *b,
            forward,
            strikes,
            market_vols,
            t,
        ))
    });
    starts.truncate(8);

    let mut best = starts[0];
    let mut best_obj = sabr_objective(best, forward, strikes, market_vols, t);

    for start in starts {
        let (p, obj) = lm_refine(start, forward, strikes, market_vols, t);
        if obj < best_obj {
            best = project(p);
            best_obj = obj;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sabr_hagan_reference_case_matches_expected_values() {
        let params = SabrParams {
            alpha: 0.3,
            beta: 0.5,
            rho: -0.4,
            nu: 0.8,
        };
        let forward = 0.04;
        let t = 5.0;

        let strikes = [
            0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08,
        ];
        let expected = [
            2.380257906011173,
            2.107269276162069,
            1.925_635_528_505_62,
            1.791378193429727,
            1.686220961919487,
            1.600827072338926,
            1.529_781_25,
            1.469656043231035,
            1.418137287693584,
            1.334774783651072,
            1.221122758700629,
        ];

        for (k, exp) in strikes.iter().zip(expected.iter()) {
            let vol = params.implied_vol(forward, *k, t);
            assert_relative_eq!(vol, *exp, epsilon = 1e-12);
        }
    }

    #[test]
    fn sabr_atm_and_near_atm_are_consistent() {
        let params = SabrParams {
            alpha: 0.3,
            beta: 0.5,
            rho: -0.4,
            nu: 0.8,
        };
        let f = 0.04;
        let t = 5.0;

        let atm = params.implied_vol(f, f, t);
        let near = params.implied_vol(f, f * (1.0 + 1e-10), t);
        assert_relative_eq!(atm, near, epsilon = 1e-8);
    }

    #[test]
    fn fit_sabr_recovers_synthetic_parameters() {
        let true_params = SabrParams {
            alpha: 0.24,
            beta: 0.5,
            rho: -0.2,
            nu: 0.7,
        };
        let forward = 0.04;
        let t = 2.5;
        let strikes = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06];
        let market_vols: Vec<f64> = strikes
            .iter()
            .map(|k| true_params.implied_vol(forward, *k, t))
            .collect();

        let fit = fit_sabr(forward, &strikes, &market_vols, t, 0.5);

        let mse = strikes
            .iter()
            .zip(market_vols.iter())
            .map(|(k, mkt)| {
                let model = fit.implied_vol(forward, *k, t);
                let e = model - mkt;
                e * e
            })
            .sum::<f64>()
            / strikes.len() as f64;

        assert_relative_eq!(fit.beta, 0.5, epsilon = 1e-12);
        assert!(mse < 1e-8, "mse={mse}, fit={fit:?}");
    }
}
