//! Module `pricing::asian`.
//!
//! Implements asian workflows with concrete routines such as `geometric_asian_fixed_closed_form`, `geometric_asian_discrete_fixed_closed_form`, `arithmetic_asian_price_mc`, `geometric_asian_price_mc`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `AsianStrike` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::math::normal_cdf;
use crate::mc::{GbmPathGenerator, MonteCarloEngine};
use crate::models::Gbm;
use crate::pricing::OptionType;

#[derive(Debug, Clone, Copy)]
pub enum AsianStrike {
    Fixed(f64),
    Floating,
}

fn arithmetic_average(path: &[f64]) -> f64 {
    path.iter().sum::<f64>() / path.len() as f64
}

fn geometric_average(path: &[f64]) -> f64 {
    (path.iter().map(|s| s.ln()).sum::<f64>() / path.len() as f64).exp()
}

fn asian_payoff(option_type: OptionType, strike: AsianStrike, avg: f64, s_t: f64) -> f64 {
    match strike {
        AsianStrike::Fixed(k) => match option_type {
            OptionType::Call => (avg - k).max(0.0),
            OptionType::Put => (k - avg).max(0.0),
        },
        AsianStrike::Floating => match option_type {
            OptionType::Call => (s_t - avg).max(0.0),
            OptionType::Put => (avg - s_t).max(0.0),
        },
    }
}

fn vanilla_payoff(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[inline]
pub fn geometric_asian_fixed_closed_form(
    option_type: OptionType,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    if t <= 0.0 {
        return match option_type {
            OptionType::Call => (s0 - k).max(0.0),
            OptionType::Put => (k - s0).max(0.0),
        };
    }

    if sigma <= 0.0 {
        // With dS/S = r dt the path is deterministic and its continuously
        // sampled geometric mean is S0 * exp(rT/2).  Returning spot intrinsic
        // here (the former behaviour) omitted both deterministic carry and
        // discounting.
        let geometric_average = s0 * (0.5 * r * t).exp();
        return (-r * t).exp() * vanilla_payoff(option_type, geometric_average, k);
    }

    // Continuous-time geometric average under GBM.
    let sigma_sq = sigma * sigma;
    let m = s0.ln() + (-0.5 * sigma).mul_add(sigma, r) * t / 2.0;
    let v = sigma_sq * t / 3.0;
    let sqrt_v = v.sqrt();

    let d1 = (m - k.ln() + v) / sqrt_v;
    let d2 = d1 - sqrt_v;

    let eg = (m + 0.5 * v).exp();
    let df = (-r * t).exp();

    match option_type {
        OptionType::Call => df * (eg * normal_cdf(d1) - k * normal_cdf(d2)),
        OptionType::Put => df * (k * normal_cdf(-d2) - eg * normal_cdf(-d1)),
    }
}

#[inline]
pub fn geometric_asian_discrete_fixed_closed_form(
    option_type: OptionType,
    s0: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    observation_times: &[f64],
) -> f64 {
    if observation_times.is_empty() {
        return vanilla_payoff(option_type, s0, k);
    }

    let maturity = observation_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    if maturity <= 0.0 {
        return (-r * maturity).exp() * vanilla_payoff(option_type, s0, k);
    }

    let n = observation_times.len() as f64;
    let mean_t = observation_times.iter().sum::<f64>() / n;

    if sigma <= 0.0 {
        // Each fixing is S(t_i)=S0*exp((r-q)t_i), so the geometric
        // average is known exactly from the mean observation time.
        let geometric_average = s0 * ((r - q) * mean_t).exp();
        return (-r * maturity).exp() * vanilla_payoff(option_type, geometric_average, k);
    }

    let mut cov_sum = 0.0;
    for &ti in observation_times {
        for &tj in observation_times {
            cov_sum += ti.min(tj);
        }
    }
    let sigma_sq = sigma * sigma;
    let var = sigma_sq * cov_sum / (n * n);
    let m = s0.ln() + (-0.5 * sigma).mul_add(sigma, r - q) * mean_t;
    let df = (-r * maturity).exp();

    if var <= 0.0 {
        let g = m.exp();
        return df * vanilla_payoff(option_type, g, k);
    }

    let sqrt_v = var.sqrt();
    let d1 = (m - k.ln() + var) / sqrt_v;
    let d2 = d1 - sqrt_v;
    let eg = (m + 0.5 * var).exp();

    match option_type {
        OptionType::Call => df * (eg * normal_cdf(d1) - k * normal_cdf(d2)),
        OptionType::Put => df * (k * normal_cdf(-d2) - eg * normal_cdf(-d1)),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn arithmetic_asian_price_mc(
    option_type: OptionType,
    strike: AsianStrike,
    s0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    paths: usize,
    seed: u64,
) -> (f64, f64) {
    let generator = GbmPathGenerator {
        model: Gbm { mu: r, sigma },
        s0,
        maturity: t,
        steps,
    };
    let engine = MonteCarloEngine::new(paths, seed).with_antithetic(true);
    let discount = (-r * t).exp();

    engine.run(
        &generator,
        |path| {
            let avg = arithmetic_average(path);
            let st = path[path.len() - 1];
            asian_payoff(option_type, strike, avg, st)
        },
        discount,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn geometric_asian_price_mc(
    option_type: OptionType,
    strike: AsianStrike,
    s0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    paths: usize,
    seed: u64,
) -> (f64, f64) {
    let generator = GbmPathGenerator {
        model: Gbm { mu: r, sigma },
        s0,
        maturity: t,
        steps,
    };
    let engine = MonteCarloEngine::new(paths, seed).with_antithetic(true);
    let discount = (-r * t).exp();

    engine.run(
        &generator,
        |path| {
            let avg = geometric_average(path);
            let st = path[path.len() - 1];
            asian_payoff(option_type, strike, avg, st)
        },
        discount,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_mc_reference(label: &str, actual: f64, stderr: f64, reference: f64) {
        let roundoff = 16.0 * f64::EPSILON * reference.abs().max(1.0);
        let tolerance = 4.0 * stderr + roundoff;
        assert!(
            (actual - reference).abs() <= tolerance,
            "{label}: actual={actual:.15}, SciPy={reference:.15}, stderr={stderr:.3e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn zero_vol_geometric_asians_match_exact_deterministic_average() {
        let s0 = 100.0;
        let k = 101.0;
        let r = 0.04;
        let q = 0.015;
        let t = 1.5;

        let continuous_average = s0 * (0.5_f64 * r * t).exp();
        let continuous_discount = (-r * t).exp();
        let call = geometric_asian_fixed_closed_form(OptionType::Call, s0, k, r, 0.0, t);
        let put = geometric_asian_fixed_closed_form(OptionType::Put, s0, k, r, 0.0, t);
        assert_relative_eq!(
            call,
            continuous_discount * (continuous_average - k).max(0.0),
            epsilon = 4.0 * f64::EPSILON * s0
        );
        assert_relative_eq!(
            put,
            continuous_discount * (k - continuous_average).max(0.0),
            epsilon = 4.0 * f64::EPSILON * s0
        );

        let observation_times = [0.25, 0.75, 1.0, t];
        let mean_t = observation_times.iter().sum::<f64>() / observation_times.len() as f64;
        let discrete_average = s0 * ((r - q) * mean_t).exp();
        for (option_type, payoff) in [
            (OptionType::Call, (discrete_average - k).max(0.0)),
            (OptionType::Put, (k - discrete_average).max(0.0)),
        ] {
            let actual = geometric_asian_discrete_fixed_closed_form(
                option_type,
                s0,
                k,
                r,
                q,
                0.0,
                &observation_times,
            );
            let expected = (-r * t).exp() * payoff;
            assert_relative_eq!(actual, expected, epsilon = 4.0 * f64::EPSILON * s0);
        }
    }

    #[test]
    fn one_step_arithmetic_asian_mc_matches_exact_vanilla_reductions() {
        // A one-step path contains [S0, ST].  Therefore
        //   floating call = .5*(ST-S0)+,
        //   fixed-K call = .5*(ST-(2K-S0))+,
        // with the analogous put identities.  The cached references are
        // SciPy 1.17.1 `special.ndtr` evaluations of those Black-Scholes
        // prices, not values generated by OpenFerric.
        let s0 = 100.0;
        let r = 0.03;
        let sigma = 0.25;
        let t = 1.2;
        let paths = 160_000;
        let cases = [
            (
                OptionType::Call,
                AsianStrike::Floating,
                6.279_290_931_894_756_5,
                "floating arithmetic call",
            ),
            (
                OptionType::Put,
                AsianStrike::Floating,
                4.511_305_606_050_914_5,
                "floating arithmetic put",
            ),
            (
                OptionType::Call,
                AsianStrike::Fixed(102.0),
                5.374_229_885_719_7,
                "fixed arithmetic call",
            ),
            (
                OptionType::Put,
                AsianStrike::Fixed(102.0),
                5.535_525_146_842_101,
                "fixed arithmetic put",
            ),
        ];

        for (option_type, strike, reference, label) in cases {
            let (actual, stderr) =
                arithmetic_asian_price_mc(option_type, strike, s0, r, sigma, t, 1, paths, 7_104);
            assert_mc_reference(label, actual, stderr, reference);
        }
    }

    #[test]
    fn floating_geometric_asian_mc_matches_joint_lognormal_reference() {
        let s0 = 100.0;
        let r = 0.03;
        let sigma = 0.25;
        let t = 1.2;
        let steps = 64;

        // SciPy 1.17.1 analytic joint-lognormal exchange-option evaluation.
        // The geometric average contains all 65 path points, including S0.
        // For X=ln(ST), Y=ln(G), the reference evaluates
        // E[e^-rT (e^X-e^Y)+] and its put counterpart from their joint
        // normal means, variances, and covariance.
        for (option_type, reference, label) in [
            (
                OptionType::Call,
                7.477_938_248_635_350_5,
                "floating geometric call",
            ),
            (
                OptionType::Put,
                5.072_720_567_462_737,
                "floating geometric put",
            ),
        ] {
            let (actual, stderr) = geometric_asian_price_mc(
                option_type,
                AsianStrike::Floating,
                s0,
                r,
                sigma,
                t,
                steps,
                160_000,
                98_731,
            );
            assert_mc_reference(label, actual, stderr, reference);
        }
    }

    #[test]
    fn geometric_asian_mc_converges_to_closed_form() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let steps = 252;
        // `geometric_asian_price_mc` averages every simulated path point,
        // including S_0.  Match that discrete contract exactly rather than
        // comparing it with the continuous-average approximation.
        let observation_times = (0..=steps)
            .map(|i| t * i as f64 / steps as f64)
            .collect::<Vec<_>>();
        let closed = geometric_asian_discrete_fixed_closed_form(
            OptionType::Call,
            s0,
            k,
            r,
            0.0,
            sigma,
            &observation_times,
        );
        let (mc, stderr) = geometric_asian_price_mc(
            OptionType::Call,
            AsianStrike::Fixed(k),
            s0,
            r,
            sigma,
            t,
            steps,
            80_000,
            44,
        );

        let roundoff = 16.0 * f64::EPSILON * closed.abs().max(1.0);
        assert!(
            (mc - closed).abs() <= 4.0 * stderr + roundoff,
            "discrete geometric Asian mismatch: mc={mc} exact={closed} stderr={stderr}"
        );
    }

    #[test]
    fn geometric_asian_fixed_put_call_parity() {
        let s0 = 100.0;
        let k = 95.0;
        let r = 0.03;
        let sigma = 0.25;
        let t = 1.2;

        let c = geometric_asian_fixed_closed_form(OptionType::Call, s0, k, r, sigma, t);
        let p = geometric_asian_fixed_closed_form(OptionType::Put, s0, k, r, sigma, t);

        // Independent SciPy 1.17.1 evaluation of the continuous geometric
        // average's exact lognormal law.  Parity below remains a supplemental
        // identity rather than the sole pricing oracle.
        assert_relative_eq!(c, 9.410_685_737_812_25, epsilon = 2.0e-12);
        assert_relative_eq!(p, 3.447_346_735_023_272_5, epsilon = 2.0e-12);

        let expected_g = s0 * ((r * 0.5 - sigma * sigma / 12.0) * t).exp();
        let rhs = (-r * t).exp() * (expected_g - k);

        assert_relative_eq!(
            c - p,
            rhs,
            epsilon = 64.0 * f64::EPSILON * rhs.abs().max(1.0)
        );
    }

    #[test]
    fn arithmetic_call_is_above_geometric_call_for_fixed_strike() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.02;
        let sigma = 0.2;
        let t = 1.0;

        let (arith, _) = arithmetic_asian_price_mc(
            OptionType::Call,
            AsianStrike::Fixed(k),
            s0,
            r,
            sigma,
            t,
            128,
            30_000,
            11,
        );
        let (geo, _) = geometric_asian_price_mc(
            OptionType::Call,
            AsianStrike::Fixed(k),
            s0,
            r,
            sigma,
            t,
            128,
            30_000,
            11,
        );

        // The two calls use identical seeded antithetic paths.  AM >= GM on
        // every positive path and the fixed-strike call payoff is monotone,
        // so this is a pathwise identity, not a statistical comparison.
        let roundoff = 16.0 * f64::EPSILON * arith.abs().max(geo.abs()).max(1.0);
        assert!(
            arith + roundoff >= geo,
            "pathwise AM-GM ordering failed: arithmetic={arith} geometric={geo}"
        );
    }
}
