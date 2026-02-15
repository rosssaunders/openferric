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

pub fn geometric_asian_fixed_closed_form(
    option_type: OptionType,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return match option_type {
            OptionType::Call => (s0 - k).max(0.0),
            OptionType::Put => (k - s0).max(0.0),
        };
    }

    // Continuous-time geometric average under GBM.
    let m = s0.ln() + (r - 0.5 * sigma * sigma) * t / 2.0;
    let v = sigma * sigma * t / 3.0;
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
    if maturity <= 0.0 || sigma <= 0.0 {
        return (-r * maturity).exp() * vanilla_payoff(option_type, s0, k);
    }

    let n = observation_times.len() as f64;
    let mean_t = observation_times.iter().sum::<f64>() / n;

    let mut cov_sum = 0.0;
    for &ti in observation_times {
        for &tj in observation_times {
            cov_sum += ti.min(tj);
        }
    }
    let var = sigma * sigma * cov_sum / (n * n);
    let m = s0.ln() + (r - q - 0.5 * sigma * sigma) * mean_t;
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

    #[test]
    fn geometric_asian_mc_converges_to_closed_form() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let closed = geometric_asian_fixed_closed_form(OptionType::Call, s0, k, r, sigma, t);
        let (mc, stderr) = geometric_asian_price_mc(
            OptionType::Call,
            AsianStrike::Fixed(k),
            s0,
            r,
            sigma,
            t,
            252,
            80_000,
            44,
        );

        assert!((mc - closed).abs() <= 2.0 * stderr + 2e-2);
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

        let expected_g = s0 * ((r * 0.5 - sigma * sigma / 12.0) * t).exp();
        let rhs = (-r * t).exp() * (expected_g - k);

        assert_relative_eq!(c - p, rhs, epsilon = 2e-6);
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

        assert!(arith >= geo - 0.05);
    }
}
