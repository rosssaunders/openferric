use crate::mc::{GbmPathGenerator, MonteCarloEngine};
use crate::models::Gbm;
use crate::pricing::OptionType;
use crate::pricing::european::black_scholes_price;
use crate::math::normal_cdf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierStyle {
    In,
    Out,
}

fn breached_at_start(s0: f64, barrier: f64, direction: BarrierDirection) -> bool {
    match direction {
        BarrierDirection::Up => s0 >= barrier,
        BarrierDirection::Down => s0 <= barrier,
    }
}

fn path_hits_barrier(path: &[f64], barrier: f64, direction: BarrierDirection) -> bool {
    match direction {
        BarrierDirection::Up => path.iter().any(|&s| s >= barrier),
        BarrierDirection::Down => path.iter().any(|&s| s <= barrier),
    }
}

fn vanilla_payoff(option_type: OptionType, s_t: f64, k: f64) -> f64 {
    match option_type {
        OptionType::Call => (s_t - k).max(0.0),
        OptionType::Put => (k - s_t).max(0.0),
    }
}

fn rr_out_price(option_type: OptionType, direction: BarrierDirection, s: f64, k: f64, h: f64, r: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return vanilla_payoff(option_type, s, k);
    }

    let eta = match direction {
        BarrierDirection::Down => 1.0,
        BarrierDirection::Up => -1.0,
    };
    let phi = option_type.sign();

    let st = sigma * t.sqrt();
    let mu = (r - 0.5 * sigma * sigma) / (sigma * sigma);
    let df = (-r * t).exp();

    let x1 = (s / k).ln() / st + (1.0 + mu) * st;
    let x2 = (s / h).ln() / st + (1.0 + mu) * st;
    let y1 = ((h * h) / (s * k)).ln() / st + (1.0 + mu) * st;
    let y2 = (h / s).ln() / st + (1.0 + mu) * st;

    let hs_mu = (h / s).powf(2.0 * mu);
    let hs_mu1 = (h / s).powf(2.0 * (mu + 1.0));

    let a = phi * s * normal_cdf(phi * x1) - phi * k * df * normal_cdf(phi * (x1 - st));
    let b = phi * s * normal_cdf(phi * x2) - phi * k * df * normal_cdf(phi * (x2 - st));
    let c = phi * s * hs_mu1 * normal_cdf(eta * y1) - phi * k * df * hs_mu * normal_cdf(eta * (y1 - st));
    let d = phi * s * hs_mu1 * normal_cdf(eta * y2) - phi * k * df * hs_mu * normal_cdf(eta * (y2 - st));

    let out = match (direction, option_type) {
        (BarrierDirection::Down, OptionType::Call) => {
            if k >= h { a - c } else { b - d }
        }
        (BarrierDirection::Down, OptionType::Put) => {
            if k >= h { a - b + c - d } else { 0.0 }
        }
        (BarrierDirection::Up, OptionType::Call) => {
            if k >= h { 0.0 } else { a - b + c - d }
        }
        (BarrierDirection::Up, OptionType::Put) => {
            if k >= h { b - d } else { a - c }
        }
    };

    out.max(0.0)
}

pub fn barrier_price_closed_form(
    option_type: OptionType,
    style: BarrierStyle,
    direction: BarrierDirection,
    s0: f64,
    k: f64,
    barrier: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    let vanilla = black_scholes_price(option_type, s0, k, r, sigma, t);

    if breached_at_start(s0, barrier, direction) {
        return match style {
            BarrierStyle::Out => 0.0,
            BarrierStyle::In => vanilla,
        };
    }

    let out = rr_out_price(option_type, direction, s0, k, barrier, r, sigma, t);

    match style {
        BarrierStyle::Out => out,
        BarrierStyle::In => (vanilla - out).max(0.0),
    }
}

pub fn barrier_price_mc(
    option_type: OptionType,
    style: BarrierStyle,
    direction: BarrierDirection,
    s0: f64,
    k: f64,
    barrier: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    num_paths: usize,
    seed: u64,
) -> (f64, f64) {
    let generator = GbmPathGenerator {
        model: Gbm { mu: r, sigma },
        s0,
        maturity: t,
        steps,
    };

    let engine = MonteCarloEngine::new(num_paths, seed).with_antithetic(true);
    let discount = (-r * t).exp();

    engine.run(
        &generator,
        |path| {
            let hit = path_hits_barrier(path, barrier, direction);
            let knocked_in = match style {
                BarrierStyle::In => hit,
                BarrierStyle::Out => !hit,
            };
            if knocked_in {
                vanilla_payoff(option_type, path[path.len() - 1], k)
            } else {
                0.0
            }
        },
        discount,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn barrier_parity_closed_form_down_call() {
        let s0 = 100.0;
        let k = 100.0;
        let h = 90.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let out = barrier_price_closed_form(
            OptionType::Call,
            BarrierStyle::Out,
            BarrierDirection::Down,
            s0,
            k,
            h,
            r,
            sigma,
            t,
        );
        let inn = barrier_price_closed_form(
            OptionType::Call,
            BarrierStyle::In,
            BarrierDirection::Down,
            s0,
            k,
            h,
            r,
            sigma,
            t,
        );
        let vanilla = black_scholes_price(OptionType::Call, s0, k, r, sigma, t);
        assert_relative_eq!(out + inn, vanilla, epsilon = 1e-8);
    }

    #[test]
    fn barrier_parity_closed_form_up_put() {
        let s0 = 100.0;
        let k = 95.0;
        let h = 120.0;
        let r = 0.03;
        let sigma = 0.25;
        let t = 0.8;

        let out = barrier_price_closed_form(
            OptionType::Put,
            BarrierStyle::Out,
            BarrierDirection::Up,
            s0,
            k,
            h,
            r,
            sigma,
            t,
        );
        let inn = barrier_price_closed_form(
            OptionType::Put,
            BarrierStyle::In,
            BarrierDirection::Up,
            s0,
            k,
            h,
            r,
            sigma,
            t,
        );
        let vanilla = black_scholes_price(OptionType::Put, s0, k, r, sigma, t);
        assert_relative_eq!(out + inn, vanilla, epsilon = 1e-8);
    }

    #[test]
    fn barrier_mc_parity_matches_vanilla() {
        let s0 = 100.0;
        let k = 100.0;
        let h = 110.0;
        let r = 0.01;
        let sigma = 0.2;
        let t = 1.0;

        let (out, e_out) = barrier_price_mc(
            OptionType::Call,
            BarrierStyle::Out,
            BarrierDirection::Up,
            s0,
            k,
            h,
            r,
            sigma,
            t,
            252,
            30_000,
            10,
        );
        let (inn, e_in) = barrier_price_mc(
            OptionType::Call,
            BarrierStyle::In,
            BarrierDirection::Up,
            s0,
            k,
            h,
            r,
            sigma,
            t,
            252,
            30_000,
            10,
        );

        let vanilla = black_scholes_price(OptionType::Call, s0, k, r, sigma, t);
        assert!((out + inn - vanilla).abs() <= 2.5 * (e_out + e_in));
    }
}
