//! Module `pricing::barrier`.
//!
//! Implements barrier workflows with concrete routines such as `barrier_price_closed_form_with_carry_and_rebate`, `barrier_price_closed_form`, `barrier_price_mc`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: free functions `barrier_price_closed_form_with_carry_and_rebate`, `barrier_price_closed_form`, `barrier_price_mc`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::core::types::OptionType;
use crate::math::normal_cdf;
use crate::mc::{GbmPathGenerator, MonteCarloEngine};
use crate::models::Gbm;

pub use crate::core::types::{BarrierDirection, BarrierStyle};

#[inline]
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

#[inline]
fn vanilla_payoff(option_type: OptionType, s_t: f64, k: f64) -> f64 {
    match option_type {
        OptionType::Call => (s_t - k).max(0.0),
        OptionType::Put => (k - s_t).max(0.0),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn bs_price_with_dividend(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return vanilla_payoff(option_type, s, k);
    }

    let st = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (0.5 * sigma).mul_add(sigma, r - q) * t) / st;
    let d2 = d1 - st;
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();

    // Compute call, derive put via put-call parity to halve CDF evaluations.
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = s.mul_add(df_q * nd1, -(k * df_r * nd2));
    match option_type {
        OptionType::Call => call,
        OptionType::Put => call - s * df_q + k * df_r,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn barrier_price_closed_form_with_carry_and_rebate(
    option_type: OptionType,
    style: BarrierStyle,
    direction: BarrierDirection,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    rebate: f64,
) -> f64 {
    let vanilla = bs_price_with_dividend(option_type, s, k, r, q, sigma, t);

    if breached_at_start(s, h, direction) {
        return match style {
            BarrierStyle::Out => rebate,
            BarrierStyle::In => vanilla,
        };
    }

    if t <= 0.0 || sigma <= 0.0 {
        return match style {
            BarrierStyle::Out => vanilla,
            BarrierStyle::In => 0.0,
        };
    }

    let eta = match direction {
        BarrierDirection::Down => 1.0,
        BarrierDirection::Up => -1.0,
    };
    let phi = option_type.sign();
    let b = r - q;

    let sigma_sq = sigma * sigma;
    let st = sigma * t.sqrt();
    let mu = (b - 0.5 * sigma_sq) / sigma_sq;
    let lambda = mu.mul_add(mu, 2.0 * r / sigma_sq).sqrt();
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();

    let one_plus_mu_st = (1.0 + mu) * st;
    let x1 = (s / k).ln() / st + one_plus_mu_st;
    let x2 = (s / h).ln() / st + one_plus_mu_st;
    let y1 = ((h * h) / (s * k)).ln() / st + one_plus_mu_st;
    let y2 = (h / s).ln() / st + one_plus_mu_st;
    let z = (h / s).ln() / st + lambda * st;

    // Compute (h/s)^exp via exp(exp * ln(h/s)) - one ln, multiple exp.
    let ln_hs = (h / s).ln();
    let hs_mu = (2.0 * mu * ln_hs).exp();
    let hs_mu1 = (2.0 * (mu + 1.0) * ln_hs).exp();
    let hs_mu_lambda_plus = ((mu + lambda) * ln_hs).exp();
    let hs_mu_lambda_minus = ((mu - lambda) * ln_hs).exp();

    let a = phi * s * df_q * normal_cdf(phi * x1) - phi * k * df_r * normal_cdf(phi * (x1 - st));
    let b_term =
        phi * s * df_q * normal_cdf(phi * x2) - phi * k * df_r * normal_cdf(phi * (x2 - st));
    let c = phi * s * df_q * hs_mu1 * normal_cdf(eta * y1)
        - phi * k * df_r * hs_mu * normal_cdf(eta * (y1 - st));
    let d = phi * s * df_q * hs_mu1 * normal_cdf(eta * y2)
        - phi * k * df_r * hs_mu * normal_cdf(eta * (y2 - st));
    let e = rebate * df_r * (normal_cdf(eta * (x2 - st)) - hs_mu * normal_cdf(eta * (y2 - st)));
    let f = rebate
        * (hs_mu_lambda_plus * normal_cdf(eta * z)
            + hs_mu_lambda_minus * normal_cdf(eta * (z - 2.0 * lambda * st)));

    let k_ge_h = k >= h;
    let value = match (style, direction, option_type, k_ge_h) {
        (BarrierStyle::Out, BarrierDirection::Down, OptionType::Call, true) => a - c + f,
        (BarrierStyle::Out, BarrierDirection::Down, OptionType::Call, false) => b_term - d + f,
        (BarrierStyle::Out, BarrierDirection::Up, OptionType::Call, true) => f,
        (BarrierStyle::Out, BarrierDirection::Up, OptionType::Call, false) => {
            a - b_term + c - d + f
        }
        (BarrierStyle::Out, BarrierDirection::Down, OptionType::Put, true) => {
            a - b_term + c - d + f
        }
        (BarrierStyle::Out, BarrierDirection::Down, OptionType::Put, false) => f,
        (BarrierStyle::Out, BarrierDirection::Up, OptionType::Put, true) => b_term - d + f,
        (BarrierStyle::Out, BarrierDirection::Up, OptionType::Put, false) => a - c + f,
        (BarrierStyle::In, BarrierDirection::Down, OptionType::Call, true) => c + e,
        (BarrierStyle::In, BarrierDirection::Down, OptionType::Call, false) => a - b_term + d + e,
        (BarrierStyle::In, BarrierDirection::Up, OptionType::Call, true) => a + e,
        (BarrierStyle::In, BarrierDirection::Up, OptionType::Call, false) => b_term - c + d + e,
        (BarrierStyle::In, BarrierDirection::Down, OptionType::Put, true) => b_term - c + d + e,
        (BarrierStyle::In, BarrierDirection::Down, OptionType::Put, false) => a + e,
        (BarrierStyle::In, BarrierDirection::Up, OptionType::Put, true) => a - b_term + d + e,
        (BarrierStyle::In, BarrierDirection::Up, OptionType::Put, false) => c + e,
    };

    value.max(0.0)
}

#[allow(clippy::too_many_arguments)]
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
    barrier_price_closed_form_with_carry_and_rebate(
        option_type,
        style,
        direction,
        s0,
        k,
        barrier,
        r,
        0.0,
        sigma,
        t,
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
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
        let vanilla = bs_price_with_dividend(OptionType::Call, s0, k, r, 0.0, sigma, t);
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
        let vanilla = bs_price_with_dividend(OptionType::Put, s0, k, r, 0.0, sigma, t);
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

        let vanilla = bs_price_with_dividend(OptionType::Call, s0, k, r, 0.0, sigma, t);
        assert!((out + inn - vanilla).abs() <= 2.5 * (e_out + e_in));
    }
}
