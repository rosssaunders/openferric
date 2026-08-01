//! Audit tests for `models::hw_calibration` volatility-unit consistency.
//!
//! Background: `hw_atm_swaption_vol_approx` returns an ATM *normal
//! (Bachelier)* volatility (absolute rate units, Hull-White `sigma` scale,
//! ~0.01), not a lognormal Black volatility (~0.20 at 4% rates). A past
//! defect labeled it a Black vol and pushed it through a lognormal ATM price
//! transform next to market Black quotes; feeding realistic Black vols then
//! silently returned a boundary-pinned degenerate fit (a, sigma) ~ (4e-4,
//! 0.08) instead of ~0.01. The in-module round-trip test could not catch this
//! because it generates quotes from the same formula.
//!
//! These tests anchor the approximation and the calibrator against an
//! *independent* pricing path: the exact Hull-White ATM payer swaption price
//! via Jamshidian's (1989) decomposition into zero-coupon-bond puts, using
//! the closed forms of Brigo and Mercurio, "Interest Rate Models - Theory and
//! Practice", 2nd ed. (2006), Section 3.3 (Eqs. (3.39)-(3.41) and the
//! decomposition around Eq. (3.45)). Prices are converted to normal vols with
//! the Bachelier ATM formula `price = annuity * sigma_n * sqrt(T / (2*pi))`.

use openferric::math::normal_cdf;
use openferric::models::{calibrate_hull_white_params, hw_atm_swaption_vol_approx};

/// Flat continuously-compounded zero curve.
const R0: f64 = 0.04;

fn discount(t: f64) -> f64 {
    (-R0 * t).exp()
}

/// `B(t, T) = (1 - exp(-a (T - t))) / a`, Brigo-Mercurio Eq. (3.39).
fn hw_b(a: f64, t: f64, maturity: f64) -> f64 {
    (1.0 - (-a * (maturity - t)).exp()) / a
}

/// `A(t, T)` for Hull-White fitted to the initial curve, Brigo-Mercurio
/// Eq. (3.39). On a flat curve the instantaneous forward is `f(0, t) = R0`.
fn hw_a(a: f64, sigma: f64, t: f64, maturity: f64) -> f64 {
    let b = hw_b(a, t, maturity);
    (discount(maturity) / discount(t))
        * (b * R0 - sigma * sigma / (4.0 * a) * (1.0 - (-2.0 * a * t).exp()) * b * b).exp()
}

/// Fitted-curve Hull-White zero-coupon bond price `P(t, T; r)`.
fn hw_bond_price(a: f64, sigma: f64, t: f64, maturity: f64, short_rate: f64) -> f64 {
    hw_a(a, sigma, t, maturity) * (-hw_b(a, t, maturity) * short_rate).exp()
}

/// European put on a zero-coupon bond, `ZBP(0, expiry, bond_maturity, strike)`,
/// Brigo-Mercurio Eqs. (3.40)-(3.41).
fn hw_zero_bond_put(a: f64, sigma: f64, expiry: f64, bond_maturity: f64, strike: f64) -> f64 {
    let sigma_p = sigma
        * ((1.0 - (-2.0 * a * expiry).exp()) / (2.0 * a)).sqrt()
        * hw_b(a, expiry, bond_maturity);
    let h = (discount(bond_maturity) / (discount(expiry) * strike)).ln() / sigma_p + 0.5 * sigma_p;
    strike * discount(expiry) * normal_cdf(-h + sigma_p) - discount(bond_maturity) * normal_cdf(-h)
}

/// Annuity (PVBP) of an annual fixed leg paying at `expiry + 1, .., expiry + tenor`.
fn annuity(expiry: f64, tenor_years: usize) -> f64 {
    (1..=tenor_years).map(|i| discount(expiry + i as f64)).sum()
}

/// ATM forward swap rate on the flat curve.
fn forward_swap_rate(expiry: f64, tenor_years: usize) -> f64 {
    (discount(expiry) - discount(expiry + tenor_years as f64)) / annuity(expiry, tenor_years)
}

/// Exact Hull-White ATM payer swaption price via Jamshidian's decomposition
/// (Brigo-Mercurio Section 3.3): find `r*` with
/// `sum_i c_i P(expiry, t_i; r*) = 1`, then the payer swaption is
/// `sum_i c_i ZBP(expiry, t_i, K_i)` with `K_i = P(expiry, t_i; r*)`.
fn jamshidian_atm_payer_price(a: f64, sigma: f64, expiry: f64, tenor_years: usize) -> f64 {
    let strike_rate = forward_swap_rate(expiry, tenor_years);
    let pay_times: Vec<f64> = (1..=tenor_years).map(|i| expiry + i as f64).collect();
    let coupons: Vec<f64> = (1..=tenor_years)
        .map(|i| {
            if i == tenor_years {
                1.0 + strike_rate
            } else {
                strike_rate
            }
        })
        .collect();

    // Coupon-bond price at expiry is strictly decreasing in the short rate,
    // so bisect for r*.
    let bond_at = |r: f64| -> f64 {
        pay_times
            .iter()
            .zip(coupons.iter())
            .map(|(t, c)| c * hw_bond_price(a, sigma, expiry, *t, r))
            .sum()
    };
    let (mut lo, mut hi) = (-1.0_f64, 2.0_f64);
    assert!(bond_at(lo) > 1.0 && bond_at(hi) < 1.0, "r* bracket invalid");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if bond_at(mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r_star = 0.5 * (lo + hi);

    pay_times
        .iter()
        .zip(coupons.iter())
        .map(|(t, c)| {
            let strike_i = hw_bond_price(a, sigma, expiry, *t, r_star);
            c * hw_zero_bond_put(a, sigma, expiry, *t, strike_i)
        })
        .sum()
}

/// Invert the Bachelier ATM formula `price = annuity * sigma_n * sqrt(T/(2 pi))`
/// (exact at the money, where the Bachelier price is linear in the vol).
fn implied_atm_normal_vol(price: f64, expiry: f64, tenor_years: usize) -> f64 {
    price / (annuity(expiry, tenor_years) * (expiry / (2.0 * std::f64::consts::PI)).sqrt())
}

/// The approximation must sit at the *normal-vol* scale: within a modest
/// approximation tolerance of the Jamshidian-implied Bachelier vol, and an
/// order of magnitude away from the Black (lognormal) vol of the same price.
///
/// For (a, sigma) = (0.05, 0.01), expiry 5y, tenor 5y on a flat 4% curve the
/// implied Black ATM vol is `~ sigma_n / F ~ 0.19`, roughly 25x the normal
/// vol, so a 10% tolerance decisively pins the unit (observed relative error
/// of the frozen-weight approximation is 4.0%-5.5% across this matrix).
#[test]
fn approximation_matches_jamshidian_implied_normal_vol_scale() {
    let a = 0.05;
    let sigma = 0.01;

    for expiry in [1.0, 2.0, 5.0] {
        for tenor_years in [2usize, 5, 10] {
            let price = jamshidian_atm_payer_price(a, sigma, expiry, tenor_years);
            let implied_normal = implied_atm_normal_vol(price, expiry, tenor_years);
            let approx = hw_atm_swaption_vol_approx(a, sigma, expiry, tenor_years as f64);
            let rel = (approx - implied_normal).abs() / implied_normal;
            eprintln!(
                "expiry={expiry} tenor={tenor_years}: jamshidian_normal={implied_normal:.6} \
                 approx={approx:.6} rel_err={rel:.4} black_equiv={:.4}",
                implied_normal / forward_swap_rate(expiry, tenor_years)
            );
            assert!(
                rel < 0.10,
                "approx {approx} vs Jamshidian-implied normal vol {implied_normal} \
                 (rel err {rel}) for expiry {expiry}, tenor {tenor_years}"
            );

            // Unit pin: the same price expressed as a Black vol is
            // sigma_n / F at the money, ~20x larger. The approximation must
            // be nowhere near it.
            let black_equiv = implied_normal / forward_swap_rate(expiry, tenor_years);
            assert!(
                approx < 0.25 * black_equiv,
                "approx {approx} is at the lognormal scale ({black_equiv})"
            );
        }
    }
}

/// End-to-end external anchor: generate ATM swaption prices from the exact
/// Jamshidian pricer at known `(a, sigma) = (0.05, 0.01)`, convert to normal
/// vols with the Bachelier ATM formula, and lock the resulting least-squares
/// projection onto the calibrator's frozen-weight approximation. The pre-fix
/// code instead fed comparable data into a lognormal transform and pinned
/// sigma at an unrelated search boundary.
#[test]
fn calibration_matches_jamshidian_projection_reference() {
    let true_a = 0.05;
    let true_sigma = 0.01;

    let mut quotes = Vec::new();
    for expiry in [1.0, 2.0, 3.0, 4.0, 5.0] {
        for tenor_years in [2usize, 5, 10] {
            let price = jamshidian_atm_payer_price(true_a, true_sigma, expiry, tenor_years);
            let vol = implied_atm_normal_vol(price, expiry, tenor_years);
            quotes.push((expiry, tenor_years as f64, vol));
        }
    }

    let (cal_a, cal_sigma) =
        calibrate_hull_white_params(&quotes).expect("normal-vol quotes must calibrate");
    // The exact Jamshidian prices are projected onto the deliberately
    // approximate frozen-weight volatility formula, so its least-squares
    // parameters differ from the generating parameters. Lock that deterministic
    // projection; exact synthetic quote recovery is tested in the model module.
    let expected_a = 0.046_383_839_484;
    let expected_sigma = 0.010_312_263_46;
    assert!((cal_a - expected_a).abs() <= 1.0e-15);
    assert!((cal_sigma - expected_sigma).abs() <= 1.0e-15);
}

/// Feeding lognormal Black ATM vols (~0.20 at 4% rates, the verified failure
/// mode) must fail loudly instead of silently returning the boundary-pinned
/// (a, sigma) ~ (4e-4, 0.08) the pre-fix code produced.
#[test]
fn black_lognormal_vol_inputs_are_rejected() {
    // Realistic ATM Black vol surface at ~4% rates: 15-25% lognormal.
    let mut quotes = Vec::new();
    for expiry in [1.0, 2.0, 3.0, 4.0, 5.0] {
        for tenor in [2.0, 5.0, 10.0] {
            let black_vol = 0.20 - 0.005 * (expiry - 1.0) + 0.002 * (tenor - 2.0) / 8.0;
            quotes.push((expiry, tenor, black_vol));
        }
    }

    assert!(
        calibrate_hull_white_params(&quotes).is_none(),
        "Black-scale vols must be rejected, not fitted to a boundary-pinned sigma"
    );
}
