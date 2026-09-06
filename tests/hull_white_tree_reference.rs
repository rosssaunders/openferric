//! Independent Hull-White references for the short-rate tree.
//!
//! The European (single-exercise) swaption oracle below is Jamshidian's
//! decomposition into zero-coupon-bond options, using the fitted-curve
//! Hull-White formulas in Brigo and Mercurio (2006), section 3.3.  It is
//! deliberately separate from the lattice implementation under test.
//!
//! Primary model reference: Hull and White (1990), *The Review of Financial
//! Studies* 3(4), 573-592, doi:10.1093/rfs/3.4.573.

use openferric::engines::tree::BermudanSwaptionEngine;
use openferric::math::normal_cdf;
use openferric::models::HullWhite;
use openferric::rates::{Swaption, YieldCurve};

const FLAT_RATE: f64 = 0.05;

fn discount(t: f64) -> f64 {
    (-FLAT_RATE * t).exp()
}

fn flat_curve() -> YieldCurve {
    YieldCurve::new(
        (0..=160)
            .map(|i| {
                let t = i as f64 * 0.125;
                (t, discount(t))
            })
            .collect(),
    )
}

fn hw_b(a: f64, t: f64, maturity: f64) -> f64 {
    (1.0 - (-a * (maturity - t)).exp()) / a
}

fn hw_a(a: f64, sigma: f64, t: f64, maturity: f64) -> f64 {
    let b = hw_b(a, t, maturity);
    (discount(maturity) / discount(t))
        * (b * FLAT_RATE - sigma * sigma / (4.0 * a) * (1.0 - (-2.0 * a * t).exp()) * b * b).exp()
}

fn hw_bond_price(a: f64, sigma: f64, t: f64, maturity: f64, short_rate: f64) -> f64 {
    hw_a(a, sigma, t, maturity) * (-hw_b(a, t, maturity) * short_rate).exp()
}

fn zero_bond_put(a: f64, sigma: f64, expiry: f64, maturity: f64, strike: f64) -> f64 {
    let sigma_p =
        sigma * ((1.0 - (-2.0 * a * expiry).exp()) / (2.0 * a)).sqrt() * hw_b(a, expiry, maturity);
    let h = (discount(maturity) / (discount(expiry) * strike)).ln() / sigma_p + 0.5 * sigma_p;
    strike * discount(expiry) * normal_cdf(-h + sigma_p) - discount(maturity) * normal_cdf(-h)
}

fn jamshidian_payer_price(
    a: f64,
    sigma: f64,
    expiry: f64,
    tenor_years: usize,
    strike: f64,
    notional: f64,
) -> f64 {
    let pay_times: Vec<f64> = (1..=tenor_years).map(|i| expiry + i as f64).collect();
    let coupons: Vec<f64> = (1..=tenor_years)
        .map(|i| strike + f64::from(i == tenor_years))
        .collect();

    // At exercise, the fixed-rate bond is strictly decreasing in r.  The
    // payer swaption is a put on that bond with strike one.
    let coupon_bond = |r: f64| -> f64 {
        pay_times
            .iter()
            .zip(coupons.iter())
            .map(|(maturity, coupon)| coupon * hw_bond_price(a, sigma, expiry, *maturity, r))
            .sum()
    };
    let (mut lo, mut hi) = (-1.0_f64, 2.0_f64);
    assert!(coupon_bond(lo) > 1.0 && coupon_bond(hi) < 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if coupon_bond(mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r_star = 0.5 * (lo + hi);

    notional
        * pay_times
            .iter()
            .zip(coupons.iter())
            .map(|(maturity, coupon)| {
                let bond_strike = hw_bond_price(a, sigma, expiry, *maturity, r_star);
                coupon * zero_bond_put(a, sigma, expiry, *maturity, bond_strike)
            })
            .sum::<f64>()
}

/// Deterministic shift in the fitted flat-curve Hull-White representation
/// `r(t) = phi(t) + x(t)`, `dx = -a*x dt + sigma dW`.
fn hw_flat_phi(a: f64, sigma: f64, t: f64) -> f64 {
    FLAT_RATE + sigma * sigma / (2.0 * a * a) * (1.0 - (-a * t).exp()).powi(2)
}

fn hw_flat_phi_integral(a: f64, sigma: f64, start: f64, end: f64) -> f64 {
    let delta = end - start;
    let exp_start = (-a * start).exp();
    let exp_end = (-a * end).exp();
    let exp_2_start = (-2.0 * a * start).exp();
    let exp_2_end = (-2.0 * a * end).exp();
    FLAT_RATE * delta
        + sigma * sigma / (2.0 * a * a)
            * (delta - 2.0 * (exp_start - exp_end) / a + (exp_2_start - exp_2_end) / (2.0 * a))
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Exact expectation of a piecewise-linear function under a normal law.
/// Values outside the grid use constant endpoint extrapolation; the reference
/// grid below puts both endpoints more than seven unconditional standard
/// deviations from the root, making that tail treatment immaterial at the
/// stated precision.
fn normal_piecewise_linear_expectation(
    grid: &[f64],
    values: &[f64],
    mean: f64,
    standard_deviation: f64,
) -> f64 {
    let first_z = (grid[0] - mean) / standard_deviation;
    let last_z = (grid[grid.len() - 1] - mean) / standard_deviation;
    let mut expectation =
        values[0] * normal_cdf(first_z) + values[values.len() - 1] * normal_cdf(-last_z);

    for i in 0..grid.len() - 1 {
        let x0 = grid[i];
        let x1 = grid[i + 1];
        let z0 = (x0 - mean) / standard_deviation;
        let z1 = (x1 - mean) / standard_deviation;
        let probability = normal_cdf(z1) - normal_cdf(z0);
        let truncated_first_moment =
            mean * probability + standard_deviation * (normal_pdf(z0) - normal_pdf(z1));
        let slope = (values[i + 1] - values[i]) / (x1 - x0);
        let intercept = values[i] - slope * x0;
        expectation += intercept * probability + slope * truncated_first_moment;
    }
    expectation
}

/// Computes `E_t[exp(-integral_t^u r(s) ds) f(x(u))]`.  Exponential tilting
/// of the joint Gaussian `(x(u), integral r ds)` changes this into an ordinary
/// normal expectation whose mean is shifted by their covariance.  Integration
/// of the piecewise-linear continuation function is analytic interval by
/// interval, leaving only the explicitly refined state grid as an approximation.
fn hw_discounted_grid_transition(
    a: f64,
    sigma: f64,
    start: f64,
    end: f64,
    x_start: f64,
    grid: &[f64],
    continuation_values: &[f64],
) -> f64 {
    let delta = end - start;
    let exp_a = (-a * delta).exp();
    let b = (1.0 - exp_a) / a;
    let exp_2a_integral = (1.0 - (-2.0 * a * delta).exp()) / (2.0 * a);
    let variance_x = sigma * sigma * exp_2a_integral;
    let variance_integral = sigma * sigma / (a * a) * (delta - 2.0 * b + exp_2a_integral);
    let covariance = sigma * sigma / a * (b - exp_2a_integral);
    let mean_x = x_start * exp_a;
    let mean_integral = hw_flat_phi_integral(a, sigma, start, end) + x_start * b;
    let tilted_mean_x = mean_x - covariance;
    let discount_scale = (-mean_integral + 0.5 * variance_integral).exp();
    let standard_deviation_x = variance_x.sqrt();

    discount_scale
        * normal_piecewise_linear_expectation(
            grid,
            continuation_values,
            tilted_mean_x,
            standard_deviation_x,
        )
}

fn rolling_payer_exercise_value(
    a: f64,
    sigma: f64,
    exercise: f64,
    x: f64,
    tenor_years: usize,
    strike: f64,
    notional: f64,
) -> f64 {
    let short_rate = hw_flat_phi(a, sigma, exercise) + x;
    let end = exercise + tenor_years as f64;
    let annuity = (1..=tenor_years)
        .map(|year| hw_bond_price(a, sigma, exercise, exercise + year as f64, short_rate))
        .sum::<f64>();
    let discount_end = hw_bond_price(a, sigma, exercise, end, short_rate);
    (notional * (1.0 - discount_end - strike * annuity)).max(0.0)
}

fn rolling_bermudan_gaussian_reference(
    a: f64,
    sigma: f64,
    exercise_dates: &[f64],
    tenor_years: usize,
    strike: f64,
    notional: f64,
    state_intervals: usize,
) -> f64 {
    const X_HALF_WIDTH: f64 = 0.12;
    let dx = 2.0 * X_HALF_WIDTH / state_intervals as f64;
    let grid = (0..=state_intervals)
        .map(|i| -X_HALF_WIDTH + i as f64 * dx)
        .collect::<Vec<_>>();
    let last_index = exercise_dates.len() - 1;
    let mut values = grid
        .iter()
        .map(|&x| {
            rolling_payer_exercise_value(
                a,
                sigma,
                exercise_dates[last_index],
                x,
                tenor_years,
                strike,
                notional,
            )
        })
        .collect::<Vec<_>>();

    for exercise_index in (0..last_index).rev() {
        let start = exercise_dates[exercise_index];
        let end = exercise_dates[exercise_index + 1];
        let mut earlier_values = Vec::with_capacity(grid.len());
        for &x in &grid {
            let exercise =
                rolling_payer_exercise_value(a, sigma, start, x, tenor_years, strike, notional);
            let continuation =
                hw_discounted_grid_transition(a, sigma, start, end, x, &grid, &values);
            earlier_values.push(exercise.max(continuation));
        }
        values = earlier_values;
    }

    hw_discounted_grid_transition(a, sigma, 0.0, exercise_dates[0], 0.0, &grid, &values)
}

#[test]
fn calibrated_flat_curve_theta_matches_hull_white_closed_form() {
    let a = 0.05;
    let sigma = 0.01;
    let curve = flat_curve();
    let times: Vec<f64> = (0..=80).map(|i| i as f64 * 0.125).collect();
    let mut model = HullWhite::new(a, sigma);
    model.calibrate_theta(&curve, &times);

    for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let expected = a * FLAT_RATE + sigma * sigma * (1.0 - (-2.0 * a * t).exp()) / (2.0 * a);
        let actual = model.theta_at(t);
        let roundoff = 2.0e-10;
        assert!(
            (actual - expected).abs() <= roundoff,
            "theta({t}): expected={expected:.17e}, actual={actual:.17e}, error={:.3e}",
            (actual - expected).abs()
        );
    }
}

#[test]
fn single_exercise_tree_converges_to_jamshidian_price() {
    let a = 0.05;
    let sigma = 0.01;
    let expiry = 3.0;
    let tenor_years = 5_usize;
    let notional = 1_000_000.0;
    let curve = flat_curve();
    let strike = 0.04;
    let swaption = Swaption {
        notional,
        strike,
        option_expiry: expiry,
        swap_tenor: tenor_years as f64,
        is_payer: true,
    };
    let exact = jamshidian_payer_price(a, sigma, expiry, tenor_years, strike, notional);
    // QuantLib-Python 1.43 `JamshidianSwaptionEngine` independently gives
    // 49_085.691_116_152_433 for this contract/model fixture.
    assert!((exact - 49_085.691_116_152_43).abs() <= 2.0e-9);
    let gaussian_dp = rolling_bermudan_gaussian_reference(
        a,
        sigma,
        &[expiry],
        tenor_years,
        strike,
        notional,
        2_400,
    );
    // The independently integrated 2400-interval grid differs from the exact
    // Jamshidian value by 0.044904 currency units.
    assert!(
        (gaussian_dp - exact).abs() <= 0.05,
        "single-exercise Gaussian DP={gaussian_dp:.12}, Jamshidian={exact:.12}, error={:.6}",
        (gaussian_dp - exact).abs()
    );

    let coarse = BermudanSwaptionEngine::new(HullWhite::new(a, sigma), 300).price(
        &swaption,
        &[expiry],
        &curve,
    );
    let fine = BermudanSwaptionEngine::new(HullWhite::new(a, sigma), 1200).price(
        &swaption,
        &[expiry],
        &curve,
    );
    let coarse_error = (coarse - exact).abs();
    let fine_error = (fine - exact).abs();

    // This is a convergence assertion to an analytic price, not a price
    // range.  The absolute budget is set after measuring the stated 1200-step
    // lattice and the refinement assertion prevents silently widening it.
    let numerical_budget = 2.5;
    assert!(
        fine_error <= numerical_budget,
        "Jamshidian={exact:.12}, tree(1200)={fine:.12}, error={fine_error:.6}, budget={numerical_budget}"
    );
    assert!(
        fine_error < coarse_error,
        "tree failed to converge: error(300)={coarse_error:.6}, error(1200)={fine_error:.6}"
    );
}

#[test]
fn rolling_tenor_multi_exercise_tree_matches_gaussian_dynamic_program() {
    let a = 0.05;
    let sigma = 0.01;
    let exercise_dates = [1.0, 2.0, 3.0];
    let tenor_years = 5_usize;
    let strike = 0.04;
    let notional = 1_000_000.0;

    // Unlike a conventional Bermudan swaption on one fixed underlying swap,
    // this engine starts a new five-year swap at each exercise date.  The
    // independent oracle therefore performs dynamic programming directly in
    // the fitted one-factor Gaussian state.  Discounting between decisions
    // uses the exact joint Gaussian law of `(x(u), integral r ds)` and exact
    // normal moments of a piecewise-linear continuation value.
    let reference_1200 = rolling_bermudan_gaussian_reference(
        a,
        sigma,
        &exercise_dates,
        tenor_years,
        strike,
        notional,
        1_200,
    );
    let reference_2400 = rolling_bermudan_gaussian_reference(
        a,
        sigma,
        &exercise_dates,
        tenor_years,
        strike,
        notional,
        2_400,
    );
    let reference_refinement = (reference_2400 - reference_1200).abs();
    // Continuous-state Gaussian DP convergence:
    //  1200 state intervals -> 53_895.315308502410
    //  2400 state intervals -> 53_894.847096310594
    // The 0.6 currency-unit reference budget is the measured final
    // refinement (0.468213) rounded up.
    assert!(
        (reference_2400 - 53_894.847_096_310_594).abs() <= 2.0e-8,
        "rolling Gaussian-DP grid changed: {reference_2400:.17}"
    );

    let curve = flat_curve();
    let swaption = Swaption {
        notional,
        strike,
        option_expiry: exercise_dates[0],
        swap_tenor: tenor_years as f64,
        is_payer: true,
    };
    let coarse = BermudanSwaptionEngine::new(HullWhite::new(a, sigma), 300).price(
        &swaption,
        &exercise_dates,
        &curve,
    );
    let fine = BermudanSwaptionEngine::new(HullWhite::new(a, sigma), 2400).price(
        &swaption,
        &exercise_dates,
        &curve,
    );
    let coarse_error = (coarse - reference_2400).abs();
    let fine_error = (fine - reference_2400).abs();

    // The reference budget is its measured 1200 -> 2400 interval refinement.
    // The tree budget is stated in currency units and is paired with an explicit
    // 300 -> 2400 step convergence requirement rather than a price interval.
    assert!(
        reference_refinement <= 0.6,
        "Gaussian DP failed to converge: grid1200={reference_1200:.12}, grid2400={reference_2400:.12}, refinement={reference_refinement:.6}"
    );
    // Preserve the 1.5-unit tree budget after introducing exact curve fitting;
    // refine the lattice rather than widen the independent oracle budget.
    let tree_discretization_budget = 1.5 + reference_refinement;
    assert!(
        fine_error <= tree_discretization_budget,
        "Gaussian DP={reference_2400:.12}, tree(2400)={fine:.12}, error={fine_error:.6}, budget={tree_discretization_budget:.6}"
    );
    assert!(
        fine_error < coarse_error,
        "rolling tree failed to converge: error(300)={coarse_error:.6}, error(2400)={fine_error:.6}"
    );
}
