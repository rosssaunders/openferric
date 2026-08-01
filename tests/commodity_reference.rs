// Reference values from Schwartz (1997), Schwartz & Smith (2000),
// QuantLib BlackCalculator (BSD 3-Clause), and OpenGamma Strata (Apache 2.0)

use openferric::core::OptionType;
use openferric::engines::analytic::{black76_greeks, black76_price};
use openferric::instruments::{CommodityForward, CommodityOption, CommoditySpreadOption};
use openferric::models::{
    CommodityForwardCurve, CommoditySeasonalityModel, ForwardInterpolation, FuturesQuote,
    SchwartzOneFactor, SchwartzSmithTwoFactor, SeasonalityMode, TwoFactorCommodityProcess,
    TwoFactorSpreadModel, implied_convenience_yield,
};

// ---------------------------------------------------------------------------
// Schwartz one-factor: OU process moment tests
// ---------------------------------------------------------------------------

/// Helper: compute expected log-spot mean under the Schwartz one-factor model.
/// E[X_T] = alpha* + (X_0 - alpha*) * exp(-kappa * T)
/// where X = ln(S) and alpha* = long_run_log_mean.
fn ou_expected_mean(alpha_star: f64, x0: f64, kappa: f64, t: f64) -> f64 {
    alpha_star + (x0 - alpha_star) * (-kappa * t).exp()
}

/// Helper: compute variance of log-spot under the Schwartz one-factor model.
/// Var[X_T] = sigma^2 / (2*kappa) * (1 - exp(-2*kappa*T))
fn ou_variance(sigma: f64, kappa: f64, t: f64) -> f64 {
    sigma * sigma / (2.0 * kappa) * (1.0 - (-2.0 * kappa * t).exp())
}

/// Helper: compute the Schwartz one-factor futures price (closed-form).
/// ln F(T) = e^{-kT} * ln(S0) + (1 - e^{-kT}) * alpha* + sigma^2/(4*kappa) * (1 - e^{-2*kT})
///
/// This is E[S_T] under risk-neutral measure using the log-normal property:
/// F(T) = exp(E[X_T] + 0.5 * Var[X_T])
fn schwartz_futures_price(s0: f64, kappa: f64, sigma: f64, alpha_star: f64, t: f64) -> f64 {
    let x0 = s0.ln();
    let mean = ou_expected_mean(alpha_star, x0, kappa, t);
    let var = ou_variance(sigma, kappa, t);
    (mean + 0.5 * var).exp()
}

#[test]
fn schwartz_one_factor_ou_expected_mean() {
    // NFCP-style crude oil parameters
    let kappa = 1.49;
    let sigma = 0.286;
    // Set mu such that long_run_log_mean = 3.00
    let alpha_star = 3.00;
    let mu = alpha_star + 0.5 * sigma * sigma / kappa;

    let model = SchwartzOneFactor { kappa, mu, sigma };

    let alpha_computed = model.long_run_log_mean();
    assert!(
        (alpha_computed - alpha_star).abs() < 1e-12,
        "long_run_log_mean should match: got {}, expected {}",
        alpha_computed,
        alpha_star
    );

    // Decimal (80-digit) evaluation of the OU conditional-mean formula.
    let x0 = 20.0_f64.ln();
    let references = [
        (0.5, 2.997_973_963_873_319),
        (1.0, 2.999_038_171_157_747),
        (2.0, 2.999_783_230_079_647),
        (5.0, 2.999_997_518_566_255),
    ];
    for &(t, reference) in &references {
        let mean = ou_expected_mean(alpha_star, x0, kappa, t);
        assert!((mean - reference).abs() <= 2.0e-15);
    }
}

#[test]
fn schwartz_one_factor_ou_variance() {
    let kappa = 1.49;
    let sigma = 0.286;
    let alpha_star = 3.00;
    let mu = alpha_star + 0.5 * sigma * sigma / kappa;

    let model = SchwartzOneFactor { kappa, mu, sigma };

    // Check model validates
    model.validate().expect("model should be valid");

    // Decimal (80-digit) evaluations of the exact OU variance.
    let long_run_var = sigma * sigma / (2.0 * kappa);
    let references = [
        (0.25, 0.014_417_662_148_360_713),
        (0.5, 0.021_262_220_895_132_91),
        (1.0, 0.026_054_144_080_935_155),
        (2.0, 0.027_377_507_892_730_223),
        (5.0, 0.027_448_312_868_077_382),
        (10.0, 0.027_448_322_147_647_87),
    ];
    for &(t, reference) in &references {
        let var = ou_variance(sigma, kappa, t);
        assert!((var - reference).abs() <= 2.0e-17);
    }

    // At T=50 the remaining analytic tail is 5.35e-67 and rounds away in f64.
    let var_long = ou_variance(sigma, kappa, 50.0);
    assert_eq!(var_long, long_run_var);
}

#[test]
fn schwartz_one_factor_futures_term_structure() {
    // NFCP crude oil parameters
    let kappa = 1.49;
    let sigma = 0.286;
    let alpha_star = 3.00;
    let s0 = 20.0;

    // Compute expected futures prices from closed-form formula
    let f_05 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 0.5);
    let f_10 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 1.0);
    let f_20 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 2.0);
    let f_50 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 5.0);

    let references = [
        20.259_120_238_738_785,
        20.329_341_634_953_764,
        20.357_959_874_337_344,
        20.363_043_707_840_006,
    ];
    for (actual, reference) in [f_05, f_10, f_20, f_50].into_iter().zip(references) {
        assert!((actual - reference).abs() <= 3.0e-14);
    }

    // Futures should converge to exp(alpha* + sigma^2/(4*kappa)) as T->inf
    let long_run_futures = (alpha_star + sigma * sigma / (4.0 * kappa)).exp();
    let f_100 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 100.0);
    assert!((long_run_futures - 20.363_094_331_926_926).abs() <= 3.0e-14);
    assert_eq!(f_100, long_run_futures);
}

#[test]
fn schwartz_one_factor_backwardation() {
    // Test with S0 well above equilibrium
    let kappa = 1.49;
    let sigma = 0.286;
    let alpha_star = 3.00;
    let s0 = 25.0; // ln(25) ~ 3.219 > alpha* = 3.0

    let f_05 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 0.5);
    let f_50 = schwartz_futures_price(s0, kappa, sigma, alpha_star, 5.0);

    // Decimal (80-digit) finite-horizon references. The exact T=5 residual
    // from the infinite-horizon level is 0.002591549324584..., not a 1% band.
    assert!((f_05 - 22.523_044_204_153_94).abs() <= 3.0e-14);
    assert!((f_50 - 20.365_685_881_251_51).abs() <= 3.0e-14);
    let long_run_futures = (alpha_star + sigma * sigma / (4.0 * kappa)).exp();
    let residual_reference = 0.002_591_549_324_584_057;
    // Subtracting two independently exponentiated values near 20 magnifies
    // their libm roundoff; budget those operands explicitly.
    let cancellation_roundoff = 8.0 * f64::EPSILON * (f_50.abs() + long_run_futures.abs()).max(1.0);
    assert!(
        (f_50 - long_run_futures - residual_reference).abs() <= cancellation_roundoff,
        "finite-horizon residual={} reference={residual_reference} roundoff budget={cancellation_roundoff:.3e}",
        f_50 - long_run_futures
    );
}

#[test]
fn schwartz_one_factor_step_exact_moments() {
    // Verify that step_log_exact produces correct conditional mean and variance
    // by running many steps from the same starting point
    let kappa = 1.49;
    let sigma = 0.286;
    let alpha_star = 3.00;
    let mu = alpha_star + 0.5 * sigma * sigma / kappa;

    let model = SchwartzOneFactor { kappa, mu, sigma };

    let x0 = 20.0_f64.ln();
    let dt = 1.0;

    // Use many samples to verify moments
    let n = 100_000;
    let terminals = model
        .simulate_terminal_spots(20.0, dt, 1, n, 12345)
        .expect("simulation should succeed");

    let log_terminals: Vec<f64> = terminals.iter().map(|s| s.ln()).collect();
    let mean_log = log_terminals.iter().sum::<f64>() / n as f64;
    let var_log = log_terminals
        .iter()
        .map(|x| (x - mean_log).powi(2))
        .sum::<f64>()
        / n as f64;

    let expected_mean = ou_expected_mean(alpha_star, x0, kappa, dt);
    let expected_var = ou_variance(sigma, kappa, dt);

    // Monte Carlo tolerance: ~1/sqrt(N)
    let tol_mean = 3.0 * (expected_var / n as f64).sqrt();
    let tol_var = 3.0 * expected_var * (2.0 / n as f64).sqrt(); // var of sample variance

    assert!(
        (mean_log - expected_mean).abs() < tol_mean,
        "MC mean {} vs expected {} (tol {})",
        mean_log,
        expected_mean,
        tol_mean
    );
    assert!(
        (var_log - expected_var).abs() < tol_var,
        "MC variance {} vs expected {} (tol {})",
        var_log,
        expected_var,
        tol_var
    );
}

// ---------------------------------------------------------------------------
// Schwartz-Smith two-factor: basic properties
// ---------------------------------------------------------------------------

#[test]
fn schwartz_smith_two_factor_spot_recovery() {
    let model = SchwartzSmithTwoFactor {
        kappa: 1.5,
        sigma_chi: 0.30,
        mu_xi: 0.02,
        sigma_xi: 0.10,
        rho: 0.3,
    };
    model.validate().expect("model should be valid");

    let chi = 0.1;
    let xi = 3.0;
    let spot = SchwartzSmithTwoFactor::spot_from_factors(chi, xi);
    assert!(
        (spot - (chi + xi).exp()).abs() < 1e-14,
        "spot should equal exp(chi + xi)"
    );
}

#[test]
fn schwartz_smith_two_factor_chi_mean_reverts() {
    let model = SchwartzSmithTwoFactor {
        kappa: 2.0,
        sigma_chi: 0.25,
        mu_xi: 0.0,
        sigma_xi: 0.0, // zero xi volatility to isolate chi
        rho: 0.0,
    };

    let chi0 = 1.0;
    let xi0 = 3.0;

    // Simulate many paths to check chi mean-reverts toward zero
    let n = 50_000;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = StdRng::seed_from_u64(999);
    let t = 2.0;
    let mut sum_chi = 0.0;

    for _ in 0..n {
        let z1: f64 = StandardNormal.sample(&mut rng);
        let z2: f64 = StandardNormal.sample(&mut rng);
        let (chi_t, _xi_t) = model.step_exact(chi0, xi0, t, z1, z2).unwrap();
        sum_chi += chi_t;
    }

    let mean_chi = sum_chi / n as f64;
    let expected_chi = chi0 * (-model.kappa * t).exp(); // mean of chi_T
    let variance_chi =
        model.sigma_chi.powi(2) * (1.0 - (-2.0 * model.kappa * t).exp()) / (2.0 * model.kappa);
    let tolerance = 4.0 * (variance_chi / n as f64).sqrt();

    assert!(
        (mean_chi - expected_chi).abs() <= tolerance,
        "chi should mean-revert: mean={}, expected={}, tolerance={tolerance}",
        mean_chi,
        expected_chi
    );
}

// ---------------------------------------------------------------------------
// Black-76 reference prices (SciPy normal CDF; consistent with QuantLib BlackCalculator)
// ---------------------------------------------------------------------------
// Parameters: Forward=100, vol=0.20, T=1.0, r=0.0 => StdDev = 0.20, Discount = 1.0

#[test]
fn black76_atm_call_matches_scipy_reference() {
    let price = black76_price(OptionType::Call, 100.0, 100.0, 0.0, 0.20, 1.0).unwrap();
    // Independent SciPy 1.17.1 normal-CDF evaluation.
    assert!(
        (price - 7.965_567_455_405_804).abs() <= 3.0e-14,
        "ATM call: expected 7.965567455405804, got {}",
        price
    );
}

#[test]
fn black76_itm_call_matches_scipy_reference() {
    let price = black76_price(OptionType::Call, 100.0, 90.0, 0.0, 0.20, 1.0).unwrap();
    assert!(
        (price - 13.589_108_116_054_796).abs() <= 3.0e-14,
        "ITM call (K=90): expected 13.589108116054796, got {}",
        price
    );
}

#[test]
fn black76_otm_call_matches_scipy_reference() {
    let price = black76_price(OptionType::Call, 100.0, 110.0, 0.0, 0.20, 1.0).unwrap();
    assert!(
        (price - 4.292_010_941_409_885).abs() <= 3.0e-14,
        "OTM call (K=110): expected 4.292010941409885, got {}",
        price
    );
}

#[test]
fn black76_itm_put_matches_scipy_reference() {
    let price = black76_price(OptionType::Put, 100.0, 110.0, 0.0, 0.20, 1.0).unwrap();
    assert!(
        (price - 14.292_010_941_409_899).abs() <= 3.0e-14,
        "ITM put (K=110): expected 14.292010941409899, got {}",
        price
    );
}

#[test]
fn black76_otm_put_matches_scipy_reference() {
    let price = black76_price(OptionType::Put, 100.0, 90.0, 0.0, 0.20, 1.0).unwrap();
    assert!(
        (price - 3.589_108_116_054_795_5).abs() <= 3.0e-14,
        "OTM put (K=90): expected 3.5891081160547955, got {}",
        price
    );
}

// ---------------------------------------------------------------------------
// Black-76 Greeks (Forward=100, Strike=105, vol=0.20, T=1.0, r=-ln(0.95))
// StdDev = vol * sqrt(T) = 0.20, Discount = 0.95
// ---------------------------------------------------------------------------

#[test]
fn black76_greeks_quantlib() {
    let discount = 0.95_f64;
    let t = 1.0;
    let r = -discount.ln() / t; // r such that exp(-r*T) = 0.95
    let forward = 100.0;
    let strike = 105.0;
    let vol = 0.20;

    let greeks = black76_greeks(OptionType::Call, forward, strike, r, vol, t).unwrap();

    // Compute expected values using the same formulas as the engine.
    // d1 = (ln(F/K) + 0.5*v^2*T) / (v*sqrt(T))
    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / sig_sqrt_t;
    let _d2 = d1 - sig_sqrt_t;

    let df = (-r * t).exp();

    // Use the standard normal PDF/CDF from the math module
    let nd1 = openferric::math::normal_cdf(d1);
    let pdf_d1 = openferric::math::normal_pdf(d1);

    let expected_delta = df * nd1;
    let expected_gamma = df * pdf_d1 / (forward * vol * sqrt_t);
    let expected_vega = df * forward * pdf_d1 * sqrt_t;

    // Theta: r * price - df * F * pdf(d1) * vol / (2*sqrt(T))
    let call_price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
    let expected_theta = r * call_price - df * forward * pdf_d1 * vol / (2.0 * sqrt_t);

    // Rho: -T * price (for fixed-forward Black-76)
    let expected_rho = -t * call_price;

    assert!(
        (greeks.delta - expected_delta).abs() < 1e-10,
        "delta: expected {}, got {}",
        expected_delta,
        greeks.delta
    );
    assert!(
        (greeks.gamma - expected_gamma).abs() < 1e-10,
        "gamma: expected {}, got {}",
        expected_gamma,
        greeks.gamma
    );
    assert!(
        (greeks.vega - expected_vega).abs() < 1e-10,
        "vega: expected {}, got {}",
        expected_vega,
        greeks.vega
    );
    assert!(
        (greeks.theta - expected_theta).abs() < 1e-10,
        "theta: expected {}, got {}",
        expected_theta,
        greeks.theta
    );
    assert!(
        (greeks.rho - expected_rho).abs() < 1e-10,
        "rho: expected {}, got {}",
        expected_rho,
        greeks.rho
    );
}

// ---------------------------------------------------------------------------
// Cost-of-carry forward pricing
// ---------------------------------------------------------------------------

#[test]
fn cost_of_carry_forward_contango() {
    // F = S * exp((r + u - y) * T)
    // S=100, r=0.05, u=0.01, y=0.02, T=1.0 => F = 100 * exp(0.04) = 104.0811...
    let fwd = CommodityForward {
        spot: 100.0,
        strike: 100.0,
        notional: 1.0,
        risk_free_rate: 0.05,
        storage_cost: 0.01,
        convenience_yield: 0.02,
        maturity: 1.0,
        is_long: true,
    };

    let f = fwd.theoretical_forward_price();
    let expected = 100.0 * (0.04_f64).exp();

    assert!(
        (f - expected).abs() < 1e-10,
        "cost-of-carry forward: expected {}, got {}",
        expected,
        f
    );
    assert!(
        (f - 104.08107741923882).abs() < 1e-8,
        "absolute forward value: expected ~104.0811, got {}",
        f
    );
}

#[test]
fn cost_of_carry_forward_backwardation() {
    // S=50, r=0.03, u=0.005, y=0.04, T=0.5 => F = 50 * exp(-0.005 * 0.5) = 50 * exp(-0.0025)
    let fwd = CommodityForward {
        spot: 50.0,
        strike: 50.0,
        notional: 1.0,
        risk_free_rate: 0.03,
        storage_cost: 0.005,
        convenience_yield: 0.04,
        maturity: 0.5,
        is_long: true,
    };

    let f = fwd.theoretical_forward_price();
    let net_carry = 0.03 + 0.005 - 0.04; // = -0.005
    let expected = 50.0 * (net_carry * 0.5_f64).exp();

    assert!(
        (f - expected).abs() < 1e-10,
        "backwardation forward: expected {}, got {}",
        expected,
        f
    );
    assert!(
        f < 50.0,
        "forward should be below spot in backwardation: {}",
        f
    );
}

#[test]
fn cost_of_carry_forward_present_value() {
    // Verify PV = notional * DF * (F - K) for long position
    let spot = 100.0;
    let r = 0.05;
    let u = 0.01;
    let y = 0.02;
    let t = 1.0;
    let k = 102.0;
    let notional = 1000.0;

    let fwd = CommodityForward {
        spot,
        strike: k,
        notional,
        risk_free_rate: r,
        storage_cost: u,
        convenience_yield: y,
        maturity: t,
        is_long: true,
    };

    let pv = fwd.present_value().unwrap();
    let f = spot * ((r + u - y) * t).exp();
    let df = (-r * t).exp();
    let expected_pv = notional * df * (f - k);

    assert!(
        (pv - expected_pv).abs() < 1e-8,
        "PV: expected {}, got {}",
        expected_pv,
        pv
    );
}

// ---------------------------------------------------------------------------
// Implied convenience yield
// ---------------------------------------------------------------------------

#[test]
fn implied_convenience_yield_formula() {
    // y = r + u - ln(F/S) / T
    // S=100, F=102, r=0.05, u=0.01, T=1.0
    let y = implied_convenience_yield(100.0, 102.0, 0.05, 0.01, 1.0).unwrap();
    let expected = 0.05 + 0.01 - (102.0_f64 / 100.0).ln();
    assert!(
        (y - expected).abs() < 1e-12,
        "convenience yield: expected {}, got {}",
        expected,
        y
    );
    assert!(
        (y - 0.040_197_372_7).abs() < 1e-8,
        "absolute convenience yield value: expected ~0.0402, got {}",
        y
    );
}

#[test]
fn implied_convenience_yield_at_cost_of_carry_forward() {
    // When F = S * exp((r+u-y)*T), the implied yield should recover y
    let s = 80.0;
    let r = 0.04;
    let u = 0.02;
    let y_input = 0.03;
    let t = 2.0_f64;
    let f = s * ((r + u - y_input) * t).exp();

    let y_recovered = implied_convenience_yield(s, f, r, u, t).unwrap();
    assert!(
        (y_recovered - y_input).abs() < 1e-12,
        "should recover input convenience yield: expected {}, got {}",
        y_input,
        y_recovered
    );
}

#[test]
fn implied_convenience_yield_edge_cases() {
    // Invalid inputs should return None
    assert!(implied_convenience_yield(0.0, 100.0, 0.05, 0.01, 1.0).is_none());
    assert!(implied_convenience_yield(100.0, 0.0, 0.05, 0.01, 1.0).is_none());
    assert!(implied_convenience_yield(100.0, 100.0, 0.05, 0.01, 0.0).is_none());
    assert!(implied_convenience_yield(-10.0, 100.0, 0.05, 0.01, 1.0).is_none());
}

// ---------------------------------------------------------------------------
// Black-76 put-call parity: C - P = DF * (F - K) for all parameters
// ---------------------------------------------------------------------------

#[test]
fn black76_put_call_parity_various_strikes() {
    let forward = 100.0;
    let vol = 0.25;
    let r = 0.05;
    let t = 1.0;

    for &strike in &[80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0] {
        let call = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
        let put = black76_price(OptionType::Put, forward, strike, r, vol, t).unwrap();
        let df = (-r * t).exp();
        let parity = df * (forward - strike);

        assert!(
            (call - put - parity).abs() < 1e-10,
            "put-call parity violated at K={}: C-P={}, DF*(F-K)={}",
            strike,
            call - put,
            parity
        );
    }
}

#[test]
fn black76_put_call_parity_various_vols() {
    let forward = 100.0;
    let strike = 100.0;
    let r = 0.03;
    let t = 0.5;

    for &vol in &[0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00] {
        let call = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
        let put = black76_price(OptionType::Put, forward, strike, r, vol, t).unwrap();
        let df = (-r * t).exp();
        let parity = df * (forward - strike);

        assert!(
            (call - put - parity).abs() < 1e-10,
            "put-call parity violated at vol={}: C-P={}, DF*(F-K)={}",
            vol,
            call - put,
            parity
        );
    }
}

#[test]
fn black76_put_call_parity_various_maturities() {
    let forward = 100.0;
    let strike = 105.0;
    let r = 0.04;
    let vol = 0.25;

    for &t in &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let call = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
        let put = black76_price(OptionType::Put, forward, strike, r, vol, t).unwrap();
        let df = (-r * t).exp();
        let parity = df * (forward - strike);

        assert!(
            (call - put - parity).abs() < 1e-10,
            "put-call parity violated at T={}: C-P={}, DF*(F-K)={}",
            t,
            call - put,
            parity
        );
    }
}

// ---------------------------------------------------------------------------
// Black-76 with zero volatility: Call = max(DF*(F-K), 0)
// ---------------------------------------------------------------------------

#[test]
fn black76_zero_vol_itm_call() {
    let forward = 100.0;
    let strike = 90.0;
    let r = 0.05;
    let t = 1.0;
    let vol = 0.0;

    let price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
    let df = (-r * t).exp();
    let expected = df * (forward - strike);

    assert!(
        (price - expected).abs() < 1e-10,
        "zero-vol ITM call: expected {}, got {}",
        expected,
        price
    );
}

#[test]
fn black76_zero_vol_otm_call() {
    let forward = 90.0;
    let strike = 100.0;
    let r = 0.05;
    let t = 1.0;
    let vol = 0.0;

    let price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
    assert!(
        price.abs() < 1e-14,
        "zero-vol OTM call should be 0: got {}",
        price
    );
}

#[test]
fn black76_zero_vol_itm_put() {
    let forward = 90.0;
    let strike = 100.0;
    let r = 0.05;
    let t = 1.0;
    let vol = 0.0;

    let price = black76_price(OptionType::Put, forward, strike, r, vol, t).unwrap();
    let df = (-r * t).exp();
    let expected = df * (strike - forward);

    assert!(
        (price - expected).abs() < 1e-10,
        "zero-vol ITM put: expected {}, got {}",
        expected,
        price
    );
}

// ---------------------------------------------------------------------------
// CommodityOption wrapper delegates correctly to Black-76
// ---------------------------------------------------------------------------

#[test]
fn commodity_option_matches_raw_black76() {
    let forward = 100.0;
    let strike = 95.0;
    let vol = 0.30;
    let r = 0.03;
    let t = 1.0;
    let notional = 10_000.0;

    let option = CommodityOption {
        forward,
        strike,
        vol,
        risk_free_rate: r,
        maturity: t,
        notional,
        option_type: OptionType::Call,
    };

    let commodity_price = option.price_black76().unwrap();
    let raw_price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();

    assert!(
        (commodity_price - notional * raw_price).abs() < 1e-8,
        "CommodityOption should equal notional * black76: {} vs {}",
        commodity_price,
        notional * raw_price
    );
}

#[test]
fn commodity_option_put_matches_raw_black76() {
    let forward = 100.0;
    let strike = 105.0;
    let vol = 0.25;
    let r = 0.04;
    let t = 0.5;
    let notional = 5_000.0;

    let option = CommodityOption {
        forward,
        strike,
        vol,
        risk_free_rate: r,
        maturity: t,
        notional,
        option_type: OptionType::Put,
    };

    let commodity_price = option.price_black76().unwrap();
    let raw_price = black76_price(OptionType::Put, forward, strike, r, vol, t).unwrap();

    assert!(
        (commodity_price - notional * raw_price).abs() < 1e-8,
        "CommodityOption put should equal notional * black76: {} vs {}",
        commodity_price,
        notional * raw_price
    );
}

// ---------------------------------------------------------------------------
// CommodityForwardCurve interpolation
// ---------------------------------------------------------------------------

#[test]
fn forward_curve_from_futures_quotes() {
    let quotes = vec![
        FuturesQuote {
            maturity: 0.25,
            price: 50.0,
        },
        FuturesQuote {
            maturity: 0.50,
            price: 51.0,
        },
        FuturesQuote {
            maturity: 1.00,
            price: 53.0,
        },
        FuturesQuote {
            maturity: 2.00,
            price: 55.0,
        },
    ];

    let curve = CommodityForwardCurve::from_futures_quotes(&quotes).unwrap();

    // Exact nodes
    assert!(
        (curve.forward(0.25) - 50.0).abs() < 1e-12,
        "forward at T=0.25"
    );
    assert!(
        (curve.forward(1.00) - 53.0).abs() < 1e-12,
        "forward at T=1.00"
    );

    // Linear interpolation between nodes
    let f_075 = curve.forward(0.75);
    let expected_075 = 51.0 + 0.5 * (53.0 - 51.0); // midpoint of [0.50, 1.00]
    assert!(
        (f_075 - expected_075).abs() < 1e-12,
        "forward at T=0.75: expected {}, got {}",
        expected_075,
        f_075
    );

    // Flat extrapolation
    assert!(
        (curve.forward(0.10) - 50.0).abs() < 1e-12,
        "flat extrapolation below first node"
    );
    assert!(
        (curve.forward(5.00) - 55.0).abs() < 1e-12,
        "flat extrapolation above last node"
    );
}

// ---------------------------------------------------------------------------
// CommoditySpreadOption basic tests
// ---------------------------------------------------------------------------

#[test]
fn commodity_spread_option_crack_spread_matches_scipy_kirk_reference() {
    let spread = CommoditySpreadOption::crack_spread(
        OptionType::Call,
        95.0,    // refined forward
        88.0,    // crude forward
        2.0,     // strike
        2.0,     // refined ratio
        1.0,     // crude ratio
        0.30,    // vol refined
        0.25,    // vol crude
        0.6,     // rho
        0.03,    // risk_free_rate
        0.75,    // maturity
        1_000.0, // notional
    );

    let price = spread.price_kirk().unwrap();
    assert!((price - 97_776.912_290_401_3).abs() <= 3.0e-10);
}

#[test]
fn commodity_spread_option_spark_spread_matches_scipy_kirk_reference() {
    let spread = CommoditySpreadOption::spark_spread(
        OptionType::Call,
        60.0,    // power forward
        5.0,     // gas forward
        0.0,     // strike (at-the-money spread)
        10.0,    // heat rate
        0.40,    // vol power
        0.35,    // vol gas
        0.5,     // rho
        0.03,    // risk_free_rate
        1.0,     // maturity
        1_000.0, // notional
    );

    let price = spread.price_kirk().unwrap();
    assert!(
        (price - 13_743.185_534_017_315).abs() <= 6.0e-11,
        "spark spread call price: {}",
        price
    );
}

#[test]
fn commodity_spread_option_put_call_relation() {
    // For spread options: C - P = DF * (q1*F1 - q2*F2 - K) via Kirk
    let r = 0.03;
    let t = 1.0;
    let f1 = 100.0;
    let f2 = 90.0;
    let k = 5.0;
    let notional = 1.0;

    let call_spread = CommoditySpreadOption {
        option_type: OptionType::Call,
        forward_1: f1,
        forward_2: f2,
        strike: k,
        quantity_1: 1.0,
        quantity_2: 1.0,
        vol_1: 0.25,
        vol_2: 0.20,
        rho: 0.5,
        risk_free_rate: r,
        maturity: t,
        notional,
    };

    let put_spread = CommoditySpreadOption {
        option_type: OptionType::Put,
        ..call_spread
    };

    let call_price = call_spread.price_kirk().unwrap();
    let put_price = put_spread.price_kirk().unwrap();

    let df = (-r * t).exp();
    let intrinsic_fwd = f1 - f2 - k; // q1*F1 - q2*F2 - K with q1=q2=1
    let parity = df * intrinsic_fwd;

    assert!(
        (call_price - put_price - parity).abs() <= 2.0e-14,
        "spread put-call parity: C-P={}, DF*(F1-F2-K)={}",
        call_price - put_price,
        parity
    );
}

// ---------------------------------------------------------------------------
// CommodityFutures mark-to-market
// ---------------------------------------------------------------------------

#[test]
fn commodity_futures_long_pnl() {
    use openferric::instruments::CommodityFutures;

    let futures = CommodityFutures {
        contract_price: 50.0,
        contract_size: 1000.0,
        is_long: true,
    };

    // Price goes up: profit
    let pnl_up = futures.value(55.0).unwrap();
    assert!(
        (pnl_up - 5000.0).abs() < 1e-10,
        "long PnL on price up: expected 5000, got {}",
        pnl_up
    );

    // Price goes down: loss
    let pnl_down = futures.value(48.0).unwrap();
    assert!(
        (pnl_down - (-2000.0)).abs() < 1e-10,
        "long PnL on price down: expected -2000, got {}",
        pnl_down
    );
}

#[test]
fn commodity_futures_short_pnl() {
    use openferric::instruments::CommodityFutures;

    let futures = CommodityFutures {
        contract_price: 50.0,
        contract_size: 1000.0,
        is_long: false,
    };

    // Price goes down: profit for short
    let pnl_down = futures.value(45.0).unwrap();
    assert!(
        (pnl_down - 5000.0).abs() < 1e-10,
        "short PnL on price down: expected 5000, got {}",
        pnl_down
    );

    // Price goes up: loss for short
    let pnl_up = futures.value(52.0).unwrap();
    assert!(
        (pnl_up - (-2000.0)).abs() < 1e-10,
        "short PnL on price up: expected -2000, got {}",
        pnl_up
    );
}

// ---------------------------------------------------------------------------
// Convenience yield term structure
// ---------------------------------------------------------------------------

#[test]
fn convenience_yield_term_structure_recovery() {
    use openferric::models::convenience_yield_from_term_structure;

    let spot = 100.0;
    let r = 0.05;
    let u = 0.01;

    // Build futures prices consistent with known convenience yields
    let yields = [0.02, 0.03, 0.04];
    let maturities = [0.5, 1.0, 2.0];
    let quotes: Vec<FuturesQuote> = maturities
        .iter()
        .zip(yields.iter())
        .map(|(&t, &y)| FuturesQuote {
            maturity: t,
            price: spot * ((r + u - y) * t).exp(),
        })
        .collect();

    let result = convenience_yield_from_term_structure(spot, &quotes, r, u).unwrap();

    for (i, &(mat, yield_val)) in result.iter().enumerate() {
        assert!(
            (mat - maturities[i]).abs() < 1e-12,
            "maturity mismatch at index {}",
            i
        );
        assert!(
            (yield_val - yields[i]).abs() < 1e-10,
            "convenience yield mismatch at T={}: expected {}, got {}",
            mat,
            yields[i],
            yield_val
        );
    }
}

// ---------------------------------------------------------------------------
// Black-76 monotonicity and boundary properties
// ---------------------------------------------------------------------------

#[test]
fn black76_call_price_decreases_with_strike() {
    let forward = 100.0;
    let r = 0.03;
    let vol = 0.25;
    let t = 1.0;

    let mut prev = f64::MAX;
    for &k in &[80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0] {
        let price = black76_price(OptionType::Call, forward, k, r, vol, t).unwrap();
        assert!(
            price < prev,
            "call price should decrease with strike: K={}, price={}",
            k,
            price
        );
        prev = price;
    }
}

#[test]
fn black76_put_price_increases_with_strike() {
    let forward = 100.0;
    let r = 0.03;
    let vol = 0.25;
    let t = 1.0;

    let mut prev = 0.0_f64;
    for &k in &[80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0] {
        let price = black76_price(OptionType::Put, forward, k, r, vol, t).unwrap();
        assert!(
            price > prev,
            "put price should increase with strike: K={}, price={}",
            k,
            price
        );
        prev = price;
    }
}

#[test]
fn black76_call_price_increases_with_vol() {
    let forward = 100.0;
    let strike = 100.0;
    let r = 0.03;
    let t = 1.0;

    let mut prev = 0.0_f64;
    for &vol in &[0.05, 0.10, 0.20, 0.30, 0.50, 0.80] {
        let price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
        assert!(
            price > prev,
            "ATM call price should increase with vol: vol={}, price={}",
            vol,
            price
        );
        prev = price;
    }
}

// ---------------------------------------------------------------------------
// Deep ITM/OTM boundary behavior
// ---------------------------------------------------------------------------

#[test]
fn black76_deep_itm_call_matches_scipy_closed_form() {
    let forward = 100.0;
    let strike = 10.0; // very deep ITM
    let r = 0.05;
    let vol = 0.20;
    let t = 1.0;

    let price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
    let expected = 85.610_648_205_064_27;
    assert!(
        (price - expected).abs() <= 2.0e-13,
        "deep ITM call: price={} expected={expected}",
        price,
    );
}

#[test]
fn black76_deep_otm_call_matches_scipy_tail_value() {
    let forward = 100.0;
    let strike = 300.0; // very deep OTM
    let r = 0.05;
    let vol = 0.20;
    let t = 1.0;

    let price = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();
    let expected = 1.111_590_309_260_426_6e-7;
    assert!(
        (price - expected).abs() <= 2.0e-18,
        "price={price} expected={expected}"
    );
}

// ---------------------------------------------------------------------------
// Issue #64 acceptance criteria
// ---------------------------------------------------------------------------

#[test]
fn forward_curve_reproduces_futures_at_contract_dates() {
    let quotes = vec![
        FuturesQuote {
            maturity: 1.0 / 12.0,
            price: 2.90,
        },
        FuturesQuote {
            maturity: 2.0 / 12.0,
            price: 2.95,
        },
        FuturesQuote {
            maturity: 3.0 / 12.0,
            price: 3.05,
        },
        FuturesQuote {
            maturity: 4.0 / 12.0,
            price: 3.15,
        },
        FuturesQuote {
            maturity: 6.0 / 12.0,
            price: 3.30,
        },
    ];

    for method in [
        ForwardInterpolation::PiecewiseFlat,
        ForwardInterpolation::Linear,
        ForwardInterpolation::CubicSpline,
    ] {
        let curve = CommodityForwardCurve::from_futures_quotes_with_interpolation(&quotes, method)
            .expect("curve should build");
        for q in &quotes {
            let fitted = curve.forward(q.maturity);
            assert!(
                (fitted - q.price).abs() < 1.0e-12,
                "method {:?} should match futures exactly at T={}: got {}, expected {}",
                method,
                q.maturity,
                fitted,
                q.price
            );
        }
    }
}

#[test]
fn seasonal_model_captures_nat_gas_winter_summer_pattern() {
    let seasonality = CommoditySeasonalityModel::natural_gas_winter_summer(
        SeasonalityMode::Multiplicative,
        1.20,
        0.85,
        1.0,
    )
    .unwrap();

    let winter = seasonality.apply(100.0, 1).unwrap();
    let shoulder = seasonality.apply(100.0, 4).unwrap();
    let summer = seasonality.apply(100.0, 7).unwrap();

    assert!(
        winter > shoulder && shoulder > summer,
        "expected winter > shoulder > summer, got winter={} shoulder={} summer={}",
        winter,
        shoulder,
        summer
    );
}

#[test]
fn two_factor_spread_mc_matches_exact_margrabe_reduction() {
    let maturity = 1.0_f64;
    let model = TwoFactorSpreadModel {
        leg_1: TwoFactorCommodityProcess {
            kappa_fast: 2.5,
            sigma_fast: 0.22,
            sigma_slow: 0.26,
        },
        leg_2: TwoFactorCommodityProcess {
            kappa_fast: 2.2,
            sigma_fast: 0.18,
            sigma_slow: 0.20,
        },
        rho_fast: 0.60,
        rho_slow: 0.56,
    };

    // The model's two independent factors sum to a jointly lognormal terminal
    // forward. At K=0, Kirk is exactly Margrabe when supplied with the
    // integrated marginal variances and covariance of those factors.
    let fast_var_1 = model.leg_1.sigma_fast.powi(2)
        * (1.0 - (-2.0 * model.leg_1.kappa_fast * maturity).exp())
        / (2.0 * model.leg_1.kappa_fast);
    let fast_var_2 = model.leg_2.sigma_fast.powi(2)
        * (1.0 - (-2.0 * model.leg_2.kappa_fast * maturity).exp())
        / (2.0 * model.leg_2.kappa_fast);
    let slow_var_1 = model.leg_1.sigma_slow.powi(2) * maturity;
    let slow_var_2 = model.leg_2.sigma_slow.powi(2) * maturity;
    let var_1 = fast_var_1 + slow_var_1;
    let var_2 = fast_var_2 + slow_var_2;
    let covariance = model.rho_fast * (fast_var_1 * fast_var_2).sqrt()
        + model.rho_slow * (slow_var_1 * slow_var_2).sqrt();
    let effective_rho = covariance / (var_1 * var_2).sqrt();

    let spread = CommoditySpreadOption::crack_spread(
        OptionType::Call,
        95.0,
        88.0,
        0.0,
        1.0,
        1.0,
        (var_1 / maturity).sqrt(),
        (var_2 / maturity).sqrt(),
        effective_rho,
        0.03,
        maturity,
        1.0,
    );
    let margrabe = spread.price_kirk().unwrap();
    let (mc, stderr) = spread.price_two_factor_mc(&model, 400_000, 101).unwrap();
    let tolerance = 4.0 * stderr + 1.0e-12;
    assert!(
        (mc - margrabe).abs() <= tolerance,
        "two-factor MC/Margrabe error exceeds sampling budget: exact={} mc={} stderr={} tolerance={}",
        margrabe,
        mc,
        stderr,
        tolerance
    );
}
