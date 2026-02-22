//! Rainbow option, exchange option, and spread option reference tests.
//!
//! Sources:
//! - Margrabe (1978) exchange option formula
//! - Stulz (1982) rainbow option (call on max/min of two assets)
//! - Kirk (1995) spread option approximation
//! - Haug "The Complete Guide to Option Pricing Formulas" (1998), pp. 52-60
//! - QuantLib vendored test suite: margrabeoption.cpp, basketoption.cpp

use openferric::engines::analytic::spread::{kirk_spread_price, margrabe_exchange_price};
use openferric::instruments::rainbow::{BestOfTwoCallOption, WorstOfTwoCallOption};
use openferric::instruments::spread::SpreadOption;

// =======================================================================
// Margrabe exchange option: payoff = max(S1 - S2, 0)
// QuantLib margrabeoption.cpp, Haug p.52
// Common: S1=22, S2=20, q1=0.06, q2=0.04, r=0.10, vol1=0.20
// =======================================================================

fn margrabe_case(t: f64, vol2: f64, rho: f64, expected: f64, tol: f64) {
    let option = SpreadOption {
        s1: 22.0,
        s2: 20.0,
        k: 0.0,
        vol1: 0.20,
        vol2,
        rho,
        q1: 0.06,
        q2: 0.04,
        r: 0.10,
        t,
    };
    let price = margrabe_exchange_price(&option).expect("Margrabe pricing failed");
    let err = (price - expected).abs();
    assert!(
        err < tol,
        "Margrabe T={t} vol2={vol2} rho={rho}: got {price}, expected {expected}, err={err}"
    );
}

#[test]
fn margrabe_haug_t010_vol2_015_rho_neg050() {
    margrabe_case(0.10, 0.15, -0.50, 2.125, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_020_rho_neg050() {
    margrabe_case(0.10, 0.20, -0.50, 2.199, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_025_rho_neg050() {
    margrabe_case(0.10, 0.25, -0.50, 2.283, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_015_rho_000() {
    margrabe_case(0.10, 0.15, 0.00, 2.045, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_020_rho_000() {
    margrabe_case(0.10, 0.20, 0.00, 2.091, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_025_rho_000() {
    margrabe_case(0.10, 0.25, 0.00, 2.152, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_015_rho_050() {
    margrabe_case(0.10, 0.15, 0.50, 1.974, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_020_rho_050() {
    margrabe_case(0.10, 0.20, 0.50, 1.989, 0.01);
}

#[test]
fn margrabe_haug_t010_vol2_025_rho_050() {
    margrabe_case(0.10, 0.25, 0.50, 2.019, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_015_rho_neg050() {
    margrabe_case(0.50, 0.15, -0.50, 2.762, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_020_rho_neg050() {
    margrabe_case(0.50, 0.20, -0.50, 2.989, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_025_rho_neg050() {
    margrabe_case(0.50, 0.25, -0.50, 3.228, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_015_rho_000() {
    margrabe_case(0.50, 0.15, 0.00, 2.479, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_020_rho_000() {
    margrabe_case(0.50, 0.20, 0.00, 2.650, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_025_rho_000() {
    margrabe_case(0.50, 0.25, 0.00, 2.847, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_015_rho_050() {
    margrabe_case(0.50, 0.15, 0.50, 2.138, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_020_rho_050() {
    margrabe_case(0.50, 0.20, 0.50, 2.231, 0.01);
}

#[test]
fn margrabe_haug_t050_vol2_025_rho_050() {
    margrabe_case(0.50, 0.25, 0.50, 2.374, 0.01);
}

// =======================================================================
// Stulz call on max of two assets (best-of-two call)
// QuantLib basketoption.cpp, Haug p.56-58
// =======================================================================

fn best_of_two_case(
    s1: f64,
    s2: f64,
    k: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    t: f64,
    expected: f64,
    tol: f64,
) {
    let option = BestOfTwoCallOption {
        s1,
        s2,
        k,
        vol1,
        vol2,
        rho,
        q1: 0.0,
        q2: 0.0,
        r: 0.05,
        t,
    };
    let price = openferric::engines::analytic::rainbow::best_of_two_call_price(&option)
        .expect("best-of-two pricing failed");
    let err = (price - expected).abs();
    assert!(
        err < tol,
        "BestOfTwo S1={s1} S2={s2} K={k} rho={rho}: got {price}, expected {expected}, err={err}"
    );
}

#[test]
fn stulz_best_of_two_rho090() {
    best_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.90, 1.0, 17.565, 0.01);
}

#[test]
fn stulz_best_of_two_rho070() {
    best_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.70, 1.0, 19.980, 0.01);
}

#[test]
fn stulz_best_of_two_rho050() {
    best_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.50, 1.0, 21.619, 0.01);
}

#[test]
fn stulz_best_of_two_rho030() {
    best_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.30, 1.0, 22.932, 0.01);
}

#[test]
fn stulz_best_of_two_rho010() {
    best_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.10, 1.0, 24.049, 0.05);
}

#[test]
fn stulz_best_of_two_s1_80_s2_100() {
    best_of_two_case(80.0, 100.0, 100.0, 0.30, 0.30, 0.30, 1.0, 16.508, 0.01);
}

#[test]
fn stulz_best_of_two_s1_80_s2_80() {
    best_of_two_case(80.0, 80.0, 100.0, 0.30, 0.30, 0.30, 1.0, 8.049, 0.01);
}

#[test]
fn stulz_best_of_two_s1_80_s2_120() {
    best_of_two_case(80.0, 120.0, 100.0, 0.30, 0.30, 0.30, 1.0, 30.141, 0.01);
}

#[test]
fn stulz_best_of_two_s1_120_s2_120() {
    best_of_two_case(120.0, 120.0, 100.0, 0.30, 0.30, 0.30, 1.0, 42.889, 0.01);
}

// =======================================================================
// Stulz call on min of two assets (worst-of-two call)
// =======================================================================

fn worst_of_two_case(
    s1: f64,
    s2: f64,
    k: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    t: f64,
    expected: f64,
    tol: f64,
) {
    let option = WorstOfTwoCallOption {
        s1,
        s2,
        k,
        vol1,
        vol2,
        rho,
        q1: 0.0,
        q2: 0.0,
        r: 0.05,
        t,
    };
    let price = openferric::engines::analytic::rainbow::worst_of_two_call_price(&option)
        .expect("worst-of-two pricing failed");
    let err = (price - expected).abs();
    assert!(
        err < tol,
        "WorstOfTwo S1={s1} S2={s2} K={k} rho={rho}: got {price}, expected {expected}, err={err}"
    );
}

#[test]
fn stulz_worst_of_two_rho090() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.90, 1.0, 10.898, 0.01);
}

#[test]
fn stulz_worst_of_two_rho070() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.70, 1.0, 8.483, 0.01);
}

#[test]
fn stulz_worst_of_two_rho050() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.50, 1.0, 6.844, 0.01);
}

#[test]
fn stulz_worst_of_two_rho030() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.30, 1.0, 5.531, 0.01);
}

#[test]
fn stulz_worst_of_two_rho010() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.30, 0.30, 0.10, 1.0, 4.413, 0.01);
}

#[test]
fn stulz_worst_of_two_asymmetric_vols() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.50, 0.70, 0.00, 1.0, 4.981, 0.01);
}

#[test]
fn stulz_worst_of_two_vol1_050_vol2_030() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.50, 0.30, 0.00, 1.0, 4.159, 0.01);
}

#[test]
fn stulz_worst_of_two_vol1_050_vol2_010() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.50, 0.10, 0.00, 1.0, 2.597, 0.01);
}

#[test]
fn stulz_worst_of_two_vol1_050_vol2_010_rho050() {
    worst_of_two_case(100.0, 100.0, 100.0, 0.50, 0.10, 0.50, 1.0, 4.030, 0.01);
}

// =======================================================================
// Haug p.56-58: with dividends
// =======================================================================

fn haug_dividend_best_of_two(q1: f64, q2: f64, expected: f64, tol: f64) {
    let option = BestOfTwoCallOption {
        s1: 100.0,
        s2: 105.0,
        k: 98.0,
        vol1: 0.11,
        vol2: 0.16,
        rho: 0.63,
        q1,
        q2,
        r: 0.05,
        t: 0.5,
    };
    let price = openferric::engines::analytic::rainbow::best_of_two_call_price(&option)
        .expect("best-of-two pricing failed");
    let err = (price - expected).abs();
    assert!(
        err < tol,
        "Haug dividend best-of-two q1={q1} q2={q2}: got {price}, expected {expected}, err={err}"
    );
}

#[test]
fn haug_best_of_two_no_dividends() {
    haug_dividend_best_of_two(0.0, 0.0, 11.6323, 0.01);
}

#[test]
fn haug_best_of_two_with_dividends() {
    haug_dividend_best_of_two(0.06, 0.09, 8.0701, 0.01);
}

fn haug_dividend_worst_of_two(q1: f64, q2: f64, expected: f64, tol: f64) {
    let option = WorstOfTwoCallOption {
        s1: 100.0,
        s2: 105.0,
        k: 98.0,
        vol1: 0.11,
        vol2: 0.16,
        rho: 0.63,
        q1,
        q2,
        r: 0.05,
        t: 0.5,
    };
    let price = openferric::engines::analytic::rainbow::worst_of_two_call_price(&option)
        .expect("worst-of-two pricing failed");
    let err = (price - expected).abs();
    assert!(
        err < tol,
        "Haug dividend worst-of-two q1={q1} q2={q2}: got {price}, expected {expected}, err={err}"
    );
}

#[test]
fn haug_worst_of_two_no_dividends() {
    haug_dividend_worst_of_two(0.0, 0.0, 4.8177, 0.01);
}

#[test]
fn haug_worst_of_two_with_dividends() {
    haug_dividend_worst_of_two(0.06, 0.09, 2.9340, 0.01);
}

// =======================================================================
// Kirk spread option: payoff = max(S1 - S2 - K, 0)
// QuantLib basketoption.cpp, Haug pp.59-60
// Common: K=3, S1=122, S2=120, q1=q2=0, r=0.10
// =======================================================================

fn kirk_case(t: f64, vol1: f64, vol2: f64, rho: f64, expected: f64, tol: f64) {
    // Haug/QuantLib uses BlackProcess (futures-style): q1=q2=r so F=S
    let option = SpreadOption {
        s1: 122.0,
        s2: 120.0,
        k: 3.0,
        vol1,
        vol2,
        rho,
        q1: 0.10,
        q2: 0.10,
        r: 0.10,
        t,
    };
    let price = kirk_spread_price(&option).expect("Kirk pricing failed");
    let err = (price - expected).abs();
    assert!(
        err < tol,
        "Kirk T={t} vol1={vol1} vol2={vol2} rho={rho}: got {price}, expected {expected}, err={err}"
    );
}

#[test]
fn kirk_haug_t010_v020_v020_rho_neg050() {
    kirk_case(0.10, 0.20, 0.20, -0.50, 4.7530, 0.01);
}

#[test]
fn kirk_haug_t010_v020_v020_rho_000() {
    kirk_case(0.10, 0.20, 0.20, 0.00, 3.7970, 0.01);
}

#[test]
fn kirk_haug_t010_v020_v020_rho_050() {
    kirk_case(0.10, 0.20, 0.20, 0.50, 2.5537, 0.01);
}

#[test]
fn kirk_haug_t010_v025_v020_rho_neg050() {
    kirk_case(0.10, 0.25, 0.20, -0.50, 5.4275, 0.01);
}

#[test]
fn kirk_haug_t010_v025_v020_rho_000() {
    kirk_case(0.10, 0.25, 0.20, 0.00, 4.3712, 0.01);
}

#[test]
fn kirk_haug_t010_v025_v020_rho_050() {
    kirk_case(0.10, 0.25, 0.20, 0.50, 3.0086, 0.01);
}

#[test]
fn kirk_haug_t010_v020_v025_rho_neg050() {
    kirk_case(0.10, 0.20, 0.25, -0.50, 5.4061, 0.01);
}

#[test]
fn kirk_haug_t010_v020_v025_rho_000() {
    kirk_case(0.10, 0.20, 0.25, 0.00, 4.3451, 0.01);
}

#[test]
fn kirk_haug_t010_v020_v025_rho_050() {
    kirk_case(0.10, 0.20, 0.25, 0.50, 2.9723, 0.01);
}

#[test]
fn kirk_haug_t050_v020_v020_rho_neg050() {
    kirk_case(0.50, 0.20, 0.20, -0.50, 10.7517, 0.01);
}

#[test]
fn kirk_haug_t050_v020_v020_rho_000() {
    kirk_case(0.50, 0.20, 0.20, 0.00, 8.7020, 0.01);
}

#[test]
fn kirk_haug_t050_v020_v020_rho_050() {
    kirk_case(0.50, 0.20, 0.20, 0.50, 6.0257, 0.01);
}

#[test]
fn kirk_haug_t050_v025_v020_rho_neg050() {
    kirk_case(0.50, 0.25, 0.20, -0.50, 12.1941, 0.01);
}

#[test]
fn kirk_haug_t050_v025_v020_rho_000() {
    kirk_case(0.50, 0.25, 0.20, 0.00, 9.9340, 0.01);
}

#[test]
fn kirk_haug_t050_v025_v020_rho_050() {
    kirk_case(0.50, 0.25, 0.20, 0.50, 7.0067, 0.01);
}

#[test]
fn kirk_haug_t050_v020_v025_rho_neg050() {
    kirk_case(0.50, 0.20, 0.25, -0.50, 12.1483, 0.01);
}

#[test]
fn kirk_haug_t050_v020_v025_rho_000() {
    kirk_case(0.50, 0.20, 0.25, 0.00, 9.8780, 0.01);
}

#[test]
fn kirk_haug_t050_v020_v025_rho_050() {
    kirk_case(0.50, 0.20, 0.25, 0.50, 6.9284, 0.01);
}
