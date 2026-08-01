//! Discrete cash/proportional dividend regression tests for early-exercise
//! engines (audit finding: dividend smear via `effective_dividend_yield`).
//!
//! All early-exercise engines previously converted discrete dividends into a
//! constant continuous yield over the whole lattice/grid/path set. That is
//! forward-matching (exact for Europeans under the escrowed model) but wrong
//! for Americans: the exercise incentive concentrated at each ex-date is
//! smeared over the full horizon. On the probe scenario below the effective
//! yield (~5.05%) exceeded the 3% rate and suppressed put exercise entirely,
//! producing an early-exercise premium of ~0.001 where the escrowed-model
//! premium is ~0.35.
//!
//! The library's discrete-dividend convention for vanilla options is the
//! ESCROWED model (Haug-Haug-Lewis), already used by the analytic
//! `BlackScholesEngine`: the tradable component `S* = S - PV(remaining cash
//! dividends)` diffuses as a dividend-free GBM (with the true continuous
//! yield), and exercise payoffs reconstruct the observed spot at each date.
//! Barrier Monte Carlo engines instead apply true ex-date drops on the path
//! (spot model), because knock events depend on the observed spot path.

use openferric::core::{ExerciseStyle, OptionType, PricingEngine};
use openferric::engines::analytic::BlackScholesEngine;
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::engines::numerical::american_binomial::AmericanBinomialEngine;
use openferric::engines::pde::{
    CrankNicolsonEngine, ExplicitFdEngine, HopscotchEngine, ImplicitFdEngine,
};
use openferric::engines::tree::binomial::BinomialTreeEngine;
use openferric::engines::tree::trinomial::TrinomialTreeEngine;
use openferric::instruments::{BarrierOption, BermudanOption, VanillaOption};
use openferric::market::{DividendEvent, DividendSchedule, Market};

/// Probe scenario: S = K = 100, r = 3%, q = 0, vol = 25%, T = 1y, one cash
/// dividend of 5.0 at t = 0.5.
fn probe_market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.0)
        .dividend_schedule(
            DividendSchedule::new(vec![DividendEvent::cash(0.5, 5.0).expect("valid event")])
                .expect("valid schedule"),
        )
        .flat_vol(0.25)
        .build()
        .expect("valid market")
}

/// Escrowed-model analytic European put for the probe scenario.
///
/// Hand-checkable: S* = 100 - 5 e^{-0.03*0.5} = 95.074441, then
/// Black-Scholes(S*, K=100, r=3%, q=0, vol=25%, T=1) = 10.5727. This value
/// was independently reproduced during audit verification (escrowed BS by
/// hand: 10.572688; library analytic engine: 10.572685).
const PROBE_ESCROWED_EUROPEAN_PUT: f64 = 10.5727;

/// Escrowed-model American put for the probe scenario.
///
/// Reference: independent escrowed-spot CRR implementation from the audit
/// verification (S* diffused GBM(r, vol), exercise payoff K - (S* + PV_t of
/// remaining dividends)), 8000 steps: American put = 10.9246, early-exercise
/// premium ~0.352 over the escrowed European 10.5727.
const PROBE_ESCROWED_AMERICAN_PUT: f64 = 10.9246;

#[test]
fn european_engines_match_escrowed_analytic_with_cash_dividend() {
    let market = probe_market();
    let put = VanillaOption::european_put(100.0, 1.0);
    let call = VanillaOption::european_call(100.0, 1.0);

    let analytic_put = BlackScholesEngine.price(&put, &market).unwrap().price;
    let analytic_call = BlackScholesEngine.price(&call, &market).unwrap().price;
    assert!(
        (analytic_put - PROBE_ESCROWED_EUROPEAN_PUT).abs() <= 1.0e-3,
        "analytic escrowed European put drifted: {analytic_put}"
    );

    // Requirement (1): European prices through the early-exercise engines
    // must stay forward-matching, i.e. converge to the escrowed analytic.
    let checks: Vec<(&str, f64, f64, f64)> = vec![
        (
            "binomial put",
            BinomialTreeEngine::new(2000)
                .price(&put, &market)
                .unwrap()
                .price,
            analytic_put,
            0.01,
        ),
        (
            "binomial call",
            BinomialTreeEngine::new(2000)
                .price(&call, &market)
                .unwrap()
                .price,
            analytic_call,
            0.01,
        ),
        (
            "trinomial put",
            TrinomialTreeEngine::new(1000)
                .price(&put, &market)
                .unwrap()
                .price,
            analytic_put,
            0.01,
        ),
        (
            "crank-nicolson put",
            CrankNicolsonEngine::new(800, 800)
                .price(&put, &market)
                .unwrap()
                .price,
            analytic_put,
            0.02,
        ),
        (
            "implicit fd put",
            ImplicitFdEngine::default()
                .price(&put, &market)
                .unwrap()
                .price,
            analytic_put,
            0.02,
        ),
        (
            "explicit fd put",
            ExplicitFdEngine::default()
                .price(&put, &market)
                .unwrap()
                .price,
            analytic_put,
            0.02,
        ),
        (
            "hopscotch put",
            HopscotchEngine::default()
                .price(&put, &market)
                .unwrap()
                .price,
            analytic_put,
            0.02,
        ),
    ];
    for (label, price, reference, tol) in checks {
        assert!(
            (price - reference).abs() <= tol,
            "{label}: engine={price}, escrowed analytic={reference}"
        );
    }

    // LSM European (exercise only at expiry) is unbiased for the escrowed
    // European; allow Monte Carlo noise.
    let lsm = LongstaffSchwartzEngine::new(60_000, 50, 42)
        .price(&put, &market)
        .unwrap();
    let tol = 4.0 * lsm.stderr.unwrap_or(0.0) + 1.0e-6;
    assert!(
        (lsm.price - analytic_put).abs() <= tol,
        "lsm european put: {} vs analytic {} (tol {tol})",
        lsm.price,
        analytic_put
    );
}

#[test]
fn hull_dividend_european_call_reference() {
    // External anchor: Hull, "Options, Futures, and Other Derivatives"
    // (5th ed.), Exercise 12.8 — also cached as 3.67 in QuantLib's
    // test-suite/dividendoption.cpp (testEuropeanKnownValue, Actual/360):
    // S=40, K=40, r=9%, vol=30%, T=180/360, cash dividends 0.50 at 60/360
    // and 0.50 at 150/360. Escrowed European call = 3.67.
    let market = Market::builder()
        .spot(40.0)
        .rate(0.09)
        .dividend_yield(0.0)
        .dividend_schedule(
            DividendSchedule::new(vec![
                DividendEvent::cash(60.0 / 360.0, 0.50).expect("valid event"),
                DividendEvent::cash(150.0 / 360.0, 0.50).expect("valid event"),
            ])
            .expect("valid schedule"),
        )
        .flat_vol(0.30)
        .build()
        .expect("valid market");
    let call = VanillaOption::european_call(40.0, 0.5);

    let analytic = BlackScholesEngine.price(&call, &market).unwrap().price;
    let tree = BinomialTreeEngine::new(2000)
        .price(&call, &market)
        .unwrap()
        .price;
    let pde = CrankNicolsonEngine::new(800, 800)
        .price(&call, &market)
        .unwrap()
        .price;

    for (label, price) in [("analytic", analytic), ("binomial", tree), ("cn", pde)] {
        assert!(
            (price - 3.67).abs() <= 0.02,
            "{label}: {price} vs Hull reference 3.67"
        );
    }
}

#[test]
fn american_put_cash_dividend_has_material_early_exercise_premium() {
    let market = probe_market();
    let am_put = VanillaOption::american_put(100.0, 1.0);
    let eu_put = VanillaOption::european_put(100.0, 1.0);

    let analytic_eu = BlackScholesEngine.price(&eu_put, &market).unwrap().price;

    // Requirement (2): a materially positive early-exercise premium. The
    // pre-fix engines produced ~0.001 here (q_eff > r suppressed exercise).
    let binomial = BinomialTreeEngine::new(2000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert!(
        (binomial - PROBE_ESCROWED_AMERICAN_PUT).abs() <= 0.02,
        "binomial American put {binomial} vs escrowed CRR reference {PROBE_ESCROWED_AMERICAN_PUT}"
    );
    let premium = binomial - analytic_eu;
    assert!(
        premium >= 0.30,
        "early-exercise premium {premium} should be ~0.35, got binomial={binomial}, european={analytic_eu}"
    );

    // American call premium must also come from the discrete-dividend
    // mechanism: calls on cash-dividend stocks exercise just before ex-dates.
    let am_call = VanillaOption::american_call(100.0, 1.0);
    let eu_call = VanillaOption::european_call(100.0, 1.0);
    let am_call_px = BinomialTreeEngine::new(2000)
        .price(&am_call, &market)
        .unwrap()
        .price;
    let eu_call_px = BlackScholesEngine.price(&eu_call, &market).unwrap().price;
    assert!(
        am_call_px > eu_call_px + 0.05,
        "American call {am_call_px} should carry a dividend-driven premium over European {eu_call_px}"
    );
}

#[test]
fn cross_engine_agreement_american_put_with_cash_dividend() {
    // Requirement (3): all engines implementing the escrowed model agree.
    let market = probe_market();
    let am_put = VanillaOption::american_put(100.0, 1.0);

    let reference = BinomialTreeEngine::new(4000)
        .price(&am_put, &market)
        .unwrap()
        .price;

    let american_binomial = AmericanBinomialEngine::new(4000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert!(
        (american_binomial - reference).abs() <= 1.0e-9,
        "american_binomial {american_binomial} vs binomial {reference}"
    );

    let checks: Vec<(&str, f64, f64)> = vec![
        (
            "trinomial",
            TrinomialTreeEngine::new(1500)
                .price(&am_put, &market)
                .unwrap()
                .price,
            0.02,
        ),
        (
            "crank-nicolson",
            CrankNicolsonEngine::new(800, 800)
                .price(&am_put, &market)
                .unwrap()
                .price,
            0.05,
        ),
        (
            "implicit fd",
            ImplicitFdEngine::default()
                .price(&am_put, &market)
                .unwrap()
                .price,
            0.05,
        ),
        (
            "explicit fd",
            ExplicitFdEngine::default()
                .price(&am_put, &market)
                .unwrap()
                .price,
            0.05,
        ),
        (
            "hopscotch",
            HopscotchEngine::default()
                .price(&am_put, &market)
                .unwrap()
                .price,
            0.05,
        ),
    ];
    for (label, price, tol) in checks {
        assert!(
            (price - reference).abs() <= tol,
            "{label}: {price} vs binomial reference {reference}"
        );
    }

    // LSM is a lower-bound estimator with regression bias; allow a wider
    // band but require it to sit near the escrowed American value, far above
    // the pre-fix smeared value (~10.57).
    let lsm = LongstaffSchwartzEngine::new(60_000, 100, 42)
        .price(&am_put, &market)
        .unwrap();
    let tol = 0.10 + 4.0 * lsm.stderr.unwrap_or(0.0);
    assert!(
        (lsm.price - reference).abs() <= tol,
        "lsm american put {} vs binomial reference {} (tol {tol})",
        lsm.price,
        reference
    );
}

#[test]
fn proportional_dividend_escrowed_consistency() {
    // 5% proportional dividend at t = 0.5 (requirement (3) uses a 5%
    // dividend; the proportional case must be escrowed-consistent too).
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.0)
        .dividend_schedule(
            DividendSchedule::new(vec![
                DividendEvent::proportional(0.5, 0.05).expect("valid event"),
            ])
            .expect("valid schedule"),
        )
        .flat_vol(0.25)
        .build()
        .expect("valid market");

    let eu_put = VanillaOption::european_put(100.0, 1.0);
    let am_put = VanillaOption::american_put(100.0, 1.0);

    // Europeans stay forward-matching against the escrowed analytic engine.
    let analytic_eu = BlackScholesEngine.price(&eu_put, &market).unwrap().price;
    let tree_eu = BinomialTreeEngine::new(2000)
        .price(&eu_put, &market)
        .unwrap()
        .price;
    assert!(
        (tree_eu - analytic_eu).abs() <= 0.01,
        "european binomial {tree_eu} vs analytic {analytic_eu}"
    );

    // American engines agree with each other and carry a real premium.
    let tree_am = BinomialTreeEngine::new(2000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    let tri_am = TrinomialTreeEngine::new(1000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    let cn_am = CrankNicolsonEngine::new(800, 800)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert!(
        (tree_am - tri_am).abs() <= 0.02,
        "binomial {tree_am} vs trinomial {tri_am}"
    );
    assert!(
        (tree_am - cn_am).abs() <= 0.05,
        "binomial {tree_am} vs crank-nicolson {cn_am}"
    );
    let premium = tree_am - analytic_eu;
    assert!(
        premium >= 0.10,
        "proportional-dividend American put premium too small: {premium}"
    );
}

#[test]
fn bermudan_engines_respect_escrowed_dividends() {
    let market = probe_market();
    let dates = vec![0.25, 0.5, 0.75];

    let mut bermudan_vanilla = VanillaOption::american_put(100.0, 1.0);
    bermudan_vanilla.exercise = ExerciseStyle::Bermudan {
        dates: dates.clone(),
    };
    let european = VanillaOption::european_put(100.0, 1.0);
    let american = VanillaOption::american_put(100.0, 1.0);

    let tree_engine = BinomialTreeEngine::new(2000);
    let eu = BlackScholesEngine.price(&european, &market).unwrap().price;
    let berm_tree = tree_engine.price(&bermudan_vanilla, &market).unwrap().price;
    let am = tree_engine.price(&american, &market).unwrap().price;

    assert!(
        eu - 1.0e-9 <= berm_tree && berm_tree <= am + 1.0e-9,
        "bermudan {berm_tree} must sit between european {eu} and american {am}"
    );
    // The 0.5 exercise date sits right at the ex-date, so the Bermudan must
    // capture most of the American early-exercise premium.
    assert!(
        berm_tree - eu >= 0.15,
        "bermudan premium too small: {} (eu={eu})",
        berm_tree - eu
    );

    // Dedicated Bermudan instruments through CN and LSM agree with the tree.
    let bermudan = BermudanOption::new(
        OptionType::Put,
        1.0,
        dates.clone(),
        vec![100.0, 100.0, 100.0],
    );
    let cn = CrankNicolsonEngine::new(800, 800)
        .price(&bermudan, &market)
        .unwrap()
        .price;
    assert!(
        (cn - berm_tree).abs() <= 0.05,
        "cn bermudan {cn} vs tree bermudan {berm_tree}"
    );

    let lsm = LongstaffSchwartzEngine::new(60_000, 100, 42)
        .price(&bermudan, &market)
        .unwrap();
    let tol = 0.10 + 4.0 * lsm.stderr.unwrap_or(0.0);
    assert!(
        (lsm.price - berm_tree).abs() <= tol,
        "lsm bermudan {} vs tree bermudan {berm_tree} (tol {tol})",
        lsm.price
    );
}

#[test]
fn lsm_barrier_applies_true_ex_dividend_drops_on_path() {
    // Mirrors tests/dividend_modelling_issue58.rs barrier_mc test: with
    // r = q = 0 and ~zero vol, the path is deterministic at 100 until the
    // 10.0 cash dividend at t = 0.5 drops it to 90, knocking in the
    // down-and-in 95 put, which then pays K - S_T = 10 at expiry. The old
    // effective-yield smear drifted the path smoothly and never produced
    // the discrete drop.
    let barrier = BarrierOption::builder()
        .put()
        .strike(100.0)
        .expiry(1.0)
        .down_and_in(95.0)
        .rebate(0.0)
        .build()
        .expect("valid barrier");

    let market = Market::builder()
        .spot(100.0)
        .rate(0.0)
        .dividend_yield(0.0)
        .dividend_schedule(
            DividendSchedule::new(vec![DividendEvent::cash(0.5, 10.0).expect("valid event")])
                .expect("valid schedule"),
        )
        .flat_vol(1.0e-8)
        .build()
        .expect("valid market");

    let engine = LongstaffSchwartzEngine::new(4_000, 252, 42);
    let price = engine.price(&barrier, &market).unwrap().price;
    assert!(
        (price - 10.0).abs() <= 1.0e-3,
        "expected ~10 from the knocked-in put after the ex-div drop, got {price}"
    );
}

#[test]
fn continuous_yield_only_behavior_is_preserved() {
    // With an empty discrete schedule the escrowed transform is the exact
    // identity: engines must agree with the analytic European and each other
    // exactly as before the escrowed-model change.
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.02)
        .flat_vol(0.25)
        .build()
        .expect("valid market");

    assert_eq!(market.escrowed_spot(1.0), 100.0);
    assert_eq!(market.escrowed_reconstruction(0.5, 1.0), (1.0, 0.0));

    let eu_put = VanillaOption::european_put(100.0, 1.0);
    let analytic = BlackScholesEngine.price(&eu_put, &market).unwrap().price;
    let tree = BinomialTreeEngine::new(2000)
        .price(&eu_put, &market)
        .unwrap()
        .price;
    assert!(
        (tree - analytic).abs() <= 0.01,
        "continuous-yield European drifted: tree={tree}, analytic={analytic}"
    );

    let am_put = VanillaOption::american_put(100.0, 1.0);
    let bin = BinomialTreeEngine::new(2000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    let ambin = AmericanBinomialEngine::new(2000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    let cn = CrankNicolsonEngine::new(400, 400)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert!(
        (bin - ambin).abs() <= 1.0e-12,
        "binomial engines diverged without dividends: {bin} vs {ambin}"
    );
    assert!(
        (bin - cn).abs() <= 0.05,
        "binomial vs CN diverged without dividends: {bin} vs {cn}"
    );
}
