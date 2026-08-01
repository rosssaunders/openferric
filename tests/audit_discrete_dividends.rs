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
/// Hand-checkable: S* = 100 - 5 e^{-0.03*0.5} = 95.07444030198468, then
/// Black-Scholes(S*, K=100, r=3%, q=0, vol=25%, T=1). These full-precision
/// values were independently evaluated with SciPy 1.17.1's normal CDF.
const PROBE_ESCROWED_EUROPEAN_PUT: f64 = 10.572_685_288_309_124;
const PROBE_ESCROWED_EUROPEAN_CALL: f64 = 8.602_572_235_442_985;

/// Independent NumPy CRR finite-grid references for the probe scenario.
///
/// The reference implementation diffuses S* under GBM and reconstructs the
/// observed spot before each exercise decision. Full finite-grid values avoid
/// confusing discretization error with a rounded economic-price tolerance.
const PROBE_CRR_4000_AMERICAN_PUT: f64 = 10.925_211_258_957_25;
const PROBE_CRR_8000_AMERICAN_PUT: f64 = 10.924_567_770_136_91;
const PROBE_CRR_4000_AMERICAN_CALL: f64 = 9.029_090_473_504_283;
const PROBE_CRR_4000_BERMUDAN_PUT: f64 = 10.842_664_903_548_698;
const PROBE_CRR_8000_BERMUDAN_PUT: f64 = 10.842_010_214_812_372;

/// Compare independent closed forms allowing accumulated binary64 roundoff.
fn assert_binary64_close(label: &str, actual: f64, expected: f64) {
    let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label}: expected={expected:.17e}, actual={actual:.17e}, \
         error={error:.3e}, binary64 budget={tolerance:.3e}"
    );
}

/// Exact deterministic finite-grid regression. The 2e-10 budget matches the
/// cross-platform operation-order budget in the primary QuantLib escrowed-
/// dividend suite (`dividend_modelling_issue58`); it is not a price cushion.
fn assert_grid_lock(label: &str, actual: f64, expected: f64) {
    let tolerance = 2.0e-10;
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label}: expected finite-grid value={expected:.17e}, actual={actual:.17e}, \
         error={error:.3e}, numerical budget={tolerance:.3e}"
    );
}

fn assert_mc_close(
    label: &str,
    price: f64,
    stderr: Option<f64>,
    reference: f64,
    reference_grid_uncertainty: f64,
) {
    let stderr = stderr.expect("Monte Carlo result must report standard error");
    let roundoff = 64.0 * f64::EPSILON * reference.abs().max(1.0);
    let tolerance = 4.0 * stderr + reference_grid_uncertainty + roundoff;
    let error = (price - reference).abs();
    assert!(
        error <= tolerance,
        "{label}: reference={reference:.17e}, price={price:.17e}, error={error:.3e}, \
         4SE={:.3e}, reference-grid uncertainty={reference_grid_uncertainty:.3e}, \
         roundoff={roundoff:.3e}, total budget={tolerance:.3e}",
        4.0 * stderr
    );
}

#[test]
fn european_engines_match_escrowed_analytic_with_cash_dividend() {
    let market = probe_market();
    let put = VanillaOption::european_put(100.0, 1.0);
    let call = VanillaOption::european_call(100.0, 1.0);

    let analytic_put = BlackScholesEngine.price(&put, &market).unwrap().price;
    let analytic_call = BlackScholesEngine.price(&call, &market).unwrap().price;
    assert_binary64_close(
        "analytic escrowed European put / SciPy",
        analytic_put,
        PROBE_ESCROWED_EUROPEAN_PUT,
    );
    assert_binary64_close(
        "analytic escrowed European call / SciPy",
        analytic_call,
        PROBE_ESCROWED_EUROPEAN_CALL,
    );

    // Supplemental exact finite-grid locks for the numerical engines. The
    // independent SciPy assertions above are the economic-price oracle; these
    // values pin each stated discretization without a broad price band.
    let checks: Vec<(&str, f64, f64)> = vec![
        (
            "binomial put",
            BinomialTreeEngine::new(2000)
                .price(&put, &market)
                .unwrap()
                .price,
            10.573_808_204_976_473,
        ),
        (
            "binomial call",
            BinomialTreeEngine::new(2000)
                .price(&call, &market)
                .unwrap()
                .price,
            8.603_695_152_104_963,
        ),
        (
            "trinomial put",
            TrinomialTreeEngine::new(1000)
                .price(&put, &market)
                .unwrap()
                .price,
            10.573_808_204_969_186,
        ),
        (
            "crank-nicolson put",
            CrankNicolsonEngine::new(800, 800)
                .price(&put, &market)
                .unwrap()
                .price,
            10.572_467_527_253_25,
        ),
        (
            "implicit fd put",
            ImplicitFdEngine::default()
                .price(&put, &market)
                .unwrap()
                .price,
            10.569_802_134_799_64,
        ),
        (
            "explicit fd put",
            ExplicitFdEngine::default()
                .price(&put, &market)
                .unwrap()
                .price,
            10.573_879_596_785_945,
        ),
        (
            "hopscotch put",
            HopscotchEngine::default()
                .price(&put, &market)
                .unwrap()
                .price,
            10.573_954_064_595_572,
        ),
    ];
    for (label, price, finite_grid_reference) in checks {
        assert_grid_lock(label, price, finite_grid_reference);
    }

    // LSM European (exercise only at expiry) has only its reported sampling
    // error and binary64 roundoff in the comparison budget.
    let lsm = LongstaffSchwartzEngine::new(60_000, 50, 42)
        .price(&put, &market)
        .unwrap();
    assert_mc_close(
        "LSM European put / exact escrowed BSM",
        lsm.price,
        lsm.stderr,
        PROBE_ESCROWED_EUROPEAN_PUT,
        0.0,
    );
}

#[test]
fn hull_dividend_european_call_reference() {
    // External anchor: Hull, "Options, Futures, and Other Derivatives"
    // (5th ed.), Exercise 12.8 — also cached as 3.67 in QuantLib's
    // test-suite/dividendoption.cpp (testEuropeanKnownValue, Actual/360):
    // S=40, K=40, r=9%, vol=30%, T=180/360, cash dividends 0.50 at 60/360
    // and 0.50 at 150/360. Hull publishes 3.67 after rounding; the primary
    // assertion below uses the full-precision escrowed BSM value independently
    // evaluated with SciPy 1.17.1.
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
    assert_binary64_close(
        "Hull escrowed European call / SciPy",
        analytic,
        3.671_233_209_047_678,
    );
    // Supplemental deterministic locks for the stated numerical grids.
    assert_grid_lock("Hull binomial(2000)", tree, 3.671_577_671_984_805_2);
    assert_grid_lock("Hull CN(800x800)", pde, 3.671_105_883_184_173_6);
}

#[test]
fn american_put_cash_dividend_has_material_early_exercise_premium() {
    let market = probe_market();
    let am_put = VanillaOption::american_put(100.0, 1.0);
    let eu_put = VanillaOption::european_put(100.0, 1.0);

    let analytic_eu = BlackScholesEngine.price(&eu_put, &market).unwrap().price;

    // Requirement (2): a materially positive early-exercise premium. The
    // pre-fix engines produced ~0.001 here (q_eff > r suppressed exercise).
    let binomial = BinomialTreeEngine::new(4000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert_grid_lock(
        "American put / independent NumPy CRR(4000)",
        binomial,
        PROBE_CRR_4000_AMERICAN_PUT,
    );
    let premium = binomial - analytic_eu;
    // Supplemental bug-signature check: the pre-fix smeared model produced a
    // premium of only ~0.001. Exact pricing is enforced by the CRR lock above.
    assert!(
        premium >= 0.30,
        "early-exercise premium {premium} should be ~0.35, got binomial={binomial}, european={analytic_eu}"
    );

    // American call premium must also come from the discrete-dividend
    // mechanism: calls on cash-dividend stocks exercise just before ex-dates.
    let am_call = VanillaOption::american_call(100.0, 1.0);
    let eu_call = VanillaOption::european_call(100.0, 1.0);
    let am_call_px = BinomialTreeEngine::new(4000)
        .price(&am_call, &market)
        .unwrap()
        .price;
    let eu_call_px = BlackScholesEngine.price(&eu_call, &market).unwrap().price;
    assert_grid_lock(
        "American call / independent NumPy CRR(4000)",
        am_call_px,
        PROBE_CRR_4000_AMERICAN_CALL,
    );
    // Supplemental exercise-premium invariant.
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
    assert_grid_lock(
        "binomial / independent NumPy CRR(4000)",
        reference,
        PROBE_CRR_4000_AMERICAN_PUT,
    );

    let american_binomial = AmericanBinomialEngine::new(4000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert_grid_lock(
        "AmericanBinomial / independent NumPy CRR(4000)",
        american_binomial,
        PROBE_CRR_4000_AMERICAN_PUT,
    );

    // Supplemental exact locks for each engine's stated finite grid. The
    // independent NumPy CRR assertion above is the primary model reference.
    let checks: Vec<(&str, f64, f64)> = vec![
        (
            "trinomial",
            TrinomialTreeEngine::new(1500)
                .price(&am_put, &market)
                .unwrap()
                .price,
            10.925_290_691_903_852,
        ),
        (
            "crank-nicolson",
            CrankNicolsonEngine::new(800, 800)
                .price(&am_put, &market)
                .unwrap()
                .price,
            10.923_887_968_704_91,
        ),
        (
            "implicit fd",
            ImplicitFdEngine::default()
                .price(&am_put, &market)
                .unwrap()
                .price,
            10.916_540_866_201_066,
        ),
        (
            "explicit fd",
            ExplicitFdEngine::default()
                .price(&am_put, &market)
                .unwrap()
                .price,
            10.925_154_766_982_441,
        ),
        (
            "hopscotch",
            HopscotchEngine::default()
                .price(&am_put, &market)
                .unwrap()
                .price,
            10.929_113_074_114_246,
        ),
    ];
    for (label, price, finite_grid_reference) in checks {
        assert_grid_lock(label, price, finite_grid_reference);
    }

    // LSM is statistical. Its budget is four reported standard errors plus
    // the independently measured NumPy CRR 4k->8k grid change; there is no
    // fixed economic-price cushion.
    let lsm = LongstaffSchwartzEngine::new(60_000, 100, 42)
        .price(&am_put, &market)
        .unwrap();
    assert_mc_close(
        "LSM American put / independent NumPy CRR",
        lsm.price,
        lsm.stderr,
        PROBE_CRR_4000_AMERICAN_PUT,
        (PROBE_CRR_4000_AMERICAN_PUT - PROBE_CRR_8000_AMERICAN_PUT).abs(),
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
    assert_binary64_close(
        "proportional-dividend European put / SciPy",
        analytic_eu,
        10.608_676_509_801_22,
    );
    let tree_eu = BinomialTreeEngine::new(2000)
        .price(&eu_put, &market)
        .unwrap()
        .price;
    assert_grid_lock(
        "proportional-dividend European put / NumPy CRR(2000)",
        tree_eu,
        10.609_728_274_611_916,
    );

    // American primary reference: independent NumPy CRR finite grid.
    let tree_am = BinomialTreeEngine::new(4000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    assert_grid_lock(
        "proportional-dividend American put / NumPy CRR(4000)",
        tree_am,
        10.962_788_815_441_384,
    );
    let tri_am = TrinomialTreeEngine::new(1000)
        .price(&am_put, &market)
        .unwrap()
        .price;
    let cn_am = CrankNicolsonEngine::new(800, 800)
        .price(&am_put, &market)
        .unwrap()
        .price;
    // Supplemental deterministic locks for the other finite grids.
    assert_grid_lock(
        "proportional-dividend trinomial(1000)",
        tri_am,
        10.963_087_560_257_108,
    );
    assert_grid_lock(
        "proportional-dividend CN(800x800)",
        cn_am,
        10.961_148_649_739_456,
    );
    let premium = tree_am - analytic_eu;
    // Supplemental rejection of the historical smeared-dividend behavior.
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

    let tree_engine = BinomialTreeEngine::new(4000);
    let eu = BlackScholesEngine.price(&european, &market).unwrap().price;
    let berm_tree = tree_engine.price(&bermudan_vanilla, &market).unwrap().price;
    let am = tree_engine.price(&american, &market).unwrap().price;
    assert_grid_lock(
        "Bermudan put / independent NumPy CRR(4000)",
        berm_tree,
        PROBE_CRR_4000_BERMUDAN_PUT,
    );
    assert_grid_lock(
        "American bound / independent NumPy CRR(4000)",
        am,
        PROBE_CRR_4000_AMERICAN_PUT,
    );

    // Supplemental ordering and historical-premium invariants.
    assert!(
        eu <= berm_tree && berm_tree <= am,
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
    assert_grid_lock("Bermudan CN(800x800)", cn, 10.841_974_677_707_809);

    let lsm = LongstaffSchwartzEngine::new(60_000, 100, 42)
        .price(&bermudan, &market)
        .unwrap();
    assert_mc_close(
        "LSM Bermudan put / independent NumPy CRR",
        lsm.price,
        lsm.stderr,
        PROBE_CRR_4000_BERMUDAN_PUT,
        (PROBE_CRR_4000_BERMUDAN_PUT - PROBE_CRR_8000_BERMUDAN_PUT).abs(),
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
    let result = engine.price(&barrier, &market).unwrap();
    assert_mc_close(
        "near-deterministic knocked-in put",
        result.price,
        result.stderr,
        10.0,
        0.0,
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
    assert_binary64_close(
        "continuous-yield European put / SciPy",
        analytic,
        9.222_221_299_637_46,
    );
    let tree = BinomialTreeEngine::new(2000)
        .price(&eu_put, &market)
        .unwrap()
        .price;
    assert_grid_lock(
        "continuous-yield European NumPy CRR(2000)",
        tree,
        9.221_007_681_647_961,
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
    assert_grid_lock(
        "continuous-yield American NumPy CRR(2000)",
        bin,
        9.345_697_025_914_843,
    );
    assert_grid_lock(
        "continuous-yield AmericanBinomial CRR(2000)",
        ambin,
        9.345_697_025_914_843,
    );
    // Supplemental exact finite-grid lock for the CN implementation.
    assert_grid_lock(
        "continuous-yield American CN(400x400)",
        cn,
        9.343_972_384_255_485,
    );
}
