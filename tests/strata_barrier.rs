// Reference values from OpenGamma Strata (Apache 2.0), https://github.com/OpenGamma/Strata
//
// Source: BlackBarrierPriceFormulaRepositoryTest.java (adjointPriceRegression)
//   https://github.com/OpenGamma/Strata/blob/main/modules/pricer/src/test/java/
//     com/opengamma/strata/pricer/impl/option/BlackBarrierPriceFormulaRepositoryTest.java
//
// Market data:
//   SPOT = 105, RATE_DOM = 0.05, RATE_FOR = 0.02 (=> cost_of_carry b = 0.03)
//   VOLATILITY = 0.20, REBATE = 2
//   TIME_TO_EXPIRY ~ 3.5068y (ACT/ACT ISDA from 2011-07-01 to 2015-01-02)
//   Down barriers at 90, Up barriers at 110
//   Strikes: 120 (HIGH), 100 (MID), 85 (LOW)

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::{BarrierAnalyticEngine, BlackScholesEngine};
use openferric::instruments::{BarrierOption, VanillaOption};
use openferric::market::Market;

// ── Strata market parameters ──────────────────────────────────────────────

const SPOT: f64 = 105.0;
const RATE_DOM: f64 = 0.05;
const RATE_FOR: f64 = 0.02; // dividend_yield = RATE_FOR, so cost_of_carry b = r - q = 0.03
const VOL: f64 = 0.20;
const REBATE: f64 = 2.0;

// ACT/ACT ISDA from 2011-07-01 to 2015-01-02
const T: f64 = 3.5068493150684934;

const BARRIER_DOWN: f64 = 90.0;
const BARRIER_UP: f64 = 110.0;

const STRIKE_HIGH: f64 = 120.0;
const STRIKE_MID: f64 = 100.0;
const STRIKE_LOW: f64 = 85.0;

// ── Strata reference prices (exact values from adjointPriceRegression) ────
//
// Array index mapping (matches Strata options array):
//   [0] = Call K=100, [1] = Put K=100,
//   [2] = Call K=120, [3] = Put K=120,
//   [4] = Call K=85,  [5] = Put K=85

const PRICE_DI: [f64; 6] = [
    6.625939880275156,
    8.17524397035564,
    3.51889794875554,
    16.046696834562567,
    10.70082805329517,
    4.016261046580751,
];

const PRICE_DO: [f64; 6] = [
    16.801234633074746,
    1.2809481492685348,
    11.695029389570358,
    1.9796398042263066,
    21.122005303422565,
    1.2480461457697478,
];

const PRICE_UI: [f64; 6] = [
    21.738904060619003,
    5.660922675994705,
    13.534230659666587,
    12.751249399664466,
    30.003917380997216,
    2.454685902906281,
];

const PRICE_UO: [f64; 6] = [
    1.8022701280119453,
    3.909269118910516,
    1.7936963539403596,
    5.389086914405454,
    1.9329156510015661,
    2.9236209647252656,
];

// ── Tolerance ─────────────────────────────────────────────────────────────
// Both implementations use the same Reiner-Rubinstein closed-form, but
// the normal CDF implementations differ slightly between Java (Strata)
// and Rust (openferric), producing differences up to ~2e-5.
// We use 5e-5 as a tight but robust tolerance.
const TOL: f64 = 5e-5;

// Slightly looser tolerance for in/out parity (accumulates two pricing errors).
const PARITY_TOL: f64 = 1e-8;

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_market() -> Market {
    Market::builder()
        .spot(SPOT)
        .rate(RATE_DOM)
        .dividend_yield(RATE_FOR)
        .flat_vol(VOL)
        .build()
        .expect("valid Strata market")
}

fn make_barrier_option(
    option_type: OptionType,
    strike: f64,
    barrier_level: f64,
    direction: &str,
    style: &str,
) -> BarrierOption {
    let mut builder = BarrierOption::builder()
        .strike(strike)
        .expiry(T)
        .rebate(REBATE);

    builder = match option_type {
        OptionType::Call => builder.call(),
        OptionType::Put => builder.put(),
    };

    builder = match (direction, style) {
        ("down", "in") => builder.down_and_in(barrier_level),
        ("down", "out") => builder.down_and_out(barrier_level),
        ("up", "in") => builder.up_and_in(barrier_level),
        ("up", "out") => builder.up_and_out(barrier_level),
        _ => panic!("invalid direction/style: {direction}/{style}"),
    };

    builder.build().expect("valid barrier option")
}

fn price_barrier(option: &BarrierOption) -> f64 {
    let market = make_market();
    let engine = BarrierAnalyticEngine::new();
    engine
        .price(option, &market)
        .expect("barrier pricing succeeds")
        .price
}

// ── Individual price tests: Down-In ───────────────────────────────────────

#[test]
fn strata_down_in_call_k100() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_MID, BARRIER_DOWN, "down", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DI[0]).abs() < TOL,
        "DI Call K=100: expected {} got {} diff {}",
        PRICE_DI[0], px, (px - PRICE_DI[0]).abs()
    );
}

#[test]
fn strata_down_in_put_k100() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_MID, BARRIER_DOWN, "down", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DI[1]).abs() < TOL,
        "DI Put K=100: expected {} got {} diff {}",
        PRICE_DI[1], px, (px - PRICE_DI[1]).abs()
    );
}

#[test]
fn strata_down_in_call_k120() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_HIGH, BARRIER_DOWN, "down", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DI[2]).abs() < TOL,
        "DI Call K=120: expected {} got {} diff {}",
        PRICE_DI[2], px, (px - PRICE_DI[2]).abs()
    );
}

#[test]
fn strata_down_in_put_k120() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_HIGH, BARRIER_DOWN, "down", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DI[3]).abs() < TOL,
        "DI Put K=120: expected {} got {} diff {}",
        PRICE_DI[3], px, (px - PRICE_DI[3]).abs()
    );
}

#[test]
fn strata_down_in_call_k85() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_LOW, BARRIER_DOWN, "down", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DI[4]).abs() < TOL,
        "DI Call K=85: expected {} got {} diff {}",
        PRICE_DI[4], px, (px - PRICE_DI[4]).abs()
    );
}

#[test]
fn strata_down_in_put_k85() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_LOW, BARRIER_DOWN, "down", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DI[5]).abs() < TOL,
        "DI Put K=85: expected {} got {} diff {}",
        PRICE_DI[5], px, (px - PRICE_DI[5]).abs()
    );
}

// ── Individual price tests: Down-Out ──────────────────────────────────────

#[test]
fn strata_down_out_call_k100() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_MID, BARRIER_DOWN, "down", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DO[0]).abs() < TOL,
        "DO Call K=100: expected {} got {} diff {}",
        PRICE_DO[0], px, (px - PRICE_DO[0]).abs()
    );
}

#[test]
fn strata_down_out_put_k100() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_MID, BARRIER_DOWN, "down", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DO[1]).abs() < TOL,
        "DO Put K=100: expected {} got {} diff {}",
        PRICE_DO[1], px, (px - PRICE_DO[1]).abs()
    );
}

#[test]
fn strata_down_out_call_k120() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_HIGH, BARRIER_DOWN, "down", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DO[2]).abs() < TOL,
        "DO Call K=120: expected {} got {} diff {}",
        PRICE_DO[2], px, (px - PRICE_DO[2]).abs()
    );
}

#[test]
fn strata_down_out_put_k120() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_HIGH, BARRIER_DOWN, "down", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DO[3]).abs() < TOL,
        "DO Put K=120: expected {} got {} diff {}",
        PRICE_DO[3], px, (px - PRICE_DO[3]).abs()
    );
}

#[test]
fn strata_down_out_call_k85() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_LOW, BARRIER_DOWN, "down", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DO[4]).abs() < TOL,
        "DO Call K=85: expected {} got {} diff {}",
        PRICE_DO[4], px, (px - PRICE_DO[4]).abs()
    );
}

#[test]
fn strata_down_out_put_k85() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_LOW, BARRIER_DOWN, "down", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_DO[5]).abs() < TOL,
        "DO Put K=85: expected {} got {} diff {}",
        PRICE_DO[5], px, (px - PRICE_DO[5]).abs()
    );
}

// ── Individual price tests: Up-In ─────────────────────────────────────────

#[test]
fn strata_up_in_call_k100() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_MID, BARRIER_UP, "up", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UI[0]).abs() < TOL,
        "UI Call K=100: expected {} got {} diff {}",
        PRICE_UI[0], px, (px - PRICE_UI[0]).abs()
    );
}

#[test]
fn strata_up_in_put_k100() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_MID, BARRIER_UP, "up", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UI[1]).abs() < TOL,
        "UI Put K=100: expected {} got {} diff {}",
        PRICE_UI[1], px, (px - PRICE_UI[1]).abs()
    );
}

#[test]
fn strata_up_in_call_k120() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_HIGH, BARRIER_UP, "up", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UI[2]).abs() < TOL,
        "UI Call K=120: expected {} got {} diff {}",
        PRICE_UI[2], px, (px - PRICE_UI[2]).abs()
    );
}

#[test]
fn strata_up_in_put_k120() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_HIGH, BARRIER_UP, "up", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UI[3]).abs() < TOL,
        "UI Put K=120: expected {} got {} diff {}",
        PRICE_UI[3], px, (px - PRICE_UI[3]).abs()
    );
}

#[test]
fn strata_up_in_call_k85() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_LOW, BARRIER_UP, "up", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UI[4]).abs() < TOL,
        "UI Call K=85: expected {} got {} diff {}",
        PRICE_UI[4], px, (px - PRICE_UI[4]).abs()
    );
}

#[test]
fn strata_up_in_put_k85() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_LOW, BARRIER_UP, "up", "in");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UI[5]).abs() < TOL,
        "UI Put K=85: expected {} got {} diff {}",
        PRICE_UI[5], px, (px - PRICE_UI[5]).abs()
    );
}

// ── Individual price tests: Up-Out ────────────────────────────────────────

#[test]
fn strata_up_out_call_k100() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_MID, BARRIER_UP, "up", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UO[0]).abs() < TOL,
        "UO Call K=100: expected {} got {} diff {}",
        PRICE_UO[0], px, (px - PRICE_UO[0]).abs()
    );
}

#[test]
fn strata_up_out_put_k100() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_MID, BARRIER_UP, "up", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UO[1]).abs() < TOL,
        "UO Put K=100: expected {} got {} diff {}",
        PRICE_UO[1], px, (px - PRICE_UO[1]).abs()
    );
}

#[test]
fn strata_up_out_call_k120() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_HIGH, BARRIER_UP, "up", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UO[2]).abs() < TOL,
        "UO Call K=120: expected {} got {} diff {}",
        PRICE_UO[2], px, (px - PRICE_UO[2]).abs()
    );
}

#[test]
fn strata_up_out_put_k120() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_HIGH, BARRIER_UP, "up", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UO[3]).abs() < TOL,
        "UO Put K=120: expected {} got {} diff {}",
        PRICE_UO[3], px, (px - PRICE_UO[3]).abs()
    );
}

#[test]
fn strata_up_out_call_k85() {
    let opt = make_barrier_option(OptionType::Call, STRIKE_LOW, BARRIER_UP, "up", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UO[4]).abs() < TOL,
        "UO Call K=85: expected {} got {} diff {}",
        PRICE_UO[4], px, (px - PRICE_UO[4]).abs()
    );
}

#[test]
fn strata_up_out_put_k85() {
    let opt = make_barrier_option(OptionType::Put, STRIKE_LOW, BARRIER_UP, "up", "out");
    let px = price_barrier(&opt);
    assert!(
        (px - PRICE_UO[5]).abs() < TOL,
        "UO Put K=85: expected {} got {} diff {}",
        PRICE_UO[5], px, (px - PRICE_UO[5]).abs()
    );
}

// ── In-Out Parity Tests ──────────────────────────────────────────────────
//
// For each (direction, option_type, strike) triple, the relation
//     knock_in + knock_out = vanilla + PV(rebate)
// must hold.  Since both knock-in and knock-out include the rebate
// component in the Reiner-Rubinstein formula (E and F terms), and the
// rebate terms from in + out together equal the PV of the rebate paid
// unconditionally, the standard relation with zero rebate simplifies to:
//     knock_in(rebate=R) + knock_out(rebate=R) = vanilla + PV_rebate
//
// However, when rebate = 0, in + out = vanilla exactly.  We test both
// directions x both option types at each strike with rebate = 0 for clean
// parity verification.

fn parity_check(option_type: OptionType, strike: f64, barrier_level: f64, direction: &str) {
    let market = make_market();
    let barrier_engine = BarrierAnalyticEngine::new();
    let vanilla_engine = BlackScholesEngine::new();

    let knock_in = {
        let mut builder = BarrierOption::builder()
            .strike(strike)
            .expiry(T)
            .rebate(0.0);
        builder = match option_type {
            OptionType::Call => builder.call(),
            OptionType::Put => builder.put(),
        };
        builder = match direction {
            "down" => builder.down_and_in(barrier_level),
            "up" => builder.up_and_in(barrier_level),
            _ => panic!("bad direction"),
        };
        builder.build().expect("valid knock-in")
    };

    let knock_out = {
        let mut builder = BarrierOption::builder()
            .strike(strike)
            .expiry(T)
            .rebate(0.0);
        builder = match option_type {
            OptionType::Call => builder.call(),
            OptionType::Put => builder.put(),
        };
        builder = match direction {
            "down" => builder.down_and_out(barrier_level),
            "up" => builder.up_and_out(barrier_level),
            _ => panic!("bad direction"),
        };
        builder.build().expect("valid knock-out")
    };

    let vanilla = match option_type {
        OptionType::Call => VanillaOption::european_call(strike, T),
        OptionType::Put => VanillaOption::european_put(strike, T),
    };

    let in_px = barrier_engine
        .price(&knock_in, &market)
        .expect("knock-in prices")
        .price;
    let out_px = barrier_engine
        .price(&knock_out, &market)
        .expect("knock-out prices")
        .price;
    let vanilla_px = vanilla_engine
        .price(&vanilla, &market)
        .expect("vanilla prices")
        .price;

    let diff = (in_px + out_px - vanilla_px).abs();
    assert!(
        diff < PARITY_TOL,
        "in-out parity failed for {} {:?} K={} barrier={}: in={} out={} vanilla={} diff={}",
        direction, option_type, strike, barrier_level, in_px, out_px, vanilla_px, diff
    );
}

// Down-barrier parity (barrier = 90)

#[test]
fn strata_parity_down_call_k100() {
    parity_check(OptionType::Call, STRIKE_MID, BARRIER_DOWN, "down");
}

#[test]
fn strata_parity_down_put_k100() {
    parity_check(OptionType::Put, STRIKE_MID, BARRIER_DOWN, "down");
}

#[test]
fn strata_parity_down_call_k120() {
    parity_check(OptionType::Call, STRIKE_HIGH, BARRIER_DOWN, "down");
}

#[test]
fn strata_parity_down_put_k120() {
    parity_check(OptionType::Put, STRIKE_HIGH, BARRIER_DOWN, "down");
}

#[test]
fn strata_parity_down_call_k85() {
    parity_check(OptionType::Call, STRIKE_LOW, BARRIER_DOWN, "down");
}

#[test]
fn strata_parity_down_put_k85() {
    parity_check(OptionType::Put, STRIKE_LOW, BARRIER_DOWN, "down");
}

// Up-barrier parity (barrier = 110)

#[test]
fn strata_parity_up_call_k100() {
    parity_check(OptionType::Call, STRIKE_MID, BARRIER_UP, "up");
}

#[test]
fn strata_parity_up_put_k100() {
    parity_check(OptionType::Put, STRIKE_MID, BARRIER_UP, "up");
}

#[test]
fn strata_parity_up_call_k120() {
    parity_check(OptionType::Call, STRIKE_HIGH, BARRIER_UP, "up");
}

#[test]
fn strata_parity_up_put_k120() {
    parity_check(OptionType::Put, STRIKE_HIGH, BARRIER_UP, "up");
}

#[test]
fn strata_parity_up_call_k85() {
    parity_check(OptionType::Call, STRIKE_LOW, BARRIER_UP, "up");
}

#[test]
fn strata_parity_up_put_k85() {
    parity_check(OptionType::Put, STRIKE_LOW, BARRIER_UP, "up");
}
