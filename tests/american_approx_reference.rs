//! American option approximation reference tests.
//!
//! Sources:
//! - Barone-Adesi & Whaley (1987), "Efficient Analytic Approximation of
//!   American Option Values", J. Finance 42(2), 301-320
//! - Ju (1999), "An Approximate Formula for Pricing American Options",
//!   J. Derivatives 7(2), 31-40
//! - Haug "Option Pricing Formulas" (1998), pp. 24, 27
//! - QuantLib 1.43 `BinomialVanillaEngine` (CRR, 10,000 steps)
//!
//! The BAW and Ju tables supply the published parameter grids.  Prices below
//! are independent QuantLib CRR values, not the rounded approximation outputs.
//! The 5,000-to-10,000-step QuantLib refinement and two successive OpenFerric
//! Richardson estimates (500/1,000 and 1,000/2,000 steps) form an explicit
//! discretization budget.
//!
//! To avoid rounding decimal maturities to whole calendar days, the QuantLib
//! generation used an exact one-year horizon with `r' = r*T`, `q' = q*T`, and
//! `sigma' = sigma*sqrt(T)`.  This deterministic time change preserves every
//! discount, drift, and variance integral of the constant-parameter problem.

use openferric::core::{ExerciseStyle, OptionType, PricingEngine};
use openferric::engines::numerical::american_binomial::AmericanBinomialEngine;
use openferric::instruments::vanilla::VanillaOption;
use openferric::market::Market;

fn make_market(spot: f64, rate: f64, dividend_yield: f64, vol: f64) -> Market {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend_yield)
        .flat_vol(vol)
        .build()
        .expect("market build failed")
}

fn american_price(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    t: f64,
    steps: usize,
) -> f64 {
    let option = VanillaOption {
        option_type,
        strike,
        expiry: t,
        exercise: ExerciseStyle::American,
    };
    let market = make_market(spot, rate, q, vol);
    let engine = AmericanBinomialEngine::new(steps);
    engine
        .price(&option, &market)
        .expect("pricing failed")
        .price
}

// =======================================================================
// Barone-Adesi-Whaley call options
// K=100, q=0.10, r=0.10
// Parameter grid from Haug p.24; prices from QuantLib 1.43 CRR (10,000 steps).
// =======================================================================

struct BawCase {
    option_type: OptionType,
    spot: f64,
    t: f64,
    vol: f64,
    expected: f64,
}

const BAW_CALLS: &[BawCase] = &[
    BawCase {
        option_type: OptionType::Call,
        spot: 90.0,
        t: 0.10,
        vol: 0.15,
        expected: 0.020490196149781,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 100.0,
        t: 0.10,
        vol: 0.15,
        expected: 1.876_408_969_251_2,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 110.0,
        t: 0.10,
        vol: 0.15,
        expected: 10.010527099039248,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 90.0,
        t: 0.10,
        vol: 0.25,
        expected: 0.315268554461080,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 100.0,
        t: 0.10,
        vol: 0.25,
        expected: 3.126828546869588,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 110.0,
        t: 0.10,
        vol: 0.25,
        expected: 10.396_674_823_888_77,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 90.0,
        t: 0.10,
        vol: 0.35,
        expected: 0.948280600665965,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 100.0,
        t: 0.10,
        vol: 0.35,
        expected: 4.376468924988666,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 110.0,
        t: 0.10,
        vol: 0.35,
        expected: 11.172544167496044,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 90.0,
        t: 0.50,
        vol: 0.15,
        expected: 0.811435491828549,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 100.0,
        t: 0.50,
        vol: 0.15,
        expected: 4.068415215099752,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 110.0,
        t: 0.50,
        vol: 0.15,
        expected: 10.810757220025916,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 90.0,
        t: 0.50,
        vol: 0.25,
        expected: 2.722484518052434,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 100.0,
        t: 0.50,
        vol: 0.25,
        expected: 6.775138079029924,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 110.0,
        t: 0.50,
        vol: 0.25,
        expected: 13.001371056189356,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 90.0,
        t: 0.50,
        vol: 0.35,
        expected: 4.973946980464394,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 100.0,
        t: 0.50,
        vol: 0.35,
        expected: 9.473_551_738_344_23,
    },
    BawCase {
        option_type: OptionType::Call,
        spot: 110.0,
        t: 0.50,
        vol: 0.35,
        expected: 15.539214569330785,
    },
];

const BAW_PUTS: &[BawCase] = &[
    BawCase {
        option_type: OptionType::Put,
        spot: 90.0,
        t: 0.10,
        vol: 0.15,
        expected: 10.000570244788138,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 100.0,
        t: 0.10,
        vol: 0.15,
        expected: 1.876408971104582,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 110.0,
        t: 0.10,
        vol: 0.15,
        expected: 0.040795373281465,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 90.0,
        t: 0.10,
        vol: 0.25,
        expected: 10.259986206651908,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 100.0,
        t: 0.10,
        vol: 0.25,
        expected: 3.126828561169259,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 110.0,
        t: 0.10,
        vol: 0.25,
        expected: 0.455394245631125,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 90.0,
        t: 0.10,
        vol: 0.35,
        expected: 10.883826854459237,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 100.0,
        t: 0.10,
        vol: 0.35,
        expected: 4.376468979915124,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 110.0,
        t: 0.10,
        vol: 0.35,
        expected: 1.238701850575223,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 90.0,
        t: 0.50,
        vol: 0.15,
        expected: 10.564382494713838,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 100.0,
        t: 0.50,
        vol: 0.15,
        expected: 4.068415257124626,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 110.0,
        t: 0.50,
        vol: 0.15,
        expected: 1.070863872183615,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 90.0,
        t: 0.50,
        vol: 0.25,
        expected: 12.430107266194618,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 100.0,
        t: 0.50,
        vol: 0.25,
        expected: 6.775138403109347,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 110.0,
        t: 0.50,
        vol: 0.25,
        expected: 3.298316632975663,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 90.0,
        t: 0.50,
        vol: 0.35,
        expected: 14.669536036783038,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 100.0,
        t: 0.50,
        vol: 0.35,
        expected: 9.473552982251906,
    },
    BawCase {
        option_type: OptionType::Put,
        spot: 110.0,
        t: 0.50,
        vol: 0.35,
        expected: 5.845950165586387,
    },
];

// Absolute QuantLib CRR price changes from 5,000 to 10,000 steps, in the
// same order as the tables above.  These are measured numerical errors, not
// arbitrary price bands.
const BAW_CALL_QL_REFINEMENT: &[f64] = &[
    3.404401240234e-6,
    4.4377748765756e-5,
    1.16940095740148e-5,
    5.41383908628235e-5,
    7.39574338597393e-5,
    2.94151622899363e-5,
    2.65073069206645e-5,
    1.03535166328328e-4,
    1.78417597194169e-4,
    8.01070330856968e-5,
    8.56210265594015e-5,
    6.87130151373339e-5,
    1.66610549978241e-4,
    1.42768384680636e-4,
    4.47797553686513e-5,
    3.62025828026802e-4,
    2.00160319796083e-4,
    2.71783969045813e-4,
];

const BAW_PUT_QL_REFINEMENT: &[f64] = &[
    4.28588615264403e-7,
    4.43758960348983e-5,
    2.42603248750139e-5,
    4.36191108619255e-5,
    7.39431392080547e-5,
    2.7065158232864e-5,
    3.08661517074427e-5,
    1.03480259281419e-4,
    1.79952712419773e-4,
    5.25675275682147e-5,
    8.5579011624759e-5,
    7.47833855918589e-5,
    1.3351800473238e-4,
    1.42444379360285e-4,
    6.62175921797292e-5,
    3.20476147479098e-4,
    1.98916676124483e-4,
    2.79814447006643e-4,
];

#[allow(clippy::too_many_arguments)]
fn assert_crr_converges_to_quantlib(
    label: &str,
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    t: f64,
    reference: f64,
    ql_refinement: f64,
) {
    let coarse = american_price(option_type, spot, strike, rate, q, vol, t, 500);
    let mid = american_price(option_type, spot, strike, rate, q, vol, t, 1_000);
    let fine = american_price(option_type, spot, strike, rate, q, vol, t, 2_000);
    let coarse_extrapolated = 2.0 * mid - coarse;
    let fine_extrapolated = 2.0 * fine - mid;
    let implementation_refinement = (fine_extrapolated - coarse_extrapolated).abs();
    let error = (fine_extrapolated - reference).abs();

    // CRR convergence can oscillate when the strike moves between adjacent
    // lattice nodes.  Two successive Richardson differences cover that
    // observed oscillation across this full published grid while remaining
    // tied to measured refinements.
    let roundoff = 64.0 * f64::EPSILON * reference.abs().max(1.0);
    let tolerance = 2.0 * (implementation_refinement + ql_refinement) + roundoff;
    assert!(
        error <= tolerance,
        "{label}: Richardson={fine_extrapolated:.15}, QuantLib={reference:.15}, error={error:.3e}, \
         OpenFerric extrapolation refinement={implementation_refinement:.3e}, QuantLib 5k->10k={ql_refinement:.3e}, \
         tolerance={tolerance:.3e}"
    );
}

#[test]
fn baw_american_calls() {
    for (i, case) in BAW_CALLS.iter().enumerate() {
        assert_crr_converges_to_quantlib(
            &format!("BAW call #{i}"),
            case.option_type,
            case.spot,
            100.0,
            0.10,
            0.10,
            case.vol,
            case.t,
            case.expected,
            BAW_CALL_QL_REFINEMENT[i],
        );
    }
}

#[test]
fn baw_american_puts() {
    for (i, case) in BAW_PUTS.iter().enumerate() {
        assert_crr_converges_to_quantlib(
            &format!("BAW put #{i}"),
            case.option_type,
            case.spot,
            100.0,
            0.10,
            0.10,
            case.vol,
            case.t,
            case.expected,
            BAW_PUT_QL_REFINEMENT[i],
        );
    }
}

// =======================================================================
// Ju (1999) short-dated American puts
// S=40, K=35/40/45, q=0, r=0.0488
// Parameter grid from Haug/QuantLib americanoption.cpp; prices from QuantLib
// 1.43 CRR (10,000 steps).
// =======================================================================

struct JuCase {
    strike: f64,
    t: f64,
    vol: f64,
    expected: f64,
}

const JU_PUTS: &[JuCase] = &[
    JuCase {
        strike: 35.0,
        t: 0.0833,
        vol: 0.20,
        expected: 0.006190181855462,
    },
    JuCase {
        strike: 35.0,
        t: 0.3333,
        vol: 0.20,
        expected: 0.200360805629306,
    },
    JuCase {
        strike: 35.0,
        t: 0.5833,
        vol: 0.20,
        expected: 0.432816673395836,
    },
    JuCase {
        strike: 40.0,
        t: 0.0833,
        vol: 0.20,
        expected: 0.852156087689614,
    },
    JuCase {
        strike: 40.0,
        t: 0.3333,
        vol: 0.20,
        expected: 1.579794805659671,
    },
    JuCase {
        strike: 40.0,
        t: 0.5833,
        vol: 0.20,
        expected: 1.990437445570382,
    },
    JuCase {
        strike: 45.0,
        t: 0.0833,
        vol: 0.20,
        expected: 5.000000000000000,
    },
    JuCase {
        strike: 45.0,
        t: 0.3333,
        vol: 0.20,
        expected: 5.088313167338947,
    },
    JuCase {
        strike: 45.0,
        t: 0.5833,
        vol: 0.20,
        expected: 5.267_008_315_569_01,
    },
    JuCase {
        strike: 35.0,
        t: 0.0833,
        vol: 0.30,
        expected: 0.077378360885235,
    },
    JuCase {
        strike: 35.0,
        t: 0.3333,
        vol: 0.30,
        expected: 0.697536078889310,
    },
    JuCase {
        strike: 35.0,
        t: 0.5833,
        vol: 0.30,
        expected: 1.219814180963421,
    },
    JuCase {
        strike: 40.0,
        t: 0.0833,
        vol: 0.30,
        expected: 1.309904990382515,
    },
    JuCase {
        strike: 40.0,
        t: 0.3333,
        vol: 0.30,
        expected: 2.482526071964453,
    },
    JuCase {
        strike: 40.0,
        t: 0.5833,
        vol: 0.30,
        expected: 3.169604931641105,
    },
    JuCase {
        strike: 45.0,
        t: 0.0833,
        vol: 0.30,
        expected: 5.059667979920756,
    },
    JuCase {
        strike: 45.0,
        t: 0.3333,
        vol: 0.30,
        expected: 5.705583360901169,
    },
    JuCase {
        strike: 45.0,
        t: 0.5833,
        vol: 0.30,
        expected: 6.243650247301788,
    },
    JuCase {
        strike: 35.0,
        t: 0.0833,
        vol: 0.40,
        expected: 0.246551919957140,
    },
    JuCase {
        strike: 35.0,
        t: 0.3333,
        vol: 0.40,
        expected: 1.346015449445347,
    },
    JuCase {
        strike: 35.0,
        t: 0.5833,
        vol: 0.40,
        expected: 2.154896467014023,
    },
    JuCase {
        strike: 40.0,
        t: 0.0833,
        vol: 0.40,
        expected: 1.768100630548247,
    },
    JuCase {
        strike: 40.0,
        t: 0.3333,
        vol: 0.40,
        expected: 3.387411366127216,
    },
    JuCase {
        strike: 40.0,
        t: 0.5833,
        vol: 0.40,
        expected: 4.352639106281274,
    },
    JuCase {
        strike: 45.0,
        t: 0.0833,
        vol: 0.40,
        expected: 5.286782782000495,
    },
    JuCase {
        strike: 45.0,
        t: 0.3333,
        vol: 0.40,
        expected: 6.509882690185533,
    },
    JuCase {
        strike: 45.0,
        t: 0.5833,
        vol: 0.40,
        expected: 7.383_046_955_040_51,
    },
];

const JU_PUT_QL_REFINEMENT: &[f64] = &[
    2.0747882014437e-6,
    1.2003481506706e-5,
    3.22231254430982e-5,
    1.47491009550027e-5,
    2.22737957591423e-5,
    2.56208800422097e-5,
    0.0,
    1.67554546504078e-6,
    5.06106368547421e-6,
    1.36076285044417e-5,
    6.17734933983094e-7,
    1.13905054411223e-4,
    2.48342642419352e-5,
    3.95481619079874e-5,
    4.61324185061507e-5,
    1.25259102645714e-7,
    5.20335755282986e-6,
    5.30862269698673e-7,
    1.88333785492989e-5,
    1.4638203112205e-4,
    1.75065464449009e-4,
    3.52933706728731e-5,
    5.80915873928944e-5,
    6.90976669179832e-5,
    1.92900181961164e-6,
    1.52222377589162e-4,
    1.19554100194819e-4,
];

#[test]
fn ju_american_puts() {
    for (i, case) in JU_PUTS.iter().enumerate() {
        assert_crr_converges_to_quantlib(
            &format!("Ju put #{i}"),
            OptionType::Put,
            40.0,
            case.strike,
            0.0488,
            0.0,
            case.vol,
            case.t,
            case.expected,
            JU_PUT_QL_REFINEMENT[i],
        );
    }
}

// =======================================================================
// American put: QuantLib price and exact European lower bound
// =======================================================================
#[test]
fn american_put_matches_quantlib_and_exceeds_european() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let q = 0.03;
    let vol = 0.25;
    let t = 1.0;

    // QuantLib 1.43 CRR, 10,000 steps; 5k->10k change = 1.42440273192e-4.
    assert_crr_converges_to_quantlib(
        "American put",
        OptionType::Put,
        spot,
        strike,
        rate,
        q,
        vol,
        t,
        8.882559408795435,
        1.42440273192e-4,
    );

    let am_price = american_price(OptionType::Put, spot, strike, rate, q, vol, t, 2_000);
    let eu_price = openferric::engines::analytic::black_scholes::bs_price(
        OptionType::Put,
        spot,
        strike,
        rate,
        q,
        vol,
        t,
    );
    assert!(
        am_price >= eu_price,
        "American put ({am_price}) must >= European put ({eu_price})"
    );
}

#[test]
fn american_call_no_dividend_equals_european() {
    // Merton (1973): American call on non-dividend stock = European call
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let q = 0.0; // no dividend => no early exercise premium for calls
    let vol = 0.25;
    let t = 1.0;

    let coarse = american_price(OptionType::Call, spot, strike, rate, q, vol, t, 500);
    let mid = american_price(OptionType::Call, spot, strike, rate, q, vol, t, 1_000);
    let fine = american_price(OptionType::Call, spot, strike, rate, q, vol, t, 2_000);
    let eu_price = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        spot,
        strike,
        rate,
        vol,
        t,
    );

    let coarse_extrapolated = 2.0 * mid - coarse;
    let fine_extrapolated = 2.0 * fine - mid;
    let err = (fine_extrapolated - eu_price).abs();
    let measured_error = (fine_extrapolated - coarse_extrapolated).abs();
    let roundoff = 64.0 * f64::EPSILON * eu_price.abs().max(1.0);
    assert!(
        err <= measured_error + roundoff,
        "American no-dividend call Richardson={fine_extrapolated:.15}, exact BSM={eu_price:.15}, \
         error={err:.3e}, measured refinement={measured_error:.3e}"
    );
}

// =======================================================================
// Deep ITM American put: immediate exercise is exactly optimal
// =======================================================================
#[test]
fn deep_itm_american_put_equals_intrinsic() {
    let price = american_price(OptionType::Put, 50.0, 100.0, 0.05, 0.0, 0.20, 0.25, 500);
    let intrinsic = 50.0;
    assert_eq!(
        price, 49.999999999999346,
        "500-step CRR finite-grid value changed"
    );
    assert!(
        (price - intrinsic).abs() <= 64.0 * f64::EPSILON * intrinsic,
        "immediate-exercise price {price:.15} differs from intrinsic {intrinsic:.15} beyond roundoff"
    );
}
