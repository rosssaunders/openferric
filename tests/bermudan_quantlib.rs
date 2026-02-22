//! Bermudan option reference and regression tests.
//!
//! QuantLib references were generated with QuantLib 1.41 using
//! `FdBlackScholesVanillaEngine(process, 2000, 400)` and
//! `BermudanExercise` schedules under ACT/365 year fractions.
//! Tolerance target from issue #56: 0.5%.

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::engines::numerical::AmericanBinomialEngine;
use openferric::engines::pde::CrankNicolsonEngine;
use openferric::instruments::{BermudanOption, VanillaOption};
use openferric::market::{Market, VolSurface};
use openferric::models::Heston;

#[derive(Clone, Copy)]
struct QlBermudanCase {
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    expiry: f64,
    dates: &'static [f64],
    quantlib_fd_price: f64,
}

const D4_1Y: &[f64] = &[0.249_315_068_5, 0.498_630_137_0, 0.750_684_931_5, 1.0];
const D6_15Y: &[f64] = &[
    0.249_315_068_5,
    0.501_369_863_0,
    0.750_684_931_5,
    1.0,
    1.252_054_794_5,
    1.501_369_863_0,
];
const D8_2Y: &[f64] = &[
    0.249_315_068_5,
    0.498_630_137_0,
    0.750_684_931_5,
    1.0,
    1.249_315_068_5,
    1.501_369_863_0,
    1.750_684_931_5,
    2.0,
];
const D6_2Y: &[f64] = &[
    0.334_246_575_3,
    0.665_753_424_7,
    1.0,
    1.334_246_575_3,
    1.665_753_424_7,
    2.0,
];
const D3_075Y: &[f64] = &[0.249_315_068_5, 0.501_369_863_0, 0.750_684_931_5];
const D5_05Y: &[f64] = &[0.098_630_137_0, 0.2, 0.298_630_137_0, 0.4, 0.498_630_137_0];
const D12_3Y: &[f64] = &[
    0.249_315_068_5,
    0.498_630_137_0,
    0.750_684_931_5,
    1.0,
    1.249_315_068_5,
    1.501_369_863_0,
    1.750_684_931_5,
    2.0,
    2.249_315_068_5,
    2.498_630_137_0,
    2.750_684_931_5,
    3.0,
];

fn quantlib_reference_cases() -> Vec<QlBermudanCase> {
    vec![
        QlBermudanCase {
            option_type: OptionType::Put,
            spot: 100.0,
            strike: 100.0,
            rate: 0.03,
            dividend: 0.00,
            vol: 0.20,
            expiry: 1.0,
            dates: D4_1Y,
            quantlib_fd_price: 6.658_125_199_2,
        },
        QlBermudanCase {
            option_type: OptionType::Put,
            spot: 95.0,
            strike: 100.0,
            rate: 0.05,
            dividend: 0.01,
            vol: 0.25,
            expiry: 1.501_369_863_0,
            dates: D6_15Y,
            quantlib_fd_price: 11.718_175_402_9,
        },
        QlBermudanCase {
            option_type: OptionType::Put,
            spot: 110.0,
            strike: 100.0,
            rate: 0.02,
            dividend: 0.00,
            vol: 0.30,
            expiry: 2.0,
            dates: D8_2Y,
            quantlib_fd_price: 11.428_287_704_3,
        },
        QlBermudanCase {
            option_type: OptionType::Call,
            spot: 100.0,
            strike: 95.0,
            rate: 0.04,
            dividend: 0.02,
            vol: 0.18,
            expiry: 1.0,
            dates: D4_1Y,
            quantlib_fd_price: 10.684_103_943_7,
        },
        QlBermudanCase {
            option_type: OptionType::Call,
            spot: 105.0,
            strike: 100.0,
            rate: 0.01,
            dividend: 0.03,
            vol: 0.22,
            expiry: 2.0,
            dates: D6_2Y,
            quantlib_fd_price: 13.138_338_528_3,
        },
        QlBermudanCase {
            option_type: OptionType::Put,
            spot: 80.0,
            strike: 90.0,
            rate: 0.06,
            dividend: 0.00,
            vol: 0.35,
            expiry: 0.750_684_931_5,
            dates: D3_075Y,
            quantlib_fd_price: 13.859_666_158_9,
        },
        QlBermudanCase {
            option_type: OptionType::Call,
            spot: 120.0,
            strike: 110.0,
            rate: 0.03,
            dividend: 0.00,
            vol: 0.15,
            expiry: 0.498_630_137_0,
            dates: D5_05Y,
            quantlib_fd_price: 12.712_610_941_6,
        },
        QlBermudanCase {
            option_type: OptionType::Put,
            spot: 150.0,
            strike: 140.0,
            rate: 0.025,
            dividend: 0.005,
            vol: 0.28,
            expiry: 3.0,
            dates: D12_3Y,
            quantlib_fd_price: 19.297_566_667_3,
        },
    ]
}

fn market(case_: &QlBermudanCase) -> Market {
    Market::builder()
        .spot(case_.spot)
        .rate(case_.rate)
        .dividend_yield(case_.dividend)
        .flat_vol(case_.vol)
        .build()
        .unwrap()
}

fn rel_err(got: f64, expected: f64) -> f64 {
    (got - expected).abs() / expected.abs().max(1.0e-12)
}

#[test]
fn bermudan_cn_matches_quantlib_fd_within_half_percent() {
    let engine = CrankNicolsonEngine::new(450, 450).with_s_max_multiplier(5.0);
    for (idx, case_) in quantlib_reference_cases().iter().enumerate() {
        let inst = BermudanOption::with_constant_strike(
            case_.option_type,
            case_.strike,
            case_.expiry,
            case_.dates.to_vec(),
        );
        let out = engine
            .price_bermudan_with_boundary(&inst, &market(case_))
            .unwrap();
        let err = rel_err(out.result.price, case_.quantlib_fd_price);
        assert!(
            err <= 0.005,
            "CN Bermudan case {idx} rel_err={err:.6} got={} ref={}",
            out.result.price,
            case_.quantlib_fd_price
        );
        assert!(
            !out.exercise_boundary.is_empty(),
            "boundary output missing for case {idx}"
        );
    }
}

#[test]
fn bermudan_lsm_matches_quantlib_fd_within_half_percent_on_reference_puts() {
    let engine = LongstaffSchwartzEngine::new(450_000, 120, 7);
    let put_cases = quantlib_reference_cases()
        .into_iter()
        .filter(|c| c.option_type == OptionType::Put)
        .collect::<Vec<_>>();
    for (idx, case_) in put_cases.iter().enumerate() {
        let inst = BermudanOption::with_constant_strike(
            case_.option_type,
            case_.strike,
            case_.expiry,
            case_.dates.to_vec(),
        );
        let out = engine
            .price_bermudan_with_boundary(&inst, &market(case_))
            .unwrap();
        let err = rel_err(out.result.price, case_.quantlib_fd_price);
        assert!(
            err <= 0.005,
            "LSM Bermudan put case {idx} rel_err={err:.6} got={} ref={} stderr={:?}",
            out.result.price,
            case_.quantlib_fd_price,
            out.result.stderr
        );
        assert!(
            out.exercise_boundary
                .iter()
                .any(|b| b.boundary_spot.is_some()),
            "boundary output has no populated points for put case {idx}"
        );
    }
}

#[test]
fn bermudan_time_varying_strike_step_down_put_prices_lower_than_constant_strike() {
    let dates = vec![0.25, 0.5, 0.75, 1.0];
    let step_down = BermudanOption::new(
        OptionType::Put,
        1.0,
        dates.clone(),
        vec![100.0, 97.5, 95.0, 92.5],
    );
    let constant = BermudanOption::with_constant_strike(OptionType::Put, 100.0, 1.0, dates);

    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.0)
        .flat_vol(0.25)
        .build()
        .unwrap();

    let lsm = LongstaffSchwartzEngine::new(250_000, 100, 42);
    let pde = CrankNicolsonEngine::new(300, 300).with_s_max_multiplier(5.0);

    let px_step_lsm = lsm.price(&step_down, &market).unwrap().price;
    let px_const_lsm = lsm.price(&constant, &market).unwrap().price;
    assert!(px_step_lsm < px_const_lsm);

    let px_step_pde = pde.price(&step_down, &market).unwrap().price;
    let px_const_pde = pde.price(&constant, &market).unwrap().price;
    assert!(px_step_pde < px_const_pde);
}

#[test]
fn bermudan_lsm_supports_local_vol_and_heston_dynamics() {
    let dates = vec![0.2, 0.4, 0.6, 0.8, 1.0];
    let inst = BermudanOption::with_constant_strike(OptionType::Put, 100.0, 1.0, dates);

    #[derive(Debug, Clone)]
    struct LocalSmile;
    impl VolSurface for LocalSmile {
        fn vol(&self, strike: f64, expiry: f64) -> f64 {
            let m = (strike / 100.0 - 1.0).abs();
            let term = (expiry.max(1.0e-6)).sqrt();
            (0.18 + 0.08 * m + 0.03 * term).max(0.05)
        }
    }

    let local_vol_market = Market::builder()
        .spot(100.0)
        .rate(0.02)
        .dividend_yield(0.01)
        .vol_surface(Box::new(LocalSmile))
        .build()
        .unwrap();

    let local_engine = LongstaffSchwartzEngine::new(200_000, 90, 123).with_local_vol_dynamics();
    let local_out = local_engine
        .price_bermudan_with_boundary(&inst, &local_vol_market)
        .unwrap();
    assert!(local_out.result.price.is_finite() && local_out.result.price > 0.0);
    assert!(
        local_out
            .exercise_boundary
            .iter()
            .any(|pt| pt.boundary_spot.is_some())
    );

    let heston = Heston {
        mu: 0.0,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.6,
        rho: -0.5,
        v0: 0.04,
    };
    let flat_market = Market::builder()
        .spot(100.0)
        .rate(0.02)
        .dividend_yield(0.01)
        .flat_vol(0.20)
        .build()
        .unwrap();
    let heston_engine =
        LongstaffSchwartzEngine::new(220_000, 100, 321).with_heston_dynamics(heston);
    let heston_out = heston_engine
        .price_bermudan_with_boundary(&inst, &flat_market)
        .unwrap();
    assert!(heston_out.result.price.is_finite() && heston_out.result.price > 0.0);
    assert!(
        heston_out
            .exercise_boundary
            .iter()
            .any(|pt| pt.boundary_spot.is_some())
    );
}

#[test]
fn bermudan_converges_toward_american_as_exercise_dates_increase() {
    let maturity = 1.0;
    let strike = 100.0;
    let market = Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.20)
        .build()
        .unwrap();
    let american = VanillaOption::american_put(strike, maturity);
    let american_ref = AmericanBinomialEngine::new(2500)
        .price(&american, &market)
        .unwrap()
        .price;

    let date_grid = |n: usize| -> Vec<f64> {
        (1..=n)
            .map(|i| maturity * i as f64 / n as f64)
            .collect::<Vec<_>>()
    };

    let pde = CrankNicolsonEngine::new(400, 400).with_s_max_multiplier(5.0);
    let berm_4 =
        BermudanOption::with_constant_strike(OptionType::Put, strike, maturity, date_grid(4));
    let berm_12 =
        BermudanOption::with_constant_strike(OptionType::Put, strike, maturity, date_grid(12));
    let berm_52 =
        BermudanOption::with_constant_strike(OptionType::Put, strike, maturity, date_grid(52));

    let p4 = pde.price(&berm_4, &market).unwrap().price;
    let p12 = pde.price(&berm_12, &market).unwrap().price;
    let p52 = pde.price(&berm_52, &market).unwrap().price;

    assert!(p12 >= p4 - 1.0e-8);
    assert!(p52 >= p12 - 1.0e-8);
    assert!(p52 <= american_ref + 0.05);
    assert!(
        rel_err(p52, american_ref) <= 0.01,
        "dense Bermudan should approach American: berm={p52} am={american_ref}"
    );

    let lsm = LongstaffSchwartzEngine::new(350_000, 120, 11);
    let l52 = lsm.price(&berm_52, &market).unwrap();
    assert!(l52.price <= american_ref + 0.20);
    assert!(
        rel_err(l52.price, american_ref) <= 0.03,
        "LSM dense Bermudan should be close to American: berm={} am={} stderr={:?}",
        l52.price,
        american_ref,
        l52.stderr
    );
}
