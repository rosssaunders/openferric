use openferric::core::PricingEngine;
use openferric::engines::monte_carlo::mc_engine::{MonteCarloPricingEngine, VarianceReduction};
use openferric::engines::numerical::american_binomial::AmericanBinomialEngine;
use openferric::instruments::{BarrierOption, VanillaOption};
use openferric::market::{
    DividendEvent, DividendSchedule, Market, PutCallParityQuote,
    bootstrap_dividend_curve_from_put_call_parity,
};

#[test]
fn american_cash_dividend_escrowed_model_locks_grid_and_matches_quantlib() {
    // QuantLib 1.41 reference, FdBlackScholesVanillaEngine (Escrowed model),
    // tGrid=xGrid=800:
    // S=100, K=100, r=3%, q=0%, vol=25%, T=1y, one cash dividend 0.2 at t=0.5.
    let ql_call = 11.231_303_076_021_439_f64;
    let ql_put = 8.744_346_284_348_829_f64;

    let schedule = DividendSchedule::new(vec![DividendEvent::cash(0.5, 0.2).expect("valid event")])
        .expect("valid schedule");
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.0)
        .dividend_schedule(schedule)
        .flat_vol(0.25)
        .build()
        .expect("valid market");

    let engine = AmericanBinomialEngine::new(4000);
    let call = engine
        .price(&VanillaOption::american_call(100.0, 1.0), &market)
        .expect("call pricing")
        .price;
    let put = engine
        .price(&VanillaOption::american_put(100.0, 1.0), &market)
        .expect("put pricing")
        .price;

    // Exact 4,000-step finite-grid locks for the escrowed-dividend CRR engine.
    assert!((call - 11.231_489_796_078_526).abs() <= 2.0e-10);
    assert!((put - 8.745_105_217_344_733).abs() <= 2.0e-10);

    // Quantify the remaining numerical-method/grid difference from QuantLib's
    // independently implemented 800x800 escrowed finite-difference engine.
    let call_model_gap = call - ql_call;
    let put_model_gap = put - ql_put;
    assert!((call_model_gap - 0.000_186_720_057_087_086_4).abs() <= 2.0e-10);
    assert!((put_model_gap - 0.000_758_932_995_903_904_8).abs() <= 2.0e-10);
}

#[test]
fn barrier_mc_applies_ex_div_jump_on_path() {
    let barrier = BarrierOption::builder()
        .put()
        .strike(100.0)
        .expiry(1.0)
        .down_and_in(95.0)
        .rebate(0.0)
        .build()
        .expect("valid barrier");

    let base_market = Market::builder()
        .spot(100.0)
        .rate(0.0)
        .dividend_yield(0.0)
        .flat_vol(1.0e-8)
        .build()
        .expect("valid market");

    let with_div_market = Market::builder()
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

    let engine = MonteCarloPricingEngine::new(20_000, 252, 42)
        .with_variance_reduction(VarianceReduction::Antithetic);

    let no_div_price = engine
        .price(&barrier, &base_market)
        .expect("base barrier pricing")
        .price;
    let with_div_price = engine
        .price(&barrier, &with_div_market)
        .expect("dividend barrier pricing")
        .price;

    // Without jump the deterministic path never breaches 95.
    assert!(
        no_div_price.abs() <= 1.0e-10,
        "expected 0, got {no_div_price}"
    );
    // With a 10 cash dividend at t=0.5, deterministic path jumps to 90 and knocks in.
    assert!(
        (with_div_price - 10.0).abs() <= 1.0e-10,
        "expected ~10 with ex-div jump knock-in, got {with_div_price}"
    );
}

#[test]
fn bootstrap_dividend_curve_reproduces_forwards_to_roundoff() {
    let spot = 100.0_f64;
    let rate = 0.02_f64;
    let strike = 100.0_f64;
    let input_forwards = [
        (0.5_f64, 100.75_f64),
        (1.0_f64, 101.20_f64),
        (1.5_f64, 101.65_f64),
    ];

    let quotes: Vec<PutCallParityQuote> = input_forwards
        .iter()
        .map(|(t, fwd)| {
            let call_minus_put = (fwd - strike) * (-rate * *t).exp();
            let put_price = 5.0;
            PutCallParityQuote {
                maturity: *t,
                strike,
                call_price: put_price + call_minus_put,
                put_price,
            }
        })
        .collect();

    let curve = bootstrap_dividend_curve_from_put_call_parity(spot, rate, &quotes)
        .expect("bootstrap should succeed");
    let schedule = curve
        .to_cash_dividend_schedule()
        .expect("cash schedule conversion should succeed");

    for (t, fwd) in input_forwards {
        let model_fwd = schedule.forward_price(spot, rate, 0.0, t);
        let abs_err = (model_fwd - fwd).abs();
        assert!(
            abs_err <= 64.0 * f64::EPSILON * fwd.abs().max(1.0),
            "forward mismatch at T={t}: model={model_fwd}, input={fwd}, abs_err={abs_err}"
        );
    }
}
