use openferric::core::PricingEngine;
use openferric::engines::analytic::BlackScholesEngine;
use openferric::engines::tree::BinomialTreeEngine;
use openferric::instruments::VanillaOption;
use openferric::market::{DividendEvent, DividendSchedule, Market, VolSource};
use openferric::vol::surface::{SviParams, VolSurface};

fn valid_market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.2)
        .build()
        .unwrap()
}

#[test]
fn high_level_engines_revalidate_directly_mutated_or_deserialized_markets() {
    let option = VanillaOption::european_call(100.0, 1.0);
    let analytic = BlackScholesEngine::new();
    let tree = BinomialTreeEngine::new(32);

    let mut invalid_markets = Vec::new();
    for mutation in 0..4 {
        let mut market = valid_market();
        match mutation {
            0 => market.spot = f64::NAN,
            1 => market.rate = f64::INFINITY,
            2 => market.dividend_yield = f64::NEG_INFINITY,
            3 => market.vol = VolSource::Flat(f64::NAN),
            _ => unreachable!(),
        }
        invalid_markets.push(market);
    }

    for market in invalid_markets {
        let analytic_result = analytic.price(&option, &market);
        assert!(
            analytic_result.is_err(),
            "analytic engine accepted {market:?}: {analytic_result:?}"
        );

        let tree_result = tree.price(&option, &market);
        assert!(
            tree_result.is_err(),
            "tree engine accepted {market:?}: {tree_result:?}"
        );
    }
}

#[test]
fn engines_reject_non_finite_result_from_a_structurally_valid_surface_query() {
    // These finite SVI parameters satisfy the surface-domain constraints and
    // remain finite around the forward, but b * |log(K/F)| overflows for the
    // extreme (still finite) strike queried below.
    let surface = VolSurface::new(
        vec![(
            1.0,
            SviParams {
                a: 0.0,
                b: 1.0e308,
                rho: 0.0,
                m: 0.0,
                sigma: 1.0e-308,
            },
        )],
        100.0,
    )
    .expect("surface is structurally valid");
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .vol_surface(Box::new(surface))
        .build()
        .expect("surface remains finite on the builder validation grid");
    market.validate().expect("market structure is valid");

    let strike = 100.0 * 10.0_f64.exp();
    assert!(market.vol_for(strike, 1.0).is_infinite());
    assert!(market.checked_vol_for(strike, 1.0).is_err());

    let option = VanillaOption::european_call(strike, 1.0);
    for result in [
        BlackScholesEngine::new().price(&option, &market),
        BinomialTreeEngine::new(32).price(&option, &market),
    ] {
        assert!(
            result.is_err(),
            "engine must not return an Ok result containing NaN/inf: {result:?}"
        );
    }

    // Expired instruments do not require a volatility observation. Both the
    // analytic and AAD entry points must return the same intrinsic boundary
    // value even when a query at that strike would overflow.
    let expired = VanillaOption::european_put(strike, 0.0);
    let engine = BlackScholesEngine::new();
    let analytic = engine.price(&expired, &market).unwrap();
    let aad = engine.price_with_greeks_aad(&expired, &market).unwrap();
    assert_eq!(analytic.price, strike - market.spot);
    assert_eq!(aad.price, analytic.price);
    for result in [analytic, aad] {
        let greeks = result.greeks.unwrap();
        assert_eq!(greeks.delta, 0.0);
        assert_eq!(greeks.gamma, 0.0);
        assert_eq!(greeks.vega, 0.0);
        assert_eq!(greeks.theta, 0.0);
        assert_eq!(greeks.rho, 0.0);
    }
}

#[test]
fn analytic_and_aad_prices_preserve_tiny_atm_time_value() {
    let market = Market::builder()
        .spot(100.0)
        .rate(0.0)
        .dividend_yield(0.0)
        .flat_vol(1.0e-16)
        .build()
        .unwrap();
    let option = VanillaOption::european_call(100.0, 1.0);
    let expected = 100.0 * 0.398_942_280_401_432_7e-16;
    let engine = BlackScholesEngine::new();

    for result in [
        engine.price(&option, &market).unwrap(),
        engine.price_with_greeks_aad(&option, &market).unwrap(),
    ] {
        let relative_error = ((result.price - expected) / expected).abs();
        assert!(
            relative_error <= 8.0 * f64::EPSILON,
            "price={} reference={expected} rel={relative_error}",
            result.price
        );
    }
}

#[test]
fn analytic_and_aad_greeks_agree_when_cash_dividends_exhaust_prepaid_spot() {
    let schedule = DividendSchedule::new(vec![DividendEvent::cash(0.5, 200.0).unwrap()]).unwrap();
    let market = Market::builder()
        .spot(100.0)
        .rate(0.0)
        .dividend_yield(0.0)
        .dividend_schedule(schedule)
        .flat_vol(0.2)
        .build()
        .unwrap();
    assert_eq!(market.prepaid_forward_spot(1.0), 0.0);

    let option = VanillaOption::european_put(100.0, 1.0);
    let engine = BlackScholesEngine::new();
    let analytic = engine.price(&option, &market).unwrap();
    let aad = engine.price_with_greeks_aad(&option, &market).unwrap();
    assert_eq!(analytic.price, 100.0);
    assert_eq!(aad.price, analytic.price);
    assert_eq!(aad.greeks, analytic.greeks);

    let greeks = analytic.greeks.unwrap();
    assert_eq!(greeks.delta, -1.0);
    assert_eq!(greeks.gamma, 0.0);
    assert_eq!(greeks.vega, 0.0);
    assert_eq!(greeks.theta, 0.0);
    assert_eq!(greeks.rho, -100.0);
}
