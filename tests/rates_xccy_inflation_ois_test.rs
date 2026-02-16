use approx::assert_relative_eq;

use openferric::rates::{
    InflationCurveBuilder, OvernightIndexSwap, XccySwap, YieldCurve, ZeroCouponInflationSwap,
};

fn flat_curve_continuous(rate: f64, max_tenor_years: u32) -> YieldCurve {
    let tenors = (1..=max_tenor_years)
        .map(|t| {
            let tf = t as f64;
            (tf, (-rate * tf).exp())
        })
        .collect();
    YieldCurve::new(tenors)
}

#[test]
fn xccy_swap_usd_eur_par_trade_npv_is_near_zero_at_inception() {
    let usd_curve = flat_curve_continuous(0.04, 10);
    let eur_curve = flat_curve_continuous(0.03, 10);

    let template = XccySwap {
        notional1: 100_000_000.0,
        notional2: 90_000_000.0,
        fixed_rate: 0.03,
        float_spread: 0.0025,
        tenor: 5.0,
        fx_spot: 1.1111,
    };

    let npv_given_rate = template.npv(&usd_curve, &eur_curve, true);
    assert!(npv_given_rate.is_finite());

    let par_fixed = template.par_fixed_rate(&usd_curve, &eur_curve, &eur_curve);
    let par_swap = XccySwap {
        fixed_rate: par_fixed,
        ..template
    };

    let fixed_leg = par_swap.fixed_leg_pv_ccy1(&usd_curve);
    let float_leg_ccy1 = par_swap.float_leg_pv_ccy2(&eur_curve, &eur_curve) * par_swap.fx_spot;

    assert_relative_eq!(fixed_leg, float_leg_ccy1, epsilon = 1.0e-5);
    assert_relative_eq!(
        par_swap.npv(&usd_curve, &eur_curve, true),
        0.0,
        epsilon = 1.0e-5
    );
}

#[test]
fn zc_inflation_swap_is_par_at_inception_and_positive_after_higher_realized_inflation() {
    let discount_curve = flat_curve_continuous(0.02, 10);
    let inflation_curve = InflationCurveBuilder::from_zc_swap_rates(&[(1.0, 0.025), (5.0, 0.025)]);

    let swap = ZeroCouponInflationSwap {
        notional: 100_000_000.0,
        cpi_base: 100.0,
        fixed_rate: 0.025,
        tenor: 5.0,
        receive_inflation: true,
    };

    assert_relative_eq!(
        swap.npv_from_curve(&discount_curve, &inflation_curve),
        0.0,
        epsilon = 1.0e-8
    );

    // After one year, CPI has realized +3% (103 vs 100), above the 2.5% fixed leg.
    let mtm = swap.mtm(1.0, 103.0, &discount_curve, &inflation_curve);
    assert!(mtm > 0.0);
}

#[test]
fn ois_swap_npv_is_near_zero_at_flat_par_rate() {
    let ois_curve = flat_curve_continuous(0.035, 10);

    let swap = OvernightIndexSwap {
        notional: 100_000_000.0,
        fixed_rate: 0.035,
        float_spread: 0.0,
        tenor: 2.0,
    };

    assert_relative_eq!(
        swap.par_fixed_rate(&ois_curve, &ois_curve),
        0.035,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        swap.npv(&ois_curve, &ois_curve, true),
        0.0,
        epsilon = 1.0e-8
    );
}
