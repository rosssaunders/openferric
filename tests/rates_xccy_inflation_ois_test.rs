use approx::assert_relative_eq;

use openferric::rates::{
    InflationCurveBuilder, InflationIndexedBond, OvernightIndexSwap, XccySwap,
    YearOnYearInflationSwap, YieldCurve, ZeroCouponInflationSwap,
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

fn quantlib_usd_discount_curve_annual_nodes() -> YieldCurve {
    // Source: QuantLib test-suite/constnotionalcrosscurrencyswap.cpp USDDiscountCurve.
    YieldCurve::new(vec![
        (1.0, 0.975_723_193_871_552),
        (2.0, 0.948_588_232_418_325),
        (3.0, 0.922_796_367_734_64),
        (4.0, 0.898_345_201_557_914),
        (5.0, 0.874_715_322_269_088),
    ])
}

fn quantlib_usd_projection_curve_annual_nodes() -> YieldCurve {
    // Source: QuantLib test-suite/constnotionalcrosscurrencyswap.cpp USDProjectionCurve.
    YieldCurve::new(vec![
        (1.0, 0.972_708_376_777_628),
        (2.0, 0.943_264_331_984_248),
        (3.0, 0.914_816_470_778_467),
        (4.0, 0.887_647_146_416_23),
        (5.0, 0.861_475_671_008_934),
    ])
}

fn quantlib_try_discount_curve_annual_nodes() -> YieldCurve {
    // Source: QuantLib test-suite/constnotionalcrosscurrencyswap.cpp TRYDiscountCurve.
    YieldCurve::new(vec![
        (1.0, 0.763_745_028_010_31),
        (2.0, 0.595_566_112_318_217),
        (3.0, 0.483_132_147_134_316),
        (4.0, 0.402_466_076_327_945),
        (5.0, 0.345_531_820_837_392),
    ])
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
fn quantlib_usd_try_const_notional_xccy_cached_npv_is_close_under_annual_model() {
    // Source: QuantLib test-suite/constnotionalcrosscurrencyswap.cpp
    // testFloatFixXCCYSwapPricing. QuantLib uses full date schedules and
    // quarterly USD Libor; OpenFerric's current XCCY API is annualized, so
    // this checks the overlapping fixed-vs-floating economics within model
    // granularity rather than exact coupon-level BPS.
    let usd_discount = quantlib_usd_discount_curve_annual_nodes();
    let usd_projection = quantlib_usd_projection_curve_annual_nodes();
    let try_discount = quantlib_try_discount_curve_annual_nodes();
    let fx_spot = 6.4304;
    let swap = XccySwap {
        notional1: 10_000_000.0 * fx_spot,
        notional2: 10_000_000.0,
        fixed_rate: 0.249,
        float_spread: 0.0,
        tenor: 5.0,
        fx_spot,
    };

    let npv_try = swap.npv_dual_curve(&try_discount, &usd_discount, &usd_projection, true);
    let npv_usd = npv_try / fx_spot;

    assert_relative_eq!(npv_usd, 218_961.99, epsilon = 750.0);

    let par_fixed = swap.par_fixed_rate(&try_discount, &usd_discount, &usd_projection);
    let par_swap = XccySwap {
        fixed_rate: par_fixed,
        ..swap
    };
    assert_relative_eq!(
        par_swap.npv_dual_curve(&try_discount, &usd_discount, &usd_projection, true),
        0.0,
        epsilon = 1.0e-7
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
fn quantlib_uk_rpi_zero_inflation_quote_grid_reprices_zc_swaps() {
    // Source: QuantLib test-suite/inflation.cpp testZeroTermStructure zcData table.
    let quantlib_zc_rates = [
        (1.0, 0.0293),
        (2.0, 0.0295),
        (3.0, 0.02965),
        (4.0, 0.0298),
        (5.0, 0.03),
        (7.0, 0.0306),
        (10.0, 0.03175),
        (12.0, 0.03243),
        (15.0, 0.03293),
        (20.0, 0.03338),
        (25.0, 0.03348),
        (30.0, 0.03348),
        (40.0, 0.03308),
        (50.0, 0.03228),
    ];
    let inflation_curve = InflationCurveBuilder::from_zc_swap_rates(&quantlib_zc_rates);
    let discount_curve = flat_curve_continuous(0.04, 50);

    for &(tenor, quote) in &quantlib_zc_rates {
        let swap = ZeroCouponInflationSwap {
            notional: 1_000_000.0,
            cpi_base: 100.0,
            fixed_rate: quote,
            tenor,
            receive_inflation: true,
        };
        assert_relative_eq!(
            inflation_curve.zero_inflation_rate(tenor),
            quote,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            swap.npv_from_curve(&discount_curve, &inflation_curve),
            0.0,
            epsilon = 1.0e-6
        );
    }
}

#[test]
fn quantlib_rpi_fixings_drive_yoy_inflation_swap_cashflows() {
    // Source: QuantLib test-suite/inflation.cpp UK RPI fixData, Jan-2005..Jan-2007.
    let cpi_fixings = [189.9, 194.1, 202.7];
    let fixed_rate = 0.03;
    let discount_curve = flat_curve_continuous(0.04, 2);
    let swap = YearOnYearInflationSwap {
        notional: 1_000_000.0,
        fixed_rate,
        maturity_years: 2,
        receive_inflation: true,
    };

    let expected = (cpi_fixings[1] / cpi_fixings[0] - 1.0 - fixed_rate)
        * 1_000_000.0
        * discount_curve.discount_factor(1.0)
        + (cpi_fixings[2] / cpi_fixings[1] - 1.0 - fixed_rate)
            * 1_000_000.0
            * discount_curve.discount_factor(2.0);

    assert_relative_eq!(
        swap.npv_from_fixings(&discount_curve, &cpi_fixings),
        expected,
        epsilon = 1.0e-8
    );
    assert!(swap.npv_from_fixings(&discount_curve, &[189.9]).is_nan());
}

#[test]
fn inflation_indexed_bond_prices_projected_quantlib_rpi_principal_and_coupons() {
    // Source: QuantLib inflation.cpp zero-inflation quotes and UK RPI base fixing.
    let inflation_curve =
        InflationCurveBuilder::from_zc_swap_rates(&[(1.0, 0.0293), (2.0, 0.0295)]);
    let nominal_curve = flat_curve_continuous(0.04, 2);
    let cpi_base = 189.9;
    let bond = InflationIndexedBond {
        face_value: 1_000.0,
        coupon_rate: 0.01,
        maturity_years: 2,
        coupon_frequency: 1,
        cpi_base,
    };

    let principal_1y = 1_000.0 * inflation_curve.projected_cpi(cpi_base, 1.0) / cpi_base;
    let principal_2y = 1_000.0 * inflation_curve.projected_cpi(cpi_base, 2.0) / cpi_base;
    let expected = principal_1y * 0.01 * nominal_curve.discount_factor(1.0)
        + principal_2y * 1.01 * nominal_curve.discount_factor(2.0);

    assert_relative_eq!(
        bond.indexed_principal(inflation_curve.projected_cpi(cpi_base, 2.0)),
        principal_2y,
        epsilon = 1.0e-10
    );
    assert_relative_eq!(
        bond.price(&nominal_curve, &inflation_curve),
        expected,
        epsilon = 1.0e-10
    );
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
