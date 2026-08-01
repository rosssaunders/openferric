use approx::assert_relative_eq;

use openferric::rates::{
    Frequency, InflationCurveBuilder, InflationIndexedBond, OvernightIndexSwap, XccySwap,
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

#[test]
fn xccy_quarterly_dual_curve_cashflows_match_high_precision_reference() {
    // Independent Python Decimal (80 digits) reference.  Each quarterly
    // floating coupon uses (DFp(t0)/DFp(t1)-1) and both legs are discounted
    // at their own continuously compounded flat curve:
    //   N1=100m, N2=90m, K=3%, spread=25bp, FX=1.1111, T=5
    //   r1=4%, r2_discount=3%, r2_projection=3.5%.
    let ccy1_discount = flat_curve_continuous(0.04, 5);
    let ccy2_discount = flat_curve_continuous(0.03, 5);
    let ccy2_projection = flat_curve_continuous(0.035, 5);
    let swap = XccySwap {
        notional1: 100_000_000.0,
        notional2: 90_000_000.0,
        fixed_rate: 0.03,
        float_spread: 0.0025,
        tenor: 5.0,
        fx_spot: 1.1111,
    };

    let fixed = swap.fixed_leg_pv_ccy1_with_frequency(&ccy1_discount, Frequency::Quarterly);
    let floating = swap.float_leg_pv_ccy2_with_frequency(
        &ccy2_discount,
        &ccy2_projection,
        Frequency::Quarterly,
    );
    let npv = swap.npv_dual_curve_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        Frequency::Quarterly,
        Frequency::Quarterly,
        true,
    );

    // Eight binary64 rounding units per accumulated leg also covers the
    // cancellation in NPV; this is about 1.7e-7 currency units, not a pricing
    // tolerance.
    let roundoff = 8.0 * f64::EPSILON * fixed.max(floating);
    assert!((fixed - 95_400_406.152_444_3).abs() <= roundoff);
    assert!((floating - 93_139_314.121_691_67).abs() <= roundoff);
    assert!((npv - 8_086_685.768_167_322).abs() <= roundoff);

    let par = swap.par_fixed_rate_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        Frequency::Quarterly,
        Frequency::Quarterly,
    );
    assert_relative_eq!(par, 0.047_934_105_096_648_605, epsilon = 2.0e-16);
    let par_swap = XccySwap {
        fixed_rate: par,
        ..swap
    };
    let par_fixed = par_swap.fixed_leg_pv_ccy1_with_frequency(&ccy1_discount, Frequency::Quarterly);
    let par_floating = par_swap.float_leg_pv_ccy2_with_frequency(
        &ccy2_discount,
        &ccy2_projection,
        Frequency::Quarterly,
    ) * par_swap.fx_spot;
    let par_npv = par_swap.npv_dual_curve_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        Frequency::Quarterly,
        Frequency::Quarterly,
        true,
    );
    let cancellation_roundoff =
        8.0 * f64::EPSILON * par_fixed.abs().max(par_floating.abs()).max(1.0);
    assert!(
        par_npv.abs() <= cancellation_roundoff,
        "quarterly XCCY par residual {par_npv:e} exceeds cancellation budget \
         {cancellation_roundoff:e}"
    );
}

#[test]
fn xccy_legacy_methods_are_exact_annual_frequency_wrappers() {
    let ccy1_discount = flat_curve_continuous(0.04, 6);
    let ccy2_discount = flat_curve_continuous(0.03, 6);
    let ccy2_projection = flat_curve_continuous(0.035, 6);
    let swap = XccySwap {
        notional1: 125_000_000.0,
        notional2: 100_000_000.0,
        fixed_rate: 0.042,
        float_spread: 0.0015,
        tenor: 4.5,
        fx_spot: 1.25,
    };

    assert_eq!(
        swap.fixed_leg_pv_ccy1(&ccy1_discount),
        swap.fixed_leg_pv_ccy1_with_frequency(&ccy1_discount, Frequency::Annual)
    );
    assert_eq!(
        swap.float_leg_pv_ccy2(&ccy2_discount, &ccy2_projection),
        swap.float_leg_pv_ccy2_with_frequency(&ccy2_discount, &ccy2_projection, Frequency::Annual,)
    );
    assert_eq!(
        swap.npv_dual_curve(&ccy1_discount, &ccy2_discount, &ccy2_projection, true,),
        swap.npv_dual_curve_with_frequencies(
            &ccy1_discount,
            &ccy2_discount,
            &ccy2_projection,
            Frequency::Annual,
            Frequency::Annual,
            true,
        )
    );
    assert_eq!(
        swap.par_fixed_rate(&ccy1_discount, &ccy2_discount, &ccy2_projection),
        swap.par_fixed_rate_with_frequencies(
            &ccy1_discount,
            &ccy2_discount,
            &ccy2_projection,
            Frequency::Annual,
            Frequency::Annual,
        )
    );
    assert_eq!(
        swap.mtm_basis_npv(
            &ccy1_discount,
            &ccy2_discount,
            &ccy2_projection,
            1.31,
            false,
        ),
        swap.mtm_basis_npv_with_frequencies(
            &ccy1_discount,
            &ccy2_discount,
            &ccy2_projection,
            1.31,
            Frequency::Annual,
            Frequency::Annual,
            false,
        )
    );
}

fn xccy_stub_fixture() -> (XccySwap, YieldCurve, YieldCurve, YieldCurve) {
    (
        XccySwap {
            notional1: 1_000_000.0,
            notional2: 800_000.0,
            fixed_rate: 0.027,
            float_spread: 0.0012,
            tenor: 1.1,
            fx_spot: 1.25,
        },
        flat_curve_continuous(0.04, 2),
        flat_curve_continuous(0.03, 2),
        flat_curve_continuous(0.035, 2),
    )
}

#[test]
fn xccy_non_integral_tenor_final_stubs_match_decimal_reference() {
    // Independent 80-digit Decimal sum. The semiannual fixed leg has a final
    // 0.1-year stub after t=1.0; the quarterly floating leg does likewise.
    let (swap, ccy1_discount, ccy2_discount, ccy2_projection) = xccy_stub_fixture();
    let fixed = swap.fixed_leg_pv_ccy1_with_frequency(&ccy1_discount, Frequency::SemiAnnual);
    let floating = swap.float_leg_pv_ccy2_with_frequency(
        &ccy2_discount,
        &ccy2_projection,
        Frequency::Quarterly,
    );
    let npv = swap.npv_dual_curve_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        Frequency::SemiAnnual,
        Frequency::Quarterly,
        true,
    );
    let par = swap.par_fixed_rate_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        Frequency::SemiAnnual,
        Frequency::Quarterly,
    );

    let price_roundoff = 128.0 * f64::EPSILON * fixed.max(floating);
    assert!((fixed - 985_741.072_676_421_5).abs() <= price_roundoff);
    assert!((floating - 805_381.224_181_015_9).abs() <= price_roundoff);
    assert!((npv - 20_985.457_549_848_36).abs() <= price_roundoff);
    let rate_roundoff = 128.0 * f64::EPSILON * par.abs();
    assert!((par - 0.046_682_672_259_548_98).abs() <= rate_roundoff);
}

#[test]
fn xccy_explicit_frequency_mtm_matches_decimal_reference() {
    let (swap, ccy1_discount, ccy2_discount, ccy2_projection) = xccy_stub_fixture();
    let pay_fixed = swap.mtm_basis_npv_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        1.30,
        Frequency::SemiAnnual,
        Frequency::Quarterly,
        true,
    );
    let receive_fixed = swap.mtm_basis_npv_with_frequencies(
        &ccy1_discount,
        &ccy2_discount,
        &ccy2_projection,
        1.30,
        Frequency::SemiAnnual,
        Frequency::Quarterly,
        false,
    );

    let roundoff = 128.0 * f64::EPSILON * 1_000_000.0;
    assert!((pay_fixed - 61_254.518_758_899_15).abs() <= roundoff);
    assert_eq!(receive_fixed, -pay_fixed);
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

    let fixed_leg_expected = template.notional1
        * template.fixed_rate
        * (1..=5).map(|year| (-0.04 * year as f64).exp()).sum::<f64>()
        + template.notional1 * (-0.04_f64 * 5.0).exp();
    let float_leg_ccy2_expected = template.notional2
        * (0.03_f64.exp() - 1.0 + template.float_spread)
        * (1..=5).map(|year| (-0.03 * year as f64).exp()).sum::<f64>()
        + template.notional2 * (-0.03_f64 * 5.0).exp();
    let expected_npv = float_leg_ccy2_expected * template.fx_spot - fixed_leg_expected;
    assert_relative_eq!(
        template.npv(&usd_curve, &eur_curve, true),
        expected_npv,
        epsilon = 1.0e-7
    );
    assert_relative_eq!(expected_npv, 5_944_253.381_263_331, epsilon = 1.0e-7);

    let par_fixed = template.par_fixed_rate(&usd_curve, &eur_curve, &eur_curve);
    let par_swap = XccySwap {
        fixed_rate: par_fixed,
        ..template
    };

    let fixed_leg = par_swap.fixed_leg_pv_ccy1(&usd_curve);
    let float_leg_ccy1 = par_swap.float_leg_pv_ccy2(&eur_curve, &eur_curve) * par_swap.fx_spot;

    // Each annual leg accumulates five coupons plus principal.  Bound the
    // at-par cancellation by the leg scale, not a fixed currency tolerance.
    let cancellation_roundoff =
        64.0 * f64::EPSILON * fixed_leg.abs().max(float_leg_ccy1.abs()).max(1.0);
    let par_npv = par_swap.npv(&usd_curve, &eur_curve, true);
    assert!(
        (fixed_leg - float_leg_ccy1).abs() <= cancellation_roundoff,
        "XCCY par-leg residual {:e} exceeds cancellation budget {cancellation_roundoff:e}",
        fixed_leg - float_leg_ccy1
    );
    assert!(
        par_npv.abs() <= cancellation_roundoff,
        "XCCY par NPV residual {par_npv:e} exceeds cancellation budget {cancellation_roundoff:e}"
    );
}

#[test]
fn xccy_mtm_basis_npv_has_exact_fx_revaluation() {
    let ccy1 = flat_curve_continuous(0.04, 7);
    let ccy2_discount = flat_curve_continuous(0.03, 7);
    let ccy2_projection = flat_curve_continuous(0.035, 7);
    let swap = XccySwap {
        notional1: 125_000_000.0,
        notional2: 100_000_000.0,
        fixed_rate: 0.042,
        float_spread: 0.0015,
        tenor: 4.5,
        fx_spot: 1.25,
    };
    let current_fx = 1.31;

    let fixed = swap.fixed_leg_pv_ccy1(&ccy1);
    let floating = swap.float_leg_pv_ccy2(&ccy2_discount, &ccy2_projection);
    let expected_pay_fixed = current_fx * floating - fixed;
    let actual_pay_fixed =
        swap.mtm_basis_npv(&ccy1, &ccy2_discount, &ccy2_projection, current_fx, true);
    let roundoff = 32.0 * f64::EPSILON * expected_pay_fixed.abs().max(floating);
    assert!((actual_pay_fixed - expected_pay_fixed).abs() <= roundoff);

    let actual_receive_fixed =
        swap.mtm_basis_npv(&ccy1, &ccy2_discount, &ccy2_projection, current_fx, false);
    assert_eq!(actual_receive_fixed, -actual_pay_fixed);

    // MTM is affine in the externally supplied spot FX; the exact slope is
    // the currency-2 floating-leg PV expressed per unit of spot FX.
    let inception = swap.npv_dual_curve(&ccy1, &ccy2_discount, &ccy2_projection, true);
    let exact_change = (current_fx - swap.fx_spot) * floating;
    assert!((actual_pay_fixed - inception - exact_change).abs() <= roundoff);
}

#[test]
fn quantlib_usd_try_curve_nodes_match_exact_annual_and_mixed_frequency_cashflows() {
    // Source: QuantLib test-suite/constnotionalcrosscurrencyswap.cpp
    // testFloatFixXCCYSwapPricing. The source supplies the curve nodes and an
    // annual TRY fixed / quarterly USD floating frequency pairing. Its cached
    // NPV also includes dated calendars and day counts, so the compatible
    // year-fraction models below are asserted against independent Decimal
    // cashflow sums instead of against that different contract.
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

    // Deterministic closed-form value under the library's annual
    // simple-forward model, computed from the same curve nodes with explicit
    // arithmetic: USD float leg pays N2 * (DFp(i-1)/DFp(i) - 1) annually plus
    // the terminal notional, all discounted on the USD discount curve; the
    // TRY fixed leg pays N1 * fixed_rate annually plus the terminal notional
    // on the TRY discount curve.
    let mut float_pv_usd = 0.0;
    for i in 1..=5_u32 {
        let dfp_prev = usd_projection.discount_factor((i - 1) as f64);
        let dfp_curr = usd_projection.discount_factor(i as f64);
        let fwd = dfp_prev / dfp_curr - 1.0;
        float_pv_usd += swap.notional2 * fwd * usd_discount.discount_factor(i as f64);
    }
    float_pv_usd += swap.notional2 * usd_discount.discount_factor(5.0);

    let mut fixed_pv_try = 0.0;
    for i in 1..=5_u32 {
        fixed_pv_try += swap.notional1 * swap.fixed_rate * try_discount.discount_factor(i as f64);
    }
    fixed_pv_try += swap.notional1 * try_discount.discount_factor(5.0);

    let expected_npv_usd = (float_pv_usd * fx_spot - fixed_pv_try) / fx_spot;
    // This NPV is the small residual of two roughly 10m notionals. Bound only
    // binary64 accumulation/cancellation error; a relative price band would
    // silently permit a materially wrong residual.
    let cashflow_roundoff = 64.0 * f64::EPSILON * float_pv_usd.max(fixed_pv_try / fx_spot);
    assert!((npv_usd - expected_npv_usd).abs() <= cashflow_roundoff);

    // QuantLib's 218,961.99 cached value is intentionally not asserted against
    // this year-fraction contract. With the source's actual annual-fixed /
    // quarterly-floating frequencies, an independent 60-digit Decimal sum of
    // the log-linearly interpolated nodes gives the exact compatible value.
    let mixed_frequency_npv_usd = swap.npv_dual_curve_with_frequencies(
        &try_discount,
        &usd_discount,
        &usd_projection,
        Frequency::Annual,
        Frequency::Quarterly,
        true,
    ) / fx_spot;
    let mixed_reference = 237_563.442_390_497_74;
    assert!((mixed_frequency_npv_usd - mixed_reference).abs() <= cashflow_roundoff);

    let par_fixed = swap.par_fixed_rate(&try_discount, &usd_discount, &usd_projection);
    let par_swap = XccySwap {
        fixed_rate: par_fixed,
        ..swap
    };
    let par_fixed_leg_try = par_swap.fixed_leg_pv_ccy1(&try_discount);
    let par_float_leg_try = par_swap.float_leg_pv_ccy2(&usd_discount, &usd_projection) * fx_spot;
    let par_npv = par_swap.npv_dual_curve(&try_discount, &usd_discount, &usd_projection, true);
    // The rate is solved from these same annual legs, so four rounding units
    // at leg scale cover the final multiply/add/subtract cancellation while
    // remaining stricter than the former 1e-7 currency-unit band.
    let par_cancellation_roundoff = 4.0
        * f64::EPSILON
        * par_fixed_leg_try
            .abs()
            .max(par_float_leg_try.abs())
            .max(1.0);
    assert!(
        par_npv.abs() <= par_cancellation_roundoff,
        "TRY/USD par residual {par_npv:e} exceeds cancellation budget {par_cancellation_roundoff:e}"
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

    let mtm = swap.mtm(1.0, 103.0, &discount_curve, &inflation_curve);
    let projected_terminal_cpi = 103.0 * 1.025_f64.powi(4);
    let expected_mtm = swap.notional
        * (projected_terminal_cpi / swap.cpi_base - 1.0 - (1.0 + swap.fixed_rate).powf(swap.tenor)
            + 1.0)
        * (-0.02_f64 * 4.0).exp();
    assert_relative_eq!(mtm, expected_mtm, epsilon = 1.0e-7);
    assert_relative_eq!(mtm, 509_473.861_344_120_8, epsilon = 1.0e-7);
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
        let par_npv = swap.npv_from_curve(&discount_curve, &inflation_curve);
        let growth = (1.0 + quote).powf(tenor);
        let discounted_leg_scale = swap.notional * growth * discount_curve.discount_factor(tenor);
        // Projection and fixed growth are the same quote-derived amount.  The
        // residual can therefore contain only the terminal-CPI divide and
        // payoff cancellation roundoff.
        let cancellation_roundoff = 32.0 * f64::EPSILON * discounted_leg_scale.abs().max(1.0);
        assert!(
            par_npv.abs() <= cancellation_roundoff,
            "{tenor}Y ZC inflation par residual {par_npv:e} exceeds cancellation budget {cancellation_roundoff:e}"
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

    // Compounded overnight over unit annual periods on a flat continuous
    // curve pays the simple annual equivalent e^r - 1.
    let par = 0.035_f64.exp() - 1.0;
    let swap = OvernightIndexSwap {
        notional: 100_000_000.0,
        fixed_rate: par,
        float_spread: 0.0,
        tenor: 2.0,
    };

    assert_relative_eq!(
        swap.par_fixed_rate(&ois_curve, &ois_curve),
        par,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        swap.npv(&ois_curve, &ois_curve, true),
        0.0,
        epsilon = 1.0e-8
    );
}
