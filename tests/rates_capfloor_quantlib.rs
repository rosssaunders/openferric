//! Black cap/floor references generated offline with QuantLib-Python 1.43.
//!
//! Provenance: Python 3.11.15, QuantLib 1.43, SciPy 1.17.1, NumPy 2.4.3, and
//! mpmath 1.4.1 (installed in the reference environment, but not needed here).
//! `Settings.evaluationDate` is 2024-01-01.  Each QuantLib contract uses a
//! `WeekendsOnly` calendar, Modified Following adjustment, Actual/365 (Fixed),
//! zero fixing days, and a custom Ibor index forecast from the same continuously
//! compounded `FlatForward` curve used for discounting.  NPVs and optionlet
//! prices come from `BlackCapFloorEngine`; all values below are frozen literals,
//! so neither QuantLib nor the citation-only vendor submodule is used at runtime.
//!
//! The same adjusted dates were also evaluated by a separate Python strip that
//! formed `F=(DF(t1)/DF(t2)-1)/accrual` and called SciPy's `special.ndtr` in the
//! Black-76 formula.  Across the 16 aggregate fixtures the largest measured
//! `|QuantLib - SciPy strip|` was 3.7834979593753815e-10 currency units.  Each
//! case records its own measured gap.  The Linux Rust-vs-QuantLib session
//! measured a maximum 3.7834979593753815e-10 gap.  A subsequent macOS CI run
//! measured a 1.2096279533579946e-10 maximum on cases whose Linux gap was near
//! zero: the discount-factor ratio subtracts two notional-scale quantities, so
//! the expected option price is not the correct roundoff scale.  Assertions
//! therefore allow four times the per-case oracle gap, 64 scaled epsilons at
//! the fixed notional input scale for cross-libm operation order, and 64 ULP
//! for the final strip sum.  This is an arithmetic budget, not a price band.

use chrono::NaiveDate;

use openferric::rates::{
    CapFloor, DayCountConvention, Frequency, YieldCurve, generate_schedule, year_fraction,
};

const NOTIONAL: f64 = 1_000_000.0;
const START_YEAR: i32 = 2024;

fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).unwrap()
}

fn start_date() -> NaiveDate {
    date(START_YEAR, 1, 1)
}

fn curve_on_schedule(rate: f64, end_date: NaiveDate, frequency: Frequency) -> YieldCurve {
    let start = start_date();
    let nodes = generate_schedule(start, end_date, frequency)
        .into_iter()
        .skip(1)
        .map(|payment_date| {
            let t = year_fraction(start, payment_date, DayCountConvention::Act365Fixed);
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(nodes)
}

fn ulp(value: f64) -> f64 {
    let magnitude = value.abs();
    magnitude.next_up() - magnitude
}

fn assert_quantlib_black_reference(
    actual: f64,
    expected: f64,
    measured_scipy_gap: f64,
    case: &str,
) {
    let measured_oracle_budget = 4.0 * measured_scipy_gap;
    let cross_platform_libm_budget = 64.0 * f64::EPSILON * NOTIONAL;
    let strip_sum_roundoff_budget = 64.0 * ulp(expected.abs().max(1.0));
    let tolerance = measured_oracle_budget + cross_platform_libm_budget + strip_sum_roundoff_budget;
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{case}: actual={actual:.17}, QuantLib={expected:.17}, error={error:e}, \
         measured-oracle budget={measured_oracle_budget:e}, cross-libm \
         budget={cross_platform_libm_budget:e}, strip-sum \
         roundoff={strip_sum_roundoff_budget:e}, tolerance={tolerance:e}"
    );
}

#[test]
fn capfloor_schedules_match_quantlib_weekends_only_modified_following() {
    let start = start_date();

    // An offline scan over a complete 400-year Gregorian cycle found a
    // weekday-only start for this 2Y quarterly schedule.  No start makes all
    // eleven boundaries of an inclusive 5Y semiannual schedule weekdays, so
    // the two 2028 adjustments for the common 2024 start are locked below.
    let quarterly = generate_schedule(start, date(2026, 1, 1), Frequency::Quarterly);
    let quantlib_quarterly = [
        date(2024, 1, 1),
        date(2024, 4, 1),
        date(2024, 7, 1),
        date(2024, 10, 1),
        date(2025, 1, 1),
        date(2025, 4, 1),
        date(2025, 7, 1),
        date(2025, 10, 1),
        date(2026, 1, 1),
    ];
    assert_eq!(quarterly, quantlib_quarterly);

    let semiannual = generate_schedule(start, date(2029, 1, 1), Frequency::SemiAnnual);
    let quantlib_semiannual = [
        date(2024, 1, 1),
        date(2024, 7, 1),
        date(2025, 1, 1),
        date(2025, 7, 1),
        date(2026, 1, 1),
        date(2026, 7, 1),
        date(2027, 1, 1),
        date(2027, 7, 1),
        date(2028, 1, 3),
        date(2028, 7, 3),
        date(2029, 1, 1),
    ];
    assert_eq!(semiannual, quantlib_semiannual);
}

#[derive(Clone, Copy)]
struct BlackGridCase {
    label: &'static str,
    end_date: NaiveDate,
    frequency: Frequency,
    rate: f64,
    strike: f64,
    vol: f64,
    quantlib_cap: f64,
    quantlib_floor: f64,
    scipy_gap_cap: f64,
    scipy_gap_floor: f64,
}

fn black_grid() -> [BlackGridCase; 8] {
    [
        BlackGridCase {
            label: "2Y quarterly, r=3%, K=2%, vol=15%",
            end_date: date(2026, 1, 1),
            frequency: Frequency::Quarterly,
            rate: 0.03,
            strike: 0.02,
            vol: 0.15,
            quantlib_cap: 19_599.918_621_214_58,
            quantlib_floor: 16.493_431_247_276_41,
            scipy_gap_cap: 0.0,
            scipy_gap_floor: 4.547_473_508_864_641e-13,
        },
        BlackGridCase {
            label: "2Y quarterly, r=3%, K=3%, vol=25%",
            end_date: date(2026, 1, 1),
            frequency: Frequency::Quarterly,
            rate: 0.03,
            strike: 0.03,
            vol: 0.25,
            quantlib_cap: 4_964.860_184_421_881,
            quantlib_floor: 4_746.156_668_935_327,
            scipy_gap_cap: 9.094_947_017_729_282e-13,
            scipy_gap_floor: 9.094_947_017_729_282e-13,
        },
        BlackGridCase {
            label: "2Y quarterly, r=5%, K=4%, vol=15%",
            end_date: date(2026, 1, 1),
            frequency: Frequency::Quarterly,
            rate: 0.05,
            strike: 0.04,
            vol: 0.15,
            quantlib_cap: 19_892.936_623_858_866,
            quantlib_floor: 359.542_769_009_595_1,
            scipy_gap_cap: 7.275_957_614_183_426e-12,
            scipy_gap_floor: 8.299_139_153_677_97e-12,
        },
        BlackGridCase {
            label: "2Y quarterly, r=5%, K=2%, vol=25%",
            end_date: date(2026, 1, 1),
            frequency: Frequency::Quarterly,
            rate: 0.05,
            strike: 0.02,
            vol: 0.25,
            quantlib_cap: 57_412.942_565_792_73,
            quantlib_floor: 2.983_735_409_493_438,
            scipy_gap_cap: 1.455_191_522_836_685_2e-11,
            scipy_gap_floor: 7.092_104_681_305_5e-13,
        },
        BlackGridCase {
            label: "5Y semiannual, r=3%, K=4%, vol=15%",
            end_date: date(2029, 1, 1),
            frequency: Frequency::SemiAnnual,
            rate: 0.03,
            strike: 0.04,
            vol: 0.15,
            quantlib_cap: 2_055.381_010_162_325,
            quantlib_floor: 47_140.766_487_939_75,
            scipy_gap_cap: 4.547_473_508_864_641e-13,
            scipy_gap_floor: 7.275_957_614_183_426e-12,
        },
        BlackGridCase {
            label: "5Y semiannual, r=3%, K=2%, vol=25%",
            end_date: date(2029, 1, 1),
            frequency: Frequency::SemiAnnual,
            rate: 0.03,
            strike: 0.02,
            vol: 0.25,
            quantlib_cap: 50_206.911_635_270_17,
            quantlib_floor: 3_032.855_279_522_312_2,
            scipy_gap_cap: 7.275_957_614_183_426e-12,
            scipy_gap_floor: 2.273_736_754_432_320_6e-12,
        },
        BlackGridCase {
            label: "5Y semiannual, r=5%, K=3%, vol=15%",
            end_date: date(2029, 1, 1),
            frequency: Frequency::SemiAnnual,
            rate: 0.05,
            strike: 0.03,
            vol: 0.15,
            quantlib_cap: 90_515.035_114_107_68,
            quantlib_floor: 294.465_733_810_374_84,
            scipy_gap_cap: 3.783_497_959_375_381_5e-10,
            scipy_gap_floor: 9.094_947_017_729_282e-13,
        },
        BlackGridCase {
            label: "5Y semiannual, r=5%, K=4%, vol=25%",
            end_date: date(2029, 1, 1),
            frequency: Frequency::SemiAnnual,
            rate: 0.05,
            strike: 0.04,
            vol: 0.25,
            quantlib_cap: 57_232.576_195_010_54,
            quantlib_floor: 10_742.669_614_627_632,
            scipy_gap_cap: 3.346_940_502_524_376e-10,
            scipy_gap_floor: 6.184_563_972_055_912e-11,
        },
    ]
}

#[test]
fn caps_and_floors_match_quantlib_black_cap_floor_engine_grid() {
    for case in black_grid() {
        let curve = curve_on_schedule(case.rate, case.end_date, case.frequency);
        let cap = CapFloor {
            notional: NOTIONAL,
            strike: case.strike,
            start_date: start_date(),
            end_date: case.end_date,
            frequency: case.frequency,
            day_count: DayCountConvention::Act365Fixed,
            curve_day_count: DayCountConvention::Act365Fixed,
            is_cap: true,
        };
        let floor = CapFloor {
            is_cap: false,
            ..cap.clone()
        };

        let cap_price = cap.price(&curve, case.vol);
        let floor_price = floor.price(&curve, case.vol);
        assert_quantlib_black_reference(
            cap_price,
            case.quantlib_cap,
            case.scipy_gap_cap,
            &format!("{} cap", case.label),
        );
        assert_quantlib_black_reference(
            floor_price,
            case.quantlib_floor,
            case.scipy_gap_floor,
            &format!("{} floor", case.label),
        );

        // Supplemental identity: C - F is the underlying strike swap.  The
        // operation-count allowance covers each optionlet and both strip sums.
        let parity_lhs = cap_price - floor_price;
        let parity_rhs = cap.swap_npv(&curve);
        let periods = generate_schedule(start_date(), case.end_date, case.frequency).len() - 1;
        let operation_scale = cap_price
            .abs()
            .max(floor_price.abs())
            .max(parity_rhs.abs())
            .max(1.0);
        let parity_roundoff = 32.0 * periods as f64 * f64::EPSILON * operation_scale;
        assert!(
            (parity_lhs - parity_rhs).abs() <= parity_roundoff,
            "{} parity: lhs={parity_lhs:.17}, rhs={parity_rhs:.17}, budget={parity_roundoff:e}",
            case.label
        );
    }
}

#[test]
fn caplets_match_quantlib_black_cap_floor_engine_optionlets_price() {
    let end_date = date(2026, 1, 1);
    let frequency = Frequency::Quarterly;
    let curve = curve_on_schedule(0.03, end_date, frequency);
    let cap = CapFloor {
        notional: NOTIONAL,
        strike: 0.03,
        start_date: start_date(),
        end_date,
        frequency,
        day_count: DayCountConvention::Act365Fixed,
        curve_day_count: DayCountConvention::Act365Fixed,
        is_cap: true,
    };
    let schedule = generate_schedule(start_date(), end_date, frequency);

    // QuantLib `optionletsPrice()` for the same 2Y quarterly cap.  The first
    // caplet fixes on the evaluation date, so its zero-expiry value is intrinsic
    // in both engines; no historical fixing is required with zero fixing days.
    let quantlib_optionlets = [
        27.832_019_590_435_67,
        381.368_571_871_275_1,
        535.126_017_130_184_6,
        648.024_696_460_711_8,
        724.351_746_373_320_1,
        809.510_947_563_870_8,
        887.910_770_527_743_5,
        950.735_414_904_34,
    ];
    let scipy_gaps = [
        0.0,
        5.684_341_886_080_802e-14,
        0.0,
        1.182_343_112_304_806_7e-10,
        1.182_343_112_304_806_7e-10,
        0.0,
        1.136_868_377_216_160_3e-13,
        0.0,
    ];

    for (i, period) in schedule.windows(2).enumerate() {
        let actual = cap.optionlet_price(&curve, 0.25, period[0], period[1]);
        assert_quantlib_black_reference(
            actual,
            quantlib_optionlets[i],
            scipy_gaps[i],
            &format!("quarterly caplet {i}"),
        );
    }
}

#[test]
fn capfloor_implied_vol_round_trip_uses_quantlib_market_prices() {
    // These are package prices, rather than prices generated by the solver's
    // own objective function.  The bisection stopping rule is price-based, so
    // 2e-10 volatility is a conservative solver tolerance, not a price band.
    let cases = [
        (
            date(2026, 1, 1),
            Frequency::Quarterly,
            0.03,
            0.03,
            0.25,
            true,
            4_964.860_184_421_881,
        ),
        (
            date(2026, 1, 1),
            Frequency::Quarterly,
            0.05,
            0.04,
            0.15,
            false,
            359.542_769_009_595_1,
        ),
        (
            date(2029, 1, 1),
            Frequency::SemiAnnual,
            0.03,
            0.04,
            0.15,
            true,
            2_055.381_010_162_325,
        ),
        (
            date(2029, 1, 1),
            Frequency::SemiAnnual,
            0.05,
            0.04,
            0.25,
            false,
            10_742.669_614_627_632,
        ),
    ];

    for (end_date, frequency, rate, strike, vol, is_cap, quantlib_price) in cases {
        let curve = curve_on_schedule(rate, end_date, frequency);
        let instrument = CapFloor {
            notional: NOTIONAL,
            strike,
            start_date: start_date(),
            end_date,
            frequency,
            day_count: DayCountConvention::Act365Fixed,
            curve_day_count: DayCountConvention::Act365Fixed,
            is_cap,
        };
        let recovered = instrument.implied_vol(quantlib_price, &curve);
        assert!(
            (recovered - vol).abs() <= 2.0e-10,
            "is_cap={is_cap}, r={rate}, K={strike}: recovered={recovered:.17}, expected={vol:.17}"
        );
    }
}
