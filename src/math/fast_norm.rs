//! Module `math::fast_norm`.
//!
//! Implements fast norm workflows with concrete routines such as `fast_norm_pdf`, `accurate_norm_cdf`, `erfc_cody`, `hart_norm_cdf`, `beasley_springer_moro_inv_cdf`, `fast_norm_cdf`.
//!
//! References: Abramowitz and Stegun (1964), Cody (1969), Moro (1995), West (2005), Press et al. (2007), approximation formulas around Eq. (7.1.26).
//!
//! Primary API surface: free functions `fast_norm_pdf`, `accurate_norm_cdf`, `erfc_cody`, `hart_norm_cdf`, `beasley_springer_moro_inv_cdf`, `fast_norm_cdf`.
//!
//! Numerical considerations: approximation regions, branch choices, and machine-precision cancellation near boundaries should be validated with high-precision references.
//!
//! When to use: use these low-level routines in performance-sensitive calibration/pricing loops; use higher-level modules when model semantics matter more than raw numerics.

#[inline(always)]
pub fn fast_norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    // Use mul_add for the exponent: -0.5 * x * x → (-0.5_f64).mul_add(x * x, 0.0)
    // which compiles to a single FMA instruction on supported hardware.
    INV_SQRT_2PI * ((-0.5_f64) * x * x).exp()
}

/// Complementary error function `erfc(x)`, Cody (1969) rational approximation.
///
/// Max relative error is a few ULP (~1e-15) over the full double range,
/// including the deep tail down to the underflow threshold (`x ~ 26.5`).
pub fn erfc_cody(x: f64) -> f64 {
    const A: [f64; 5] = [
        3.161_123_743_870_565_6,
        1.138_641_541_510_501_6e2,
        3.774_852_376_853_02e2,
        3.209_377_589_138_469_5e3,
        1.857_777_061_846_031_5e-1,
    ];
    const B: [f64; 4] = [
        2.360_129_095_234_412_1e1,
        2.440_246_379_344_441_7e2,
        1.282_616_526_077_372_3e3,
        2.844_236_833_439_171e3,
    ];
    const C: [f64; 9] = [
        5.641_884_969_886_701e-1,
        8.883_149_794_388_376,
        6.611_919_063_714_163e1,
        2.986_351_381_974_001e2,
        8.819_522_212_417_69e2,
        1.712_047_612_634_070_6e3,
        2.051_078_377_826_071_5e3,
        1.230_339_354_797_997_2e3,
        2.153_115_354_744_038_5e-8,
    ];
    const D: [f64; 8] = [
        1.574_492_611_070_983_5e1,
        1.176_939_508_913_125e2,
        5.371_811_018_620_099e2,
        1.621_389_574_566_690_2e3,
        3.290_799_235_733_459_7e3,
        4.362_619_090_143_247e3,
        3.439_367_674_143_721_6e3,
        1.230_339_354_803_749_4e3,
    ];
    const P: [f64; 6] = [
        3.053_266_349_612_323_4e-1,
        3.603_448_999_498_044_4e-1,
        1.257_817_261_112_292_5e-1,
        1.608_378_514_874_228e-2,
        6.587_491_615_298_378e-4,
        1.631_538_713_730_209_8e-2,
    ];
    const Q: [f64; 5] = [
        2.568_520_192_289_822,
        1.872_952_849_923_460_4,
        5.279_051_029_514_284e-1,
        6.051_834_131_244_132e-2,
        2.335_204_976_268_691_8e-3,
    ];
    /// 1/sqrt(pi).
    const SQRPI: f64 = 5.641_895_835_477_563e-1;
    const THRESH: f64 = 0.46875;
    /// erfc underflows to zero beyond this point.
    const XBIG: f64 = 26.543;

    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();

    if y <= THRESH {
        // Central region via erf(x) = x * R(x^2).
        let zsq = y * y;
        let mut xnum = A[4] * zsq;
        let mut xden = zsq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * zsq;
            xden = (xden + B[i]) * zsq;
        }
        return 1.0 - x * (xnum + A[3]) / (xden + B[3]);
    }

    let result = if y <= 4.0 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        (xnum + C[7]) / (xden + D[7])
    } else if y < XBIG {
        let zsq = 1.0 / (y * y);
        let mut xnum = P[5] * zsq;
        let mut xden = zsq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * zsq;
            xden = (xden + Q[i]) * zsq;
        }
        let r = zsq * (xnum + P[4]) / (xden + Q[4]);
        (SQRPI - r) / y
    } else {
        0.0
    };

    // exp(-y^2) computed as exp(-ysq^2) * exp(-del) with ysq = trunc(16y)/16,
    // so the dominant factor has an exactly representable argument. This is
    // what preserves relative accuracy deep in the tail.
    let result = if result > 0.0 {
        let ysq = (y * 16.0).trunc() / 16.0;
        let del = (y - ysq) * (y + ysq);
        (-ysq * ysq).exp() * (-del).exp() * result
    } else {
        result
    };

    if x < 0.0 { 2.0 - result } else { result }
}

/// Tail-accurate standard normal CDF via [`erfc_cody`]:
/// `Phi(x) = erfc(-x / sqrt(2)) / 2`.
///
/// Relative accuracy is close to machine precision over the whole range,
/// including the deep lower tail (e.g. `Phi(-20) ~ 2.75e-89`), unlike the
/// absolute-error A&S 26.2.17 approximation kept in [`fast_norm_cdf`].
/// This backs the default `math::functions::normal_cdf`; use it whenever
/// tail probabilities matter (copulas, credit, quantiles).
#[inline]
pub fn accurate_norm_cdf(x: f64) -> f64 {
    0.5 * erfc_cody(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// Hart (1968) standard normal CDF, as published in West (2005),
/// *Better approximations to cumulative normal functions*.
///
/// Tail-relative accuracy is ~3e-9 (verified against mpmath references) —
/// far better than the A&S approximation in [`fast_norm_cdf`] but short of
/// double precision; prefer [`accurate_norm_cdf`] when full tail accuracy is
/// required.
#[inline]
pub fn hart_norm_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    let z = x.abs();
    let cum = if z > 37.0 {
        0.0
    } else {
        let e = (-0.5 * z * z).exp();
        if z < std::f64::consts::SQRT_2 * 5.0 {
            // Rational approximation for the central region (|x| < 7.07...).
            let num = 3.526_249_659_989_11e-2_f64
                .mul_add(z, 0.700_383_064_443_688)
                .mul_add(z, 6.373_962_203_531_65)
                .mul_add(z, 33.912_866_078_383)
                .mul_add(z, 112.079_291_497_871)
                .mul_add(z, 221.213_596_169_931)
                .mul_add(z, 220.206_867_912_376);
            let den = 8.838_834_764_831_84e-2_f64
                .mul_add(z, 1.755_667_163_182_64)
                .mul_add(z, 16.064_177_579_207)
                .mul_add(z, 86.780_732_202_946_1)
                .mul_add(z, 296.564_248_779_674)
                .mul_add(z, 637.333_633_378_831)
                .mul_add(z, 793.826_512_519_948)
                .mul_add(z, 440.413_735_824_752);
            e * num / den
        } else {
            // Continued-fraction expansion for the far tail.
            const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
            let b = z + 0.65;
            let b = z + 4.0 / b;
            let b = z + 3.0 / b;
            let b = z + 2.0 / b;
            let b = z + 1.0 / b;
            e / (b * SQRT_2PI)
        }
    };

    if x <= 0.0 { cum } else { 1.0 - cum }
}

/// Fast A&S 26.2.17 5-term polynomial approximation of the normal CDF.
///
/// Max ABSOLUTE error around 7.5e-8, which translates into large RELATIVE
/// error in the tails (~25% at x = -5, ~100% beyond -5.5). Only use this in
/// hot paths that are insensitive to tail probabilities; the default
/// `normal_cdf` routes to the tail-accurate [`accurate_norm_cdf`] (Cody).
#[inline(always)]
fn abramowitz_stegun_norm_cdf(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A1: f64 = 0.319_381_530;
    const A2: f64 = -0.356_563_782;
    const A3: f64 = 1.781_477_937;
    const A4: f64 = -1.821_255_978;
    const A5: f64 = 1.330_274_429;

    let z = x.abs();
    let t = 1.0 / P.mul_add(z, 1.0);
    let poly = A5
        .mul_add(t, A4)
        .mul_add(t, A3)
        .mul_add(t, A2)
        .mul_add(t, A1)
        * t;
    let cdf_pos = fast_norm_pdf(z).mul_add(-poly, 1.0);

    // Branch-free sign handling:
    // sign = 0 for x >= 0, sign = 1 for x < 0.
    let sign = (x.to_bits() >> 63) as f64;
    sign.mul_add(1.0 - 2.0 * cdf_pos, cdf_pos)
}

/// Beasley-Springer-Moro approximation for the inverse standard normal CDF.
#[inline(always)]
pub fn beasley_springer_moro_inv_cdf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Acklam's rational approximation, keeping this API name for compatibility.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        C[0].mul_add(q, C[1])
            .mul_add(q, C[2])
            .mul_add(q, C[3])
            .mul_add(q, C[4])
            .mul_add(q, C[5])
            / D[0]
                .mul_add(q, D[1])
                .mul_add(q, D[2])
                .mul_add(q, D[3])
                .mul_add(q, 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        A[0].mul_add(r, A[1])
            .mul_add(r, A[2])
            .mul_add(r, A[3])
            .mul_add(r, A[4])
            .mul_add(r, A[5])
            * q
            / B[0]
                .mul_add(r, B[1])
                .mul_add(r, B[2])
                .mul_add(r, B[3])
                .mul_add(r, B[4])
                .mul_add(r, 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -C[0]
            .mul_add(q, C[1])
            .mul_add(q, C[2])
            .mul_add(q, C[3])
            .mul_add(q, C[4])
            .mul_add(q, C[5])
            / D[0]
                .mul_add(q, D[1])
                .mul_add(q, D[2])
                .mul_add(q, D[3])
                .mul_add(q, 1.0)
    }
}

/// Fast normal CDF approximation (Abramowitz & Stegun 26.2.17, ~7.5e-8
/// absolute error).
///
/// Not tail-accurate in a relative sense; prefer [`accurate_norm_cdf`]
/// (Cody, the default behind `math::functions::normal_cdf`) or the cheaper
/// [`hart_norm_cdf`] unless profiling shows this approximation is needed and
/// tails are irrelevant.
#[inline]
pub fn fast_norm_cdf(x: f64) -> f64 {
    abramowitz_stegun_norm_cdf(x)
}

#[inline]
pub fn fast_norm_inv_cdf(p: f64) -> f64 {
    beasley_springer_moro_inv_cdf(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from NIST / Abramowitz & Stegun Table 26.1
    const CDF_REFERENCE: &[(f64, f64)] = &[
        (-8.0, 6.22096057427178e-16),
        (-5.0, 2.866_515_718_791_939e-7),
        (-4.0, 3.167_124_183_311_998e-5),
        (-3.0, 0.0013498980316300946),
        (-2.0, 0.02275013194817921),
        (-1.0, 0.15865525393145702),
        (-0.5, 0.308_537_538_725_986_9),
        (0.0, 0.5),
        (0.5, 0.691_462_461_274_013_1),
        (1.0, 0.841_344_746_068_542_9),
        (2.0, 0.977_249_868_051_820_8),
        (3.0, 0.99865010196837),
        (4.0, 0.999_968_328_758_166_9),
        (5.0, 0.999_999_713_348_428_1),
        (8.0, 1.0 - 6.22096057427178e-16),
    ];

    // High-precision lower-tail reference values (mpmath, 40 digits, rounded
    // to nearest f64).
    const TAIL_REFERENCE: &[(f64, f64)] = &[
        (-5.0, 2.866_515_718_791_939e-7),
        (-6.0, 9.865_876_450_376_98e-10),
        (-7.0, 1.279_812_543_885_835e-12),
        (-8.0, 6.220_960_574_271_784e-16),
        (-10.0, 7.619_853_024_160_525e-24),
        (-15.0, 3.670_966_199_312_751e-51),
        (-20.0, 2.753_624_118_606_233_7e-89),
    ];

    #[test]
    fn fast_cdf_matches_reference_table() {
        for &(x, expected) in CDF_REFERENCE {
            for got in [accurate_norm_cdf(x), hart_norm_cdf(x), fast_norm_cdf(x)] {
                let err = (got - expected).abs();
                assert!(
                    err < 1.0e-7,
                    "x={x} expected={expected} got={got} err={err}"
                );
            }
        }
    }

    #[test]
    fn accurate_cdf_is_relative_accurate_in_the_tails() {
        for &(x, expected) in TAIL_REFERENCE {
            let got = accurate_norm_cdf(x);
            let rel = ((got - expected) / expected).abs();
            let tol = if x >= -8.0 { 1.0e-12 } else { 1.0e-10 };
            assert!(rel < tol, "x={x} expected={expected} got={got} rel={rel}");

            // Upper tail mirrors via 1 - Phi(-x) up to f64 representation of 1.
            let upper = accurate_norm_cdf(-x);
            assert!(
                (upper - (1.0 - expected)).abs() <= f64::EPSILON,
                "x={} upper={} expected={}",
                -x,
                upper,
                1.0 - expected
            );
        }
        // Hard cut-off / special-value region.
        assert_eq!(accurate_norm_cdf(-40.0), 0.0);
        assert_eq!(accurate_norm_cdf(40.0), 1.0);
        assert_eq!(accurate_norm_cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(accurate_norm_cdf(f64::INFINITY), 1.0);
        assert!(accurate_norm_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn hart_cdf_tail_relative_accuracy_is_at_least_1e8() {
        // The Hart/West routine is good to ~3e-9 relative in the tails:
        // orders of magnitude better than A&S, but not double precision.
        for &(x, expected) in TAIL_REFERENCE {
            let got = hart_norm_cdf(x);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1.0e-8,
                "x={x} expected={expected} got={got} rel={rel}"
            );
        }
        assert_eq!(hart_norm_cdf(-38.0), 0.0);
        assert_eq!(hart_norm_cdf(38.0), 1.0);
        assert!(hart_norm_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn cdf_symmetry() {
        for i in 0..=80 {
            let x = i as f64 / 10.0;
            for f in [
                accurate_norm_cdf as fn(f64) -> f64,
                hart_norm_cdf,
                fast_norm_cdf,
            ] {
                let sum = f(x) + f(-x);
                assert!((sum - 1.0).abs() < 1e-12, "x={x} sum={sum}");
            }
        }
    }

    #[test]
    fn inv_cdf_round_trips_cdf() {
        for i in 1..=999 {
            let p = i as f64 / 1000.0;
            let x = beasley_springer_moro_inv_cdf(p);
            let p_back = hart_norm_cdf(x);
            assert!(
                (p_back - p).abs() < 2e-7,
                "p={p} x={x} p_back={p_back} err={}",
                (p_back - p).abs()
            );
        }
    }

    #[test]
    fn inv_cdf_handles_extreme_tail_probabilities() {
        // The copula uniform clamp is [1e-300, 1 - 1e-16]; the inverse CDF
        // must stay finite there.
        let lo = beasley_springer_moro_inv_cdf(1.0e-300);
        assert!(lo.is_finite() && lo < -37.0 && lo > -38.0, "lo={lo}");
        let hi = beasley_springer_moro_inv_cdf(1.0 - 1.0e-16);
        assert!(hi.is_finite() && hi > 8.0, "hi={hi}");
        assert_eq!(beasley_springer_moro_inv_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(beasley_springer_moro_inv_cdf(1.0), f64::INFINITY);
        assert!(beasley_springer_moro_inv_cdf(-0.1).is_nan());
        assert!(beasley_springer_moro_inv_cdf(1.1).is_nan());
        assert!(beasley_springer_moro_inv_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn inv_cdf_known_values() {
        // Phi^{-1}(0.5) = 0
        assert!(beasley_springer_moro_inv_cdf(0.5).abs() < 1e-10);
        // Phi^{-1}(0.841344746...) ≈ 1.0
        let x = beasley_springer_moro_inv_cdf(0.841_344_746_068_543);
        assert!((x - 1.0).abs() < 1e-6, "got {x}");
        // Phi^{-1}(0.977249868...) ≈ 2.0
        let x = beasley_springer_moro_inv_cdf(0.9772498680518208);
        assert!((x - 2.0).abs() < 1e-6, "got {x}");
    }
}
