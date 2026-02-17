//! Fast approximations for the standard normal CDF and inverse CDF.

#[inline]
pub fn fast_norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Hart-style polynomial approximation for the standard normal CDF.
///
/// This form has max absolute error around 7.8e-8.
/// Uses `mul_add` (FMA) Horner evaluation for the polynomial chain.
#[inline]
pub fn hart_norm_cdf(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A1: f64 = 0.319_381_530;
    const A2: f64 = -0.356_563_782;
    const A3: f64 = 1.781_477_937;
    const A4: f64 = -1.821_255_978;
    const A5: f64 = 1.330_274_429;

    let z = x.abs();
    let t = 1.0 / P.mul_add(z, 1.0);
    let poly = A5.mul_add(t, A4)
        .mul_add(t, A3)
        .mul_add(t, A2)
        .mul_add(t, A1) * t;
    let cdf_pos = fast_norm_pdf(z).mul_add(-poly, 1.0);

    // Branch-free sign handling:
    // sign = 0 for x >= 0, sign = 1 for x < 0.
    let sign = (x.to_bits() >> 63) as f64;
    sign.mul_add(1.0 - 2.0 * cdf_pos, cdf_pos)
}

/// Beasley-Springer-Moro approximation for the inverse standard normal CDF.
#[inline]
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
        C[0].mul_add(q, C[1]).mul_add(q, C[2]).mul_add(q, C[3]).mul_add(q, C[4]).mul_add(q, C[5])
            / D[0].mul_add(q, D[1]).mul_add(q, D[2]).mul_add(q, D[3]).mul_add(q, 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        A[0].mul_add(r, A[1]).mul_add(r, A[2]).mul_add(r, A[3]).mul_add(r, A[4]).mul_add(r, A[5]) * q
            / B[0].mul_add(r, B[1]).mul_add(r, B[2]).mul_add(r, B[3]).mul_add(r, B[4]).mul_add(r, 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -C[0].mul_add(q, C[1]).mul_add(q, C[2]).mul_add(q, C[3]).mul_add(q, C[4]).mul_add(q, C[5])
            / D[0].mul_add(q, D[1]).mul_add(q, D[2]).mul_add(q, D[3]).mul_add(q, 1.0)
    }
}

#[inline]
pub fn fast_norm_cdf(x: f64) -> f64 {
    hart_norm_cdf(x)
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
        (-5.0, 2.8665157187919391e-7),
        (-4.0, 3.1671241833119979e-5),
        (-3.0, 0.0013498980316300946),
        (-2.0, 0.02275013194817921),
        (-1.0, 0.15865525393145702),
        (-0.5, 0.30853753872598690),
        (0.0, 0.5),
        (0.5, 0.69146246127401310),
        (1.0, 0.84134474606854298),
        (2.0, 0.97724986805182079),
        (3.0, 0.99865010196837),
        (4.0, 0.99996832875816688),
        (5.0, 0.99999971334842808),
        (8.0, 1.0 - 6.22096057427178e-16),
    ];

    #[test]
    fn fast_cdf_matches_reference_table() {
        for &(x, expected) in CDF_REFERENCE {
            let got = hart_norm_cdf(x);
            let err = (got - expected).abs();
            assert!(
                err < 1.0e-7,
                "x={x} expected={expected} got={got} err={err}"
            );
        }
    }

    #[test]
    fn cdf_symmetry() {
        for i in 0..=80 {
            let x = i as f64 / 10.0;
            let sum = hart_norm_cdf(x) + hart_norm_cdf(-x);
            assert!((sum - 1.0).abs() < 1e-12, "x={x} sum={sum}");
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
    fn inv_cdf_known_values() {
        // Phi^{-1}(0.5) = 0
        assert!(beasley_springer_moro_inv_cdf(0.5).abs() < 1e-10);
        // Phi^{-1}(0.841344746...) ≈ 1.0
        let x = beasley_springer_moro_inv_cdf(0.8413447460685430);
        assert!((x - 1.0).abs() < 1e-6, "got {x}");
        // Phi^{-1}(0.977249868...) ≈ 2.0
        let x = beasley_springer_moro_inv_cdf(0.9772498680518208);
        assert!((x - 2.0).abs() < 1e-6, "got {x}");
    }
}
