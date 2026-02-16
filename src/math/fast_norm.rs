//! Fast approximations for the standard normal CDF and inverse CDF.

#[inline]
pub fn fast_norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Hart-style polynomial approximation for the standard normal CDF.
///
/// This form has max absolute error around 7.8e-8.
#[inline]
pub fn hart_norm_cdf(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A1: f64 = 0.319_381_530;
    const A2: f64 = -0.356_563_782;
    const A3: f64 = 1.781_477_937;
    const A4: f64 = -1.821_255_978;
    const A5: f64 = 1.330_274_429;

    let z = x.abs();
    let t = 1.0 / (1.0 + P * z);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let cdf_pos = 1.0 - fast_norm_pdf(z) * poly;

    // Branch-free sign handling:
    // sign = 0 for x >= 0, sign = 1 for x < 0.
    let sign = (x.to_bits() >> 63) as f64;
    cdf_pos + sign * (1.0 - 2.0 * cdf_pos)
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
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
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
    use statrs::distribution::{ContinuousCDF, Normal};

    #[test]
    fn fast_cdf_tracks_statrs() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in -80..=80 {
            let x = i as f64 / 10.0;
            let err = (hart_norm_cdf(x) - normal.cdf(x)).abs();
            assert!(err < 1.0e-7, "x={x} err={err}");
        }
    }

    #[test]
    fn bsm_inverse_tracks_statrs() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 1..=999 {
            let p = i as f64 / 1000.0;
            let err = (beasley_springer_moro_inv_cdf(p) - normal.inverse_cdf(p)).abs();
            assert!(err < 1.0e-6, "p={p} err={err}");
        }
    }
}
