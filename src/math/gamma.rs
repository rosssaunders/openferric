// Lanczos approximation for the gamma function.
// Ported from statrs (MIT license), based on:
// "An Analysis of the Lanczos Gamma Approximation", Glendon Ralph Pugh, 2004 p. 116

const GAMMA_R: f64 = 10.900511;

const GAMMA_DK: &[f64] = &[
    2.485_740_891_387_535_5e-5,
    1.051_423_785_817_219_7,
    -3.456_870_972_220_162_4,
    4.512_277_094_668_948,
    -2.982_852_253_235_766_6,
    1.056_397_115_771_267,
    -1.954_287_731_916_458_7e-1,
    1.709_705_434_044_412e-2,
    -5.719_261_174_043_057e-4,
    4.633_994_733_599_057e-6,
    -2.719_949_084_886_077_2e-9,
];

const TWO_SQRT_E_OVER_PI: f64 = 1.860_382_734_205_265_7;

/// Computes the gamma function with an accuracy of 16 floating point digits.
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, (i, &dk)| s + dk / (i as f64 - x));

        std::f64::consts::PI
            / ((std::f64::consts::PI * x).sin()
                * s
                * TWO_SQRT_E_OVER_PI
                * ((0.5 - x + GAMMA_R) / std::f64::consts::E).powf(0.5 - x))
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, (i, &dk)| s + dk / (x + i as f64 - 1.0));

        s * TWO_SQRT_E_OVER_PI * ((x - 0.5 + GAMMA_R) / std::f64::consts::E).powf(x - 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_known_values() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-15);
        assert!((gamma(2.0) - 1.0).abs() < 1e-15);
        assert!((gamma(3.0) - 2.0).abs() < 1e-14);
        assert!((gamma(4.0) - 6.0).abs() < 1e-13);
        assert!((gamma(5.0) - 24.0).abs() < 1e-12);
        // Gamma(0.5) = sqrt(pi)
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-14);
        // Gamma(1.5) = sqrt(pi)/2
        assert!((gamma(1.5) - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-14);
    }

    #[test]
    fn gamma_negative_arguments() {
        // Gamma(-0.5) = -2*sqrt(pi)
        assert!((gamma(-0.5) - (-2.0 * std::f64::consts::PI.sqrt())).abs() < 1e-13);
        // Gamma(-1.5) = 4*sqrt(pi)/3
        assert!((gamma(-1.5) - (4.0 * std::f64::consts::PI.sqrt() / 3.0)).abs() < 1e-13);
    }
}
