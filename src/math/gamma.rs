// Lanczos approximation for the gamma function.
// Ported from statrs (MIT license), based on:
// "An Analysis of the Lanczos Gamma Approximation", Glendon Ralph Pugh, 2004 p. 116

const GAMMA_R: f64 = 10.900511;

const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];

const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;

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
