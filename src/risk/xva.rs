use crate::credit::SurvivalCurve;
use crate::rates::YieldCurve;

/// Basic CVA/DVA calculator with deterministic curves and exposure profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct XvaCalculator {
    pub discount_curve: YieldCurve,
    pub counterparty_survival: SurvivalCurve,
    pub own_survival: SurvivalCurve,
    pub lgd: f64,
    pub lgd_own: f64,
}

impl XvaCalculator {
    pub fn new(
        discount_curve: YieldCurve,
        counterparty_survival: SurvivalCurve,
        own_survival: SurvivalCurve,
        lgd: f64,
        lgd_own: f64,
    ) -> Self {
        assert!(lgd.is_finite(), "lgd must be finite");
        assert!(lgd_own.is_finite(), "lgd_own must be finite");
        Self {
            discount_curve,
            counterparty_survival,
            own_survival,
            lgd: lgd.clamp(0.0, 1.0),
            lgd_own: lgd_own.clamp(0.0, 1.0),
        }
    }

    /// Expected Exposure (EE) profile from simulated mark-to-market paths.
    ///
    /// Input shape is `[path_index][time_index]`.
    pub fn expected_exposure_profile(exposure_paths: &[Vec<f64>]) -> Vec<f64> {
        average_profile(exposure_paths, |x| x.max(0.0))
    }

    /// Negative Expected Exposure (NEE) profile from simulated mark-to-market paths.
    ///
    /// Input shape is `[path_index][time_index]`.
    pub fn negative_expected_exposure_profile(exposure_paths: &[Vec<f64>]) -> Vec<f64> {
        average_profile(exposure_paths, |x| (-x).max(0.0))
    }

    /// CVA = -LGD * integral EE(t) * dPD_counterparty(t), with discounting.
    pub fn cva_from_expected_exposure(&self, times: &[f64], ee_profile: &[f64]) -> f64 {
        validate_profile_input(times, ee_profile);

        let mut prev_t = 0.0;
        let mut integral = 0.0;
        for (&t, &ee) in times.iter().zip(ee_profile.iter()) {
            if t <= prev_t {
                continue;
            }
            let dpd = self.counterparty_survival.default_prob(prev_t, t);
            let df = self.discount_curve.discount_factor(t);
            integral += df * ee.max(0.0) * dpd;
            prev_t = t;
        }

        -self.lgd * integral
    }

    /// DVA = LGD_own * integral NEE(t) * dPD_own(t), with discounting.
    pub fn dva_from_negative_expected_exposure(&self, times: &[f64], nee_profile: &[f64]) -> f64 {
        validate_profile_input(times, nee_profile);

        let mut prev_t = 0.0;
        let mut integral = 0.0;
        for (&t, &nee) in times.iter().zip(nee_profile.iter()) {
            if t <= prev_t {
                continue;
            }
            let dpd = self.own_survival.default_prob(prev_t, t);
            let df = self.discount_curve.discount_factor(t);
            integral += df * nee.max(0.0) * dpd;
            prev_t = t;
        }

        self.lgd_own * integral
    }

    /// Convenience method: computes CVA directly from mark-to-market simulation paths.
    pub fn cva_from_paths(&self, times: &[f64], exposure_paths: &[Vec<f64>]) -> f64 {
        let ee = Self::expected_exposure_profile(exposure_paths);
        self.cva_from_expected_exposure(times, &ee)
    }

    /// Convenience method: computes DVA directly from mark-to-market simulation paths.
    pub fn dva_from_paths(&self, times: &[f64], exposure_paths: &[Vec<f64>]) -> f64 {
        let nee = Self::negative_expected_exposure_profile(exposure_paths);
        self.dva_from_negative_expected_exposure(times, &nee)
    }
}

fn validate_profile_input(times: &[f64], profile: &[f64]) {
    assert_eq!(
        times.len(),
        profile.len(),
        "times and profile must have same length"
    );
    let mut prev = 0.0;
    for &t in times {
        assert!(t >= 0.0, "times must be non-negative");
        assert!(t >= prev, "times must be non-decreasing");
        prev = t;
    }
}

fn average_profile<F>(paths: &[Vec<f64>], transform: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    if paths.is_empty() {
        return Vec::new();
    }

    let n_steps = paths[0].len();
    if n_steps == 0 {
        return Vec::new();
    }

    for path in paths {
        assert_eq!(
            path.len(),
            n_steps,
            "all exposure paths must have same time grid length"
        );
    }

    let mut profile = vec![0.0; n_steps];
    for path in paths {
        for (idx, &x) in path.iter().enumerate() {
            profile[idx] += transform(x);
        }
    }

    let n_paths = paths.len() as f64;
    for value in &mut profile {
        *value /= n_paths;
    }
    profile
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn expected_exposure_profile_from_paths_is_correct() {
        let paths = vec![vec![10.0, -5.0, 20.0], vec![-10.0, 5.0, 40.0]];
        let ee = XvaCalculator::expected_exposure_profile(&paths);
        let nee = XvaCalculator::negative_expected_exposure_profile(&paths);

        assert_relative_eq!(ee[0], 5.0, epsilon = 1.0e-12);
        assert_relative_eq!(ee[1], 2.5, epsilon = 1.0e-12);
        assert_relative_eq!(ee[2], 30.0, epsilon = 1.0e-12);

        assert_relative_eq!(nee[0], 5.0, epsilon = 1.0e-12);
        assert_relative_eq!(nee[1], 2.5, epsilon = 1.0e-12);
        assert_relative_eq!(nee[2], 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn simple_irs_cva_matches_flat_hazard_reference() {
        let notional = 1_000_000.0;
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let discount_curve = YieldCurve::new(times.iter().map(|&t| (t, 1.0)).collect());

        let spread = 0.01;
        let recovery = 0.4;
        let lgd = 1.0 - recovery;
        let hazard = spread / lgd;

        let hazards = vec![hazard; times.len()];
        let counterparty_survival = SurvivalCurve::from_piecewise_hazard(&times, &hazards);
        let own_survival = SurvivalCurve::from_piecewise_hazard(&times, &hazards);

        let calc = XvaCalculator::new(
            discount_curve,
            counterparty_survival,
            own_survival,
            lgd,
            lgd,
        );

        let ee_profile = vec![0.05 * notional; times.len()];
        let cva = calc.cva_from_expected_exposure(&times, &ee_profile);

        let expected = -lgd * 0.05 * (1.0 - (-hazard * 5.0).exp()) * notional;
        assert_relative_eq!(cva, expected, max_relative = 0.03);
    }
}
