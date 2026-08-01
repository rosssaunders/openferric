//! Wrong-way-risk (WWR) CVA adjustments from simple regulatory and structural models.
//!
//! The module exposes three approaches that all return [`WwrResult`]
//! (`cva_independent`, `cva_wwr`, and `wwr_ratio`):
//! - [`AlphaWWR`]: Basel-style multiplier (`CVA_wwr = alpha * CVA_independent`).
//! - [`CopulaWWR`]: Gaussian-copula linkage between exposure and default drivers.
//! - [`HullWhiteWWR`]: stochastic-intensity model
//!   `lambda(t) = lambda0 * exp(beta * (S(t)/S(0) - 1))`.
//!
//! The copula and Hull-White variants are Monte Carlo estimators and include discounting and
//! LGD in per-path CVA contributions.
//!
//! Numerical notes: finite-path noise can materially move `wwr_ratio` near 1.0; increase
//! `num_paths` for stability. The copula implementation uses a lightweight internal RNG and
//! an approximation to `Phi(.)`, so this module is intended for transparent prototyping rather
//! than production-grade random number quality.
//!
//! References:
//! - Hull and White (2012), CVA with wrong-way risk modeling ideas.
//! - Basel Committee, SA-CVA/IMM discussions and supervisory alpha practice.

use crate::math::normal_inv_cdf;

/// Result of a wrong-way risk calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct WwrResult {
    /// CVA computed assuming independence between exposure and default.
    pub cva_independent: f64,
    /// CVA adjusted for wrong-way (or right-way) risk.
    pub cva_wwr: f64,
    /// Ratio cva_wwr / cva_independent (>1 for WWR, <1 for RWR).
    pub wwr_ratio: f64,
}

// ---------------------------------------------------------------------------
// 1. Alpha multiplier (Basel)
// ---------------------------------------------------------------------------

/// Basel regulatory alpha-multiplier approach to wrong-way risk.
///
/// CVA_wwr = α × CVA_independent, where α ≥ 1.4 by default.
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaWWR {
    /// Multiplier applied to independent CVA. Basel default is 1.4.
    pub alpha: f64,
}

impl Default for AlphaWWR {
    fn default() -> Self {
        Self { alpha: 1.4 }
    }
}

impl AlphaWWR {
    /// Create a new alpha-multiplier WWR with the given alpha.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 1.0, "alpha must be >= 1.0");
        Self { alpha }
    }

    /// Adjust an independently computed CVA by the alpha multiplier.
    pub fn adjust_cva(&self, independent_cva: f64) -> f64 {
        self.alpha * independent_cva
    }
}

// ---------------------------------------------------------------------------
// 2. Gaussian copula WWR
// ---------------------------------------------------------------------------

/// Gaussian copula wrong-way risk model.
///
/// Models joint distribution of exposure and default time using a Gaussian copula
/// with correlation ρ. Exposure paths are ranked by their (discounted, positive)
/// exposure metric and mapped to Gaussian quantiles, so a sampled path carries a
/// systematic factor `Z_exposure`. Default times are generated via correlated normals:
///   Z_default = ρ * Z_exposure + √(1-ρ²) * Z_idio
///
/// Larger `Z_default` maps to earlier default, so ρ > 0 makes high-exposure paths
/// default earlier (wrong-way risk) and ρ < 0 produces right-way risk.
pub struct CopulaWWR {
    /// Correlation between exposure driver and default driver (-1 to 1).
    pub correlation: f64,
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl CopulaWWR {
    pub fn new(correlation: f64, num_paths: usize, seed: u64) -> Self {
        assert!(
            (-1.0..=1.0).contains(&correlation),
            "correlation must be in [-1, 1]"
        );
        assert!(num_paths > 0, "num_paths must be positive");
        Self {
            correlation,
            num_paths,
            seed,
        }
    }

    /// Compute WWR-adjusted CVA.
    ///
    /// # Arguments
    /// * `exposure_paths` - Simulated exposure at each time point (one Vec per path).
    /// * `time_grid` - Time points (in years) corresponding to exposure columns.
    /// * `hazard_rate` - Flat hazard rate for default time simulation.
    /// * `recovery` - Recovery rate (LGD = 1 - recovery).
    /// * `risk_free_rate` - Flat risk-free rate for discounting.
    pub fn cva_with_wwr(
        &self,
        exposure_paths: &[Vec<f64>],
        time_grid: &[f64],
        hazard_rate: f64,
        recovery: f64,
        risk_free_rate: f64,
    ) -> WwrResult {
        let n_times = time_grid.len();
        assert!(
            !exposure_paths.is_empty(),
            "need at least one exposure path"
        );
        assert!(n_times > 0, "time_grid must be non-empty");
        for p in exposure_paths {
            assert_eq!(p.len(), n_times, "each path must match time_grid length");
        }

        let lgd = 1.0 - recovery;
        let n_source = exposure_paths.len();

        // Rank exposure paths by a discounted positive-exposure metric and map each rank
        // to a Gaussian quantile. The resulting latent variable is the systematic factor
        // shared between the exposure realization and the default driver.
        let metrics = exposure_paths
            .iter()
            .map(|path| {
                path.iter()
                    .zip(time_grid.iter())
                    .map(|(&e, &t)| (-risk_free_rate * t).exp() * e.max(0.0))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let mut order: Vec<usize> = (0..n_source).collect();
        order.sort_by(|&a, &b| metrics[a].total_cmp(&metrics[b]));
        let mut z_path = vec![0.0; n_source];
        for (rank, &idx) in order.iter().enumerate() {
            z_path[idx] = normal_inv_cdf((rank as f64 + 0.5) / n_source as f64);
        }

        // Simple LCG-based RNG for reproducibility without external deps
        let mut rng = LcgRng::new(self.seed);

        let mut cva_indep_sum = 0.0;
        let mut cva_wwr_sum = 0.0;

        for _ in 0..self.num_paths {
            // Pick a random exposure path
            let path_idx = (rng.next_u64() as usize) % n_source;
            let exposure = &exposure_paths[path_idx];

            // Systematic factor implied by the sampled exposure path's rank
            let z_exposure = z_path[path_idx];
            let z_idio = rng.next_normal();

            // Correlated default normal
            let z_default = self.correlation * z_exposure
                + (1.0 - self.correlation * self.correlation).sqrt() * z_idio;

            // Independent default normal (uncorrelated with the exposure factor)
            let z_default_indep = z_idio;

            // Map to default time via survival: τ = -ln(Φ(Z)) / λ
            let u_wwr = normal_cdf(z_default);
            let u_indep = normal_cdf(z_default_indep);

            let tau_wwr = if u_wwr <= 0.0 || u_wwr >= 1.0 {
                f64::INFINITY
            } else {
                -u_wwr.ln() / hazard_rate
            };

            let tau_indep = if u_indep <= 0.0 || u_indep >= 1.0 {
                f64::INFINITY
            } else {
                -u_indep.ln() / hazard_rate
            };

            // CVA contribution: LGD * DF(τ) * E(τ)
            cva_wwr_sum += cva_contribution(exposure, time_grid, tau_wwr, lgd, risk_free_rate);
            cva_indep_sum += cva_contribution(exposure, time_grid, tau_indep, lgd, risk_free_rate);
        }

        let cva_wwr = cva_wwr_sum / self.num_paths as f64;
        let cva_independent = cva_indep_sum / self.num_paths as f64;
        let wwr_ratio = if cva_independent.abs() < 1e-15 {
            1.0
        } else {
            cva_wwr / cva_independent
        };

        WwrResult {
            cva_independent,
            cva_wwr,
            wwr_ratio,
        }
    }
}

/// Compute CVA contribution for a single path given default time.
fn cva_contribution(
    exposure: &[f64],
    time_grid: &[f64],
    tau: f64,
    lgd: f64,
    risk_free_rate: f64,
) -> f64 {
    // Find the time bucket where default occurs
    let max_t = *time_grid.last().unwrap();
    if tau > max_t || tau <= 0.0 {
        return 0.0;
    }

    // Find the index: last time_grid point <= tau
    let idx = match time_grid.iter().position(|&t| t >= tau) {
        Some(i) => i,
        None => return 0.0,
    };

    let ee = exposure[idx].max(0.0);
    let df = (-risk_free_rate * tau).exp();

    lgd * df * ee
}

// ---------------------------------------------------------------------------
// 3. Hull-White credit-equity model
// ---------------------------------------------------------------------------

/// Hull-White wrong-way risk model where default intensity depends on asset value:
///   λ(t) = λ₀ × exp(β × (S(t)/S(0) - 1))
///
/// β < 0: default more likely when asset drops (wrong-way for protection seller).
/// β > 0: default more likely when asset rises (right-way for protection seller).
/// β = 0: independent.
pub struct HullWhiteWWR {
    /// Base (unconditional) hazard rate λ₀.
    pub base_hazard: f64,
    /// Sensitivity of hazard rate to asset moves.
    pub beta: f64,
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Random seed.
    pub seed: u64,
}

impl HullWhiteWWR {
    pub fn new(base_hazard: f64, beta: f64, num_paths: usize, seed: u64) -> Self {
        assert!(base_hazard > 0.0, "base_hazard must be positive");
        assert!(num_paths > 0, "num_paths must be positive");
        Self {
            base_hazard,
            beta,
            num_paths,
            seed,
        }
    }

    /// Compute CVA with stochastic hazard rate.
    ///
    /// # Arguments
    /// * `asset_paths` - Simulated asset price paths (ratio S(t)/S(0) at each time).
    /// * `exposure_paths` - Corresponding exposure at each time point.
    /// * `time_grid` - Time points in years.
    /// * `recovery` - Recovery rate.
    /// * `risk_free_rate` - Flat risk-free rate.
    pub fn cva_with_wwr(
        &self,
        asset_paths: &[Vec<f64>],
        exposure_paths: &[Vec<f64>],
        time_grid: &[f64],
        recovery: f64,
        risk_free_rate: f64,
    ) -> WwrResult {
        let n_paths = asset_paths.len();
        let n_times = time_grid.len();
        assert_eq!(n_paths, exposure_paths.len());
        assert!(n_times > 0);
        for i in 0..n_paths {
            assert_eq!(asset_paths[i].len(), n_times);
            assert_eq!(exposure_paths[i].len(), n_times);
        }

        let lgd = 1.0 - recovery;
        let mut cva_wwr_sum = 0.0;
        let mut cva_indep_sum = 0.0;

        for path_idx in 0..n_paths {
            let assets = &asset_paths[path_idx];
            let exposures = &exposure_paths[path_idx];

            // For each time step, compute conditional and unconditional default prob
            let mut surv_wwr = 1.0;
            let mut surv_indep = 1.0;

            for i in 0..n_times {
                let dt = if i == 0 {
                    time_grid[0]
                } else {
                    time_grid[i] - time_grid[i - 1]
                };

                // Stochastic hazard: λ(t) = λ₀ * exp(β * (S(t)/S(0) - 1))
                let lambda_wwr = self.base_hazard * (self.beta * (assets[i] - 1.0)).exp();
                let lambda_indep = self.base_hazard;

                let pd_wwr = 1.0 - (-lambda_wwr * dt).exp();
                let pd_indep = 1.0 - (-lambda_indep * dt).exp();

                let df = (-risk_free_rate * time_grid[i]).exp();
                let ee = exposures[i].max(0.0);

                cva_wwr_sum += lgd * df * ee * surv_wwr * pd_wwr;
                cva_indep_sum += lgd * df * ee * surv_indep * pd_indep;

                surv_wwr *= 1.0 - pd_wwr;
                surv_indep *= 1.0 - pd_indep;
            }
        }

        let cva_wwr = cva_wwr_sum / n_paths as f64;
        let cva_independent = cva_indep_sum / n_paths as f64;
        let wwr_ratio = if cva_independent.abs() < 1e-15 {
            1.0
        } else {
            cva_wwr / cva_independent
        };

        WwrResult {
            cva_independent,
            cva_wwr,
            wwr_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal RNG and math utilities (no external deps)
// ---------------------------------------------------------------------------

/// Simple LCG random number generator.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG constants from Knuth
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform for standard normal.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Approximate standard normal CDF using Abramowitz & Stegun.
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_grid_lock(actual: f64, expected: f64, label: &str) {
        let tolerance = 8.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: expected {expected:.17}, got {actual:.17}, tolerance {tolerance:.3e}"
        );
    }

    // Helper: generate a simple exposure profile (e.g., forward-starting swap)
    fn simple_exposure_profile(time_grid: &[f64], peak: f64) -> Vec<f64> {
        // Triangular profile peaking at midpoint
        let mid = time_grid.last().unwrap() / 2.0;
        time_grid
            .iter()
            .map(|&t| {
                if t <= mid {
                    peak * t / mid
                } else {
                    peak * (time_grid.last().unwrap() - t) / mid
                }
            })
            .collect()
    }

    #[test]
    fn test_alpha_default() {
        let wwr = AlphaWWR::default();
        assert!((wwr.alpha - 1.4).abs() < 1e-10);
        let cva = 100.0;
        assert!((wwr.adjust_cva(cva) - 140.0).abs() < 1e-10);
    }

    #[test]
    fn test_alpha_custom() {
        let wwr = AlphaWWR::new(1.6);
        assert!((wwr.adjust_cva(50.0) - 80.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_alpha_too_low() {
        AlphaWWR::new(0.9);
    }

    #[test]
    fn test_hull_white_beta_zero() {
        // β=0 → stochastic hazard equals base hazard → WWR ratio ≈ 1.0
        let time_grid: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
        let n_paths = 500;

        // Asset paths all at 1.0 (no moves) — should give ratio = 1 exactly
        let asset_paths: Vec<Vec<f64>> = (0..n_paths).map(|_| vec![1.0; time_grid.len()]).collect();
        let exposure = simple_exposure_profile(&time_grid, 1_000_000.0);
        let exposure_paths: Vec<Vec<f64>> = (0..n_paths).map(|_| exposure.clone()).collect();

        let hw = HullWhiteWWR::new(0.02, 0.0, n_paths, 42);
        let result = hw.cva_with_wwr(&asset_paths, &exposure_paths, &time_grid, 0.4, 0.03);

        assert!(
            (result.wwr_ratio - 1.0).abs() < 1e-10,
            "β=0 should give ratio=1, got {}",
            result.wwr_ratio
        );
    }

    fn hull_white_nonzero_beta_inputs() -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let time_grid = vec![0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
        let slopes = [0.03, 0.06, 0.09, 0.12];
        let peaks = [300_000.0, 500_000.0, 700_000.0, 900_000.0];
        let asset_paths = slopes
            .iter()
            .map(|slope| time_grid.iter().map(|t| 1.0 + slope * t).collect())
            .collect();
        let exposure_paths = peaks
            .iter()
            .map(|&peak| simple_exposure_profile(&time_grid, peak))
            .collect();
        (time_grid, asset_paths, exposure_paths)
    }

    #[test]
    fn test_hull_white_positive_beta_matches_deterministic_grid_reference() {
        let (time_grid, asset_paths, exposure_paths) = hull_white_nonzero_beta_inputs();
        let result = HullWhiteWWR::new(0.035, 1.75, 4, 42).cva_with_wwr(
            &asset_paths,
            &exposure_paths,
            &time_grid,
            0.4,
            0.027,
        );

        // Independently evaluated from the piecewise-constant survival recursion on the
        // stated six-point grid. This model averages supplied paths; it has no MC error.
        assert_grid_lock(
            result.cva_independent,
            15_057.753_122_747_8,
            "independent CVA",
        );
        assert_grid_lock(result.cva_wwr, 18_593.968_940_215_505, "beta=1.75 CVA");
        assert_grid_lock(result.wwr_ratio, 1.234_843_524_703_930_2, "beta=1.75 ratio");
        assert!(
            result.wwr_ratio > 1.0,
            "monotonic WWR check is supplemental"
        );
    }

    #[test]
    fn test_hull_white_negative_beta_matches_deterministic_grid_reference() {
        let (time_grid, asset_paths, exposure_paths) = hull_white_nonzero_beta_inputs();
        let result = HullWhiteWWR::new(0.035, -1.25, 4, 42).cva_with_wwr(
            &asset_paths,
            &exposure_paths,
            &time_grid,
            0.4,
            0.027,
        );

        assert_grid_lock(
            result.cva_independent,
            15_057.753_122_747_8,
            "independent CVA",
        );
        assert_grid_lock(result.cva_wwr, 13_030.377_411_837_926, "beta=-1.25 CVA");
        assert_grid_lock(
            result.wwr_ratio,
            0.865_360_011_259_109_4,
            "beta=-1.25 ratio",
        );
        assert!(
            result.wwr_ratio < 1.0,
            "monotonic RWR check is supplemental"
        );
    }

    // Helper: exposure paths whose overall magnitude varies across paths so the
    // exposure/default coupling has something to act on.
    fn varied_exposure_paths(time_grid: &[f64], n_paths: usize) -> Vec<Vec<f64>> {
        (0..n_paths)
            .map(|k| {
                let peak = 200_000.0 + 1_600_000.0 * k as f64 / (n_paths - 1) as f64;
                simple_exposure_profile(time_grid, peak)
            })
            .collect()
    }

    #[test]
    fn test_copula_zero_correlation() {
        // With ρ=0 the default driver collapses to the idiosyncratic normal, so the
        // WWR and independent CVAs must coincide path by path.
        let time_grid: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let exposure_paths = varied_exposure_paths(&time_grid, 100);

        let copula = CopulaWWR::new(0.0, 50_000, 12345);
        let result = copula.cva_with_wwr(&exposure_paths, &time_grid, 0.02, 0.4, 0.03);

        assert!(
            (result.wwr_ratio - 1.0).abs() < 1e-12,
            "zero correlation should give ratio = 1, got {}",
            result.wwr_ratio
        );
    }

    #[test]
    fn test_copula_positive_correlation_increases_cva() {
        let time_grid: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let exposure_paths = varied_exposure_paths(&time_grid, 200);

        let pos = CopulaWWR::new(0.7, 50_000, 777).cva_with_wwr(
            &exposure_paths,
            &time_grid,
            0.02,
            0.4,
            0.03,
        );

        // Exact regression lock for the stated LCG seed, 50,000 draws, exposure grid,
        // and A&S CDF implementation. It detects changes hidden by ratio-only checks.
        assert_grid_lock(
            pos.cva_independent,
            26_614.354_201_548_33,
            "seeded independent CVA",
        );
        assert_grid_lock(pos.cva_wwr, 41_219.470_022_413_7, "seeded rho=0.7 CVA");
        assert_grid_lock(
            pos.wwr_ratio,
            1.548_768_371_768_934_2,
            "seeded rho=0.7 ratio",
        );

        // Independent SciPy 1.17.1 adaptive quadrature target using scipy.stats.norm,
        // integrated separately over every default-time bucket and exposure rank.
        // Standard errors are sample estimates for this exact 50,000-path grid.
        const SCIPY_INDEPENDENT: f64 = 26_842.145_725_312_297;
        const SCIPY_RHO_POS: f64 = 41_592.802_133_458_96;
        const SE_INDEPENDENT: f64 = 478.376_765_311_065_1;
        const SE_RHO_POS: f64 = 683.787_780_840_040_7;
        const SCIPY_RATIO_POS: f64 = 1.549_533_429_968_555_3;
        const SE_RATIO_POS: f64 = 0.029_935_162_353_823_423;
        assert!(
            (pos.cva_independent - SCIPY_INDEPENDENT).abs() <= 4.0 * SE_INDEPENDENT,
            "independent CVA differs from quadrature by more than four standard errors"
        );
        assert!(
            (pos.cva_wwr - SCIPY_RHO_POS).abs() <= 4.0 * SE_RHO_POS,
            "rho=0.7 CVA differs from quadrature by more than four standard errors"
        );
        assert!(
            (pos.wwr_ratio - SCIPY_RATIO_POS).abs() <= 4.0 * SE_RATIO_POS,
            "rho=0.7 ratio differs from quadrature by more than four paired standard errors"
        );
        assert!(pos.wwr_ratio > 1.0, "monotonic WWR check is supplemental");
    }

    #[test]
    fn test_copula_negative_correlation_decreases_cva() {
        let time_grid: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let exposure_paths = varied_exposure_paths(&time_grid, 200);

        let neg = CopulaWWR::new(-0.7, 50_000, 777).cva_with_wwr(
            &exposure_paths,
            &time_grid,
            0.02,
            0.4,
            0.03,
        );

        assert_grid_lock(
            neg.cva_independent,
            26_614.354_201_548_33,
            "seeded independent CVA",
        );
        assert_grid_lock(neg.cva_wwr, 12_198.254_409_084_842, "seeded rho=-0.7 CVA");
        assert_grid_lock(
            neg.wwr_ratio,
            0.458_333_661_478_631_36,
            "seeded rho=-0.7 ratio",
        );

        const SCIPY_INDEPENDENT: f64 = 26_842.145_725_312_297;
        const SCIPY_RHO_NEG: f64 = 12_120.061_189_161_486;
        const SE_INDEPENDENT: f64 = 478.376_765_311_065_1;
        const SE_RHO_NEG: f64 = 228.633_090_666_537_1;
        const SCIPY_RATIO_NEG: f64 = 0.451_531_010_716_933_8;
        const SE_RATIO_NEG: f64 = 0.010_482_917_449_886_348;
        assert!(
            (neg.cva_independent - SCIPY_INDEPENDENT).abs() <= 4.0 * SE_INDEPENDENT,
            "independent CVA differs from quadrature by more than four standard errors"
        );
        assert!(
            (neg.cva_wwr - SCIPY_RHO_NEG).abs() <= 4.0 * SE_RHO_NEG,
            "rho=-0.7 CVA differs from quadrature by more than four standard errors"
        );
        assert!(
            (neg.wwr_ratio - SCIPY_RATIO_NEG).abs() <= 4.0 * SE_RATIO_NEG,
            "rho=-0.7 ratio differs from quadrature by more than four paired standard errors"
        );
        assert!(neg.wwr_ratio < 1.0, "monotonic RWR check is supplemental");
    }

    #[test]
    fn test_wwr_result_fields() {
        let r = WwrResult {
            cva_independent: 100.0,
            cva_wwr: 140.0,
            wwr_ratio: 1.4,
        };
        assert!((r.wwr_ratio - r.cva_wwr / r.cva_independent).abs() < 1e-10);
    }
}
