//! Monte Carlo liquidation analysis for isolated Boros-style funding-rate positions.
//!
//! The simulator uses mean-reverting short-rate dynamics from [`crate::models::short_rate`] and
//! evaluates liquidation against the isolated-margin rules in [`crate::risk::margin`].

use crate::math::fast_rng::{FastRng, FastRngKind, resolve_stream_seed, sample_standard_normal};
use crate::models::short_rate::{CIR, Vasicek};

use super::margin::{MarginCalculator, MarginParams};

const EPSILON: f64 = 1.0e-12;

/// Signed Boros-style position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidationPosition {
    /// Signed notional. Positive positions lose when funding rises.
    pub size: f64,
    pub entry_rate: f64,
    pub collateral: f64,
    pub margin_params: MarginParams,
}

/// Supported mean-reverting funding-rate models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FundingRateModel {
    Vasicek(Vasicek),
    CIR(CIR),
}

/// Stress scenarios for liquidation analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StressScenario {
    Baseline,
    LiquidationCascade { vol_multiplier: f64 },
    MeanShift { shift: f64 },
}

impl StressScenario {
    /// Common liquidation-cascade shocks.
    pub fn cascade_suite() -> Vec<Self> {
        vec![
            Self::LiquidationCascade {
                vol_multiplier: 3.0,
            },
            Self::LiquidationCascade {
                vol_multiplier: 5.0,
            },
            Self::LiquidationCascade {
                vol_multiplier: 10.0,
            },
        ]
    }

    /// Symmetric sustained-funding shocks around the baseline mean level.
    pub fn mean_shift_suite(shift: f64) -> Vec<Self> {
        assert!(
            shift.is_finite(),
            "shift must be finite for mean-shift scenarios"
        );
        vec![Self::MeanShift { shift }, Self::MeanShift { shift: -shift }]
    }
}

/// Liquidation-risk summary from Monte Carlo paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidationRisk {
    pub prob_liquidation: f64,
    /// Expected first-passage time in years.
    pub expected_time_to_liquidation: Option<f64>,
    pub worst_case_funding_rate: f64,
}

/// Stress-test output for one scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressTestResult {
    pub scenario: StressScenario,
    pub risk: LiquidationRisk,
}

/// Monte Carlo liquidation simulator for isolated funding-rate positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidationSimulator {
    pub position: LiquidationPosition,
    pub model: FundingRateModel,
    pub initial_funding_rate: f64,
    pub num_paths: usize,
    pub steps: usize,
    pub seed: u64,
    pub rng_kind: FastRngKind,
}

impl LiquidationSimulator {
    pub fn new(
        position: LiquidationPosition,
        model: FundingRateModel,
        initial_funding_rate: f64,
        num_paths: usize,
        steps: usize,
        seed: u64,
    ) -> Self {
        validate_position(&position);
        assert!(
            initial_funding_rate.is_finite(),
            "initial_funding_rate must be finite"
        );
        assert!(num_paths > 0, "num_paths must be > 0");
        assert!(steps > 0, "steps must be > 0");

        Self {
            position,
            model,
            initial_funding_rate,
            num_paths,
            steps,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
        }
    }

    pub fn with_rng_kind(mut self, rng_kind: FastRngKind) -> Self {
        self.rng_kind = rng_kind;
        self
    }

    /// Estimates baseline liquidation risk before maturity.
    pub fn simulate(&self) -> LiquidationRisk {
        self.simulate_stress(StressScenario::Baseline)
    }

    /// Estimates liquidation risk under a single stress scenario.
    pub fn simulate_stress(&self, scenario: StressScenario) -> LiquidationRisk {
        let position = stressed_position(self.position, scenario);
        let model = self.model.stressed(scenario);
        let initial_rate =
            model.normalize_rate(stressed_initial_rate(self.initial_funding_rate, scenario));
        let total_time = position.margin_params.time_to_maturity;

        let initial_health = health_at_rate(position, initial_rate, total_time);
        let initially_liquidatable = MarginCalculator::is_liquidatable(initial_health);
        if initially_liquidatable || total_time <= EPSILON {
            return LiquidationRisk {
                prob_liquidation: if initially_liquidatable { 1.0 } else { 0.0 },
                expected_time_to_liquidation: if initially_liquidatable {
                    Some(0.0)
                } else {
                    None
                },
                worst_case_funding_rate: initial_rate,
            };
        }

        let dt = total_time / self.steps as f64;
        let mut liquidated_paths = 0usize;
        let mut liquidation_time_sum = 0.0;
        let mut global_worst_rate = initial_rate;

        for path_idx in 0..self.num_paths {
            let seed = resolve_stream_seed(self.seed, path_idx, true);
            let mut rng = FastRng::from_seed(self.rng_kind, seed);
            let mut rate = initial_rate;
            let mut path_worst_rate = initial_rate;
            let mut liquidated_at = None;

            for step in 1..=self.steps {
                let z = sample_standard_normal(&mut rng);
                rate = model.step(rate, dt, z);
                path_worst_rate = adverse_extreme(path_worst_rate, rate, position.size);

                let elapsed = step as f64 * dt;
                if elapsed + EPSILON >= total_time {
                    continue;
                }

                let health_ratio = health_at_rate(position, rate, total_time - elapsed);
                if MarginCalculator::is_liquidatable(health_ratio) {
                    liquidated_at = Some(elapsed);
                    break;
                }
            }

            global_worst_rate = adverse_extreme(global_worst_rate, path_worst_rate, position.size);

            if let Some(time_to_liquidation) = liquidated_at {
                liquidated_paths += 1;
                liquidation_time_sum += time_to_liquidation;
            }
        }

        LiquidationRisk {
            prob_liquidation: liquidated_paths as f64 / self.num_paths as f64,
            expected_time_to_liquidation: if liquidated_paths > 0 {
                Some(liquidation_time_sum / liquidated_paths as f64)
            } else {
                None
            },
            worst_case_funding_rate: global_worst_rate,
        }
    }

    /// Runs a batch of stress scenarios.
    pub fn run_stress_scenarios(&self, scenarios: &[StressScenario]) -> Vec<StressTestResult> {
        scenarios
            .iter()
            .copied()
            .map(|scenario| StressTestResult {
                scenario,
                risk: self.simulate_stress(scenario),
            })
            .collect()
    }
}

impl FundingRateModel {
    fn stressed(self, scenario: StressScenario) -> Self {
        match (self, scenario) {
            (Self::Vasicek(model), StressScenario::Baseline) => Self::Vasicek(model),
            (Self::CIR(model), StressScenario::Baseline) => Self::CIR(model),
            (Self::Vasicek(model), StressScenario::LiquidationCascade { vol_multiplier }) => {
                Self::Vasicek(Vasicek {
                    sigma: model.sigma * vol_multiplier,
                    ..model
                })
            }
            (Self::CIR(model), StressScenario::LiquidationCascade { vol_multiplier }) => {
                Self::CIR(CIR {
                    sigma: model.sigma * vol_multiplier,
                    ..model
                })
            }
            (Self::Vasicek(model), StressScenario::MeanShift { shift }) => Self::Vasicek(Vasicek {
                b: model.b + shift,
                ..model
            }),
            (Self::CIR(model), StressScenario::MeanShift { shift }) => Self::CIR(CIR {
                b: (model.b + shift).max(0.0),
                ..model
            }),
        }
    }

    fn normalize_rate(self, rate: f64) -> f64 {
        match self {
            Self::Vasicek(_) => rate,
            Self::CIR(_) => rate.max(0.0),
        }
    }

    fn step(self, current: f64, dt: f64, z: f64) -> f64 {
        match self {
            Self::Vasicek(model) => step_vasicek(model, current, dt, z),
            Self::CIR(model) => step_cir(model, current, dt, z),
        }
    }
}

fn step_vasicek(model: Vasicek, current: f64, dt: f64, z: f64) -> f64 {
    if dt <= EPSILON {
        return current;
    }

    if model.a.abs() <= EPSILON {
        return current + model.sigma * dt.sqrt() * z;
    }

    let exp_neg_a_dt = (-model.a * dt).exp();
    let variance =
        model.sigma * model.sigma * (1.0 - (-2.0 * model.a * dt).exp()) / (2.0 * model.a);
    model.b + exp_neg_a_dt * (current - model.b) + variance.max(0.0).sqrt() * z
}

fn step_cir(model: CIR, current: f64, dt: f64, z: f64) -> f64 {
    if dt <= EPSILON {
        return current.max(0.0);
    }

    let current = current.max(0.0);
    let drift = model.a * (model.b - current) * dt;
    let diffusion = model.sigma * current.sqrt() * dt.sqrt() * z;
    (current + drift + diffusion).max(0.0)
}

fn stressed_position(
    position: LiquidationPosition,
    scenario: StressScenario,
) -> LiquidationPosition {
    let mut stressed = position;
    if let StressScenario::LiquidationCascade { vol_multiplier } = scenario {
        stressed.margin_params.funding_rate_vol *= vol_multiplier;
    }
    stressed
}

fn stressed_initial_rate(initial_rate: f64, scenario: StressScenario) -> f64 {
    match scenario {
        StressScenario::Baseline => initial_rate,
        StressScenario::LiquidationCascade { .. } => initial_rate,
        StressScenario::MeanShift { shift } => initial_rate + shift,
    }
}

fn health_at_rate(position: LiquidationPosition, funding_rate: f64, remaining_time: f64) -> f64 {
    let mut margin_params = position.margin_params;
    margin_params.time_to_maturity = remaining_time.max(0.0);
    let unrealized_pnl = position.size * (position.entry_rate - funding_rate);
    MarginCalculator::health_ratio(
        position.collateral,
        position.size.abs(),
        unrealized_pnl,
        &margin_params,
    )
}

fn adverse_extreme(current: f64, candidate: f64, size: f64) -> f64 {
    if size >= 0.0 {
        current.max(candidate)
    } else {
        current.min(candidate)
    }
}

fn validate_position(position: &LiquidationPosition) {
    assert!(
        position.size.is_finite() && position.size.abs() > EPSILON,
        "position size must be finite and non-zero"
    );
    assert!(position.entry_rate.is_finite(), "entry_rate must be finite");
    assert!(
        position.collateral.is_finite() && position.collateral >= 0.0,
        "collateral must be finite and >= 0"
    );
    let _ = MarginCalculator::initial_margin(position.size.abs(), &position.margin_params);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_roundoff(actual: f64, expected: f64, label: &str) {
        let tolerance = 32.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.17}, expected={expected:.17}, tolerance={tolerance:e}"
        );
    }

    fn margin_params() -> MarginParams {
        MarginParams {
            initial_margin_ratio: 0.20,
            maintenance_margin_ratio: 0.10,
            funding_rate_vol: 0.20,
            time_to_maturity: 1.0,
            tick_size: 0.0001,
        }
    }

    fn baseline_simulator(collateral: f64) -> LiquidationSimulator {
        let position = LiquidationPosition {
            size: 100.0,
            entry_rate: 0.05,
            collateral,
            margin_params: margin_params(),
        };
        let model = FundingRateModel::Vasicek(Vasicek {
            a: 2.5,
            b: 0.05,
            sigma: 0.08,
        });

        LiquidationSimulator::new(position, model, 0.05, 5_000, 64, 7)
    }

    #[test]
    fn vasicek_first_passage_matches_scipy_sobol_reference_and_seeded_grid() {
        let risk = baseline_simulator(4.8).simulate();

        // Independent reference: 16 independently scrambled 2^16-path Sobol
        // replicates from SciPy 1.17.1 / NumPy 2.4.3. Each path uses the exact
        // Gaussian Vasicek transition and applies the same 63 pre-maturity
        // monitoring dates. The replicate standard errors were 2.78e-4 for
        // the hit probability and 1.94e-4 years for the conditional hit time.
        const REFERENCE_PROBABILITY: f64 = 0.642_266_273_498_535_2;
        const REFERENCE_TIME: f64 = 0.315_824_911_447_977_2;
        const CONDITIONAL_TIME_STDDEV: f64 = 0.249_041_26;
        const REFERENCE_PROBABILITY_ERROR: f64 = 2.78e-4;
        const REFERENCE_TIME_ERROR: f64 = 1.94e-4;

        let probability_stderr =
            (REFERENCE_PROBABILITY * (1.0 - REFERENCE_PROBABILITY) / 5_000.0).sqrt();
        let time_stderr = CONDITIONAL_TIME_STDDEV / (5_000.0 * REFERENCE_PROBABILITY).sqrt();
        assert!(
            (risk.prob_liquidation - REFERENCE_PROBABILITY).abs()
                <= 4.0 * (probability_stderr.powi(2) + REFERENCE_PROBABILITY_ERROR.powi(2)).sqrt(),
            "implementation={risk:?}, reference probability={REFERENCE_PROBABILITY}"
        );
        let expected_time = risk
            .expected_time_to_liquidation
            .expect("the referenced scenario has liquidation events");
        assert!(
            (expected_time - REFERENCE_TIME).abs()
                <= 4.0 * (time_stderr.powi(2) + REFERENCE_TIME_ERROR.powi(2)).sqrt(),
            "implementation={risk:?}, reference time={REFERENCE_TIME}"
        );

        // Exact seeded regression locks the implementation's path count,
        // first-passage grid and stream partitioning in addition to the
        // independent distribution-level comparison above.
        assert_eq!(risk.prob_liquidation, 0.6436);
        assert_eq!(expected_time, 0.317_146_713_797_389_7);
        assert_roundoff(
            risk.worst_case_funding_rate,
            0.117_587_060_517_995_33,
            "seeded adverse rate",
        );
    }

    #[test]
    fn high_collateral_seeded_tail_probability_is_not_a_wide_range_check() {
        let risk = baseline_simulator(15.0).simulate();
        assert_eq!(risk.prob_liquidation, 0.0006);
        assert_eq!(
            risk.expected_time_to_liquidation,
            Some(0.713_541_666_666_666_6)
        );
        assert_roundoff(
            risk.worst_case_funding_rate,
            0.201_017_170_565_720_35,
            "tail-scenario adverse rate",
        );
    }

    #[test]
    fn initially_under_margined_stress_liquidates_at_time_zero() {
        let risk = baseline_simulator(6.0).simulate_stress(StressScenario::LiquidationCascade {
            vol_multiplier: 5.0,
        });
        assert_eq!(
            risk,
            LiquidationRisk {
                prob_liquidation: 1.0,
                expected_time_to_liquidation: Some(0.0),
                worst_case_funding_rate: 0.05,
            }
        );
    }

    #[test]
    fn zero_vol_cir_path_has_exact_first_passage_time() {
        let simulator = LiquidationSimulator::new(
            LiquidationPosition {
                size: 100.0,
                entry_rate: 0.05,
                collateral: 2.5,
                margin_params: margin_params(),
            },
            FundingRateModel::CIR(CIR {
                a: 1.0,
                b: 0.20,
                sigma: 0.0,
            }),
            0.05,
            17,
            4,
            123,
        );
        let risk = simulator.simulate();
        assert_eq!(risk.prob_liquidation, 1.0);
        assert_eq!(risk.expected_time_to_liquidation, Some(0.25));
        assert_roundoff(
            risk.worst_case_funding_rate,
            0.087_500_000_000_000_01,
            "deterministic CIR adverse rate",
        );
    }
}
