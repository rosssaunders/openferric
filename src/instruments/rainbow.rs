//! Module `instruments::rainbow`.
//!
//! Implements rainbow abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `BestOfTwoCallOption`, `WorstOfTwoCallOption`, `TwoAssetCorrelationOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

fn validate_common(
    s1: f64,
    s2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    t: f64,
) -> Result<(), PricingError> {
    if !s1.is_finite() || !s2.is_finite() || s1 <= 0.0 || s2 <= 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow spots s1 and s2 must be finite and > 0".to_string(),
        ));
    }
    if !vol1.is_finite() || !vol2.is_finite() || vol1 <= 0.0 || vol2 <= 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow volatilities vol1 and vol2 must be finite and > 0".to_string(),
        ));
    }
    if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
        return Err(PricingError::InvalidInput(
            "rainbow correlation rho must be finite and in [-1, 1]".to_string(),
        ));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow maturity t must be finite and >= 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_rates(q1: f64, q2: f64, r: f64) -> Result<(), PricingError> {
    if !q1.is_finite() || !q2.is_finite() || !r.is_finite() {
        return Err(PricingError::InvalidInput(
            "rainbow rates q1, q2, and r must be finite".to_string(),
        ));
    }
    Ok(())
}

/// Two-asset best-of call: `max(max(S1_T, S2_T) - K, 0)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BestOfTwoCallOption {
    pub s1: f64,
    pub s2: f64,
    pub k: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl BestOfTwoCallOption {
    /// Validates option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.k.is_finite() || self.k < 0.0 {
            return Err(PricingError::InvalidInput(
                "best-of strike k must be finite and >= 0".to_string(),
            ));
        }
        validate_common(self.s1, self.s2, self.vol1, self.vol2, self.rho, self.t)?;
        validate_rates(self.q1, self.q2, self.r)
    }
}

impl Instrument for BestOfTwoCallOption {
    fn instrument_type(&self) -> &str {
        "BestOfTwoCallOption"
    }
}

/// Two-asset worst-of call: `max(min(S1_T, S2_T) - K, 0)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorstOfTwoCallOption {
    pub s1: f64,
    pub s2: f64,
    pub k: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl WorstOfTwoCallOption {
    /// Validates option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.k.is_finite() || self.k < 0.0 {
            return Err(PricingError::InvalidInput(
                "worst-of strike k must be finite and >= 0".to_string(),
            ));
        }
        validate_common(self.s1, self.s2, self.vol1, self.vol2, self.rho, self.t)?;
        validate_rates(self.q1, self.q2, self.r)
    }
}

impl Instrument for WorstOfTwoCallOption {
    fn instrument_type(&self) -> &str {
        "WorstOfTwoCallOption"
    }
}

/// Two-asset correlation option.
///
/// Call payoff: `1_{S2_T > K2} * max(S1_T - K1, 0)`
/// Put payoff:  `1_{S2_T < K2} * max(K1 - S1_T, 0)`
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TwoAssetCorrelationOption {
    pub option_type: OptionType,
    pub s1: f64,
    pub s2: f64,
    pub k1: f64,
    pub k2: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl TwoAssetCorrelationOption {
    /// Validates option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.k1.is_finite() || !self.k2.is_finite() || self.k1 <= 0.0 || self.k2 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "correlation option strikes k1 and k2 must be finite and > 0".to_string(),
            ));
        }
        validate_common(self.s1, self.s2, self.vol1, self.vol2, self.rho, self.t)?;
        validate_rates(self.q1, self.q2, self.r)
    }
}

impl Instrument for TwoAssetCorrelationOption {
    fn instrument_type(&self) -> &str {
        "TwoAssetCorrelationOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best_of() -> BestOfTwoCallOption {
        BestOfTwoCallOption {
            s1: 100.0,
            s2: 95.0,
            k: 100.0,
            vol1: 0.20,
            vol2: 0.25,
            rho: 0.30,
            q1: 0.01,
            q2: 0.02,
            r: 0.04,
            t: 1.5,
        }
    }

    fn worst_of() -> WorstOfTwoCallOption {
        let option = best_of();
        WorstOfTwoCallOption {
            s1: option.s1,
            s2: option.s2,
            k: option.k,
            vol1: option.vol1,
            vol2: option.vol2,
            rho: option.rho,
            q1: option.q1,
            q2: option.q2,
            r: option.r,
            t: option.t,
        }
    }

    fn correlation(option_type: OptionType) -> TwoAssetCorrelationOption {
        let option = best_of();
        TwoAssetCorrelationOption {
            option_type,
            s1: option.s1,
            s2: option.s2,
            k1: 100.0,
            k2: 90.0,
            vol1: option.vol1,
            vol2: option.vol2,
            rho: option.rho,
            q1: option.q1,
            q2: option.q2,
            r: option.r,
            t: option.t,
        }
    }

    fn assert_invalid(result: Result<(), PricingError>, message: &str) {
        assert_eq!(result, Err(PricingError::InvalidInput(message.to_string())));
    }

    #[test]
    fn rainbow_contracts_accept_boundary_terms_and_report_stable_types() {
        let best = BestOfTwoCallOption {
            k: 0.0,
            rho: -1.0,
            q1: -0.01,
            r: -0.04,
            t: 0.0,
            ..best_of()
        };
        let worst = WorstOfTwoCallOption {
            k: 0.0,
            rho: 1.0,
            q2: -0.02,
            t: 0.0,
            ..worst_of()
        };
        let call = correlation(OptionType::Call);
        let put = correlation(OptionType::Put);

        assert_eq!(best.validate(), Ok(()));
        assert_eq!(worst.validate(), Ok(()));
        assert_eq!(call.validate(), Ok(()));
        assert_eq!(put.validate(), Ok(()));
        assert_eq!(best.instrument_type(), "BestOfTwoCallOption");
        assert_eq!(worst.instrument_type(), "WorstOfTwoCallOption");
        assert_eq!(call.instrument_type(), "TwoAssetCorrelationOption");

        for (option, serialized_side) in [(call, "Call"), (put, "Put")] {
            let value = serde_json::to_value(option).expect("serialize correlation option");
            assert_eq!(value["option_type"], serialized_side);
            assert_eq!(
                serde_json::from_value::<TwoAssetCorrelationOption>(value)
                    .expect("deserialize correlation option"),
                option
            );
        }
    }

    #[test]
    fn rainbow_common_validation_rejects_every_non_finite_or_out_of_domain_field() {
        const SPOT_ERROR: &str = "rainbow spots s1 and s2 must be finite and > 0";
        const VOL_ERROR: &str = "rainbow volatilities vol1 and vol2 must be finite and > 0";
        const RHO_ERROR: &str = "rainbow correlation rho must be finite and in [-1, 1]";
        const MATURITY_ERROR: &str = "rainbow maturity t must be finite and >= 0";
        const RATE_ERROR: &str = "rainbow rates q1, q2, and r must be finite";

        for (option, message) in [
            (
                BestOfTwoCallOption {
                    s1: f64::NAN,
                    ..best_of()
                },
                SPOT_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    s2: f64::INFINITY,
                    ..best_of()
                },
                SPOT_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    s1: 0.0,
                    ..best_of()
                },
                SPOT_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    s2: 0.0,
                    ..best_of()
                },
                SPOT_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    vol1: f64::NAN,
                    ..best_of()
                },
                VOL_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    vol2: f64::INFINITY,
                    ..best_of()
                },
                VOL_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    vol2: 0.0,
                    ..best_of()
                },
                VOL_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    vol1: 0.0,
                    ..best_of()
                },
                VOL_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    rho: f64::NAN,
                    ..best_of()
                },
                RHO_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    rho: 1.000_000_000_1,
                    ..best_of()
                },
                RHO_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    t: f64::NAN,
                    ..best_of()
                },
                MATURITY_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    t: -f64::EPSILON,
                    ..best_of()
                },
                MATURITY_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    q1: f64::NAN,
                    ..best_of()
                },
                RATE_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    q2: f64::NEG_INFINITY,
                    ..best_of()
                },
                RATE_ERROR,
            ),
            (
                BestOfTwoCallOption {
                    r: f64::INFINITY,
                    ..best_of()
                },
                RATE_ERROR,
            ),
        ] {
            assert_invalid(option.validate(), message);
        }
    }

    #[test]
    fn rainbow_strike_validation_covers_each_contract_shape() {
        const BEST_ERROR: &str = "best-of strike k must be finite and >= 0";
        const WORST_ERROR: &str = "worst-of strike k must be finite and >= 0";
        const CORRELATION_ERROR: &str =
            "correlation option strikes k1 and k2 must be finite and > 0";

        for strike in [-f64::EPSILON, f64::NAN, f64::INFINITY] {
            assert_invalid(
                BestOfTwoCallOption {
                    k: strike,
                    ..best_of()
                }
                .validate(),
                BEST_ERROR,
            );
            assert_invalid(
                WorstOfTwoCallOption {
                    k: strike,
                    ..worst_of()
                }
                .validate(),
                WORST_ERROR,
            );
        }

        for option in [
            TwoAssetCorrelationOption {
                k1: 0.0,
                ..correlation(OptionType::Call)
            },
            TwoAssetCorrelationOption {
                k2: -f64::EPSILON,
                ..correlation(OptionType::Call)
            },
            TwoAssetCorrelationOption {
                k1: f64::NAN,
                ..correlation(OptionType::Put)
            },
            TwoAssetCorrelationOption {
                k2: f64::INFINITY,
                ..correlation(OptionType::Put)
            },
        ] {
            assert_invalid(option.validate(), CORRELATION_ERROR);
        }
    }
}
