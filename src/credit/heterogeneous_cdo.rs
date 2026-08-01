//! Finite heterogeneous-pool synthetic CDO pricing.
//!
//! The one-factor Gaussian-copula calculation conditions on the common factor,
//! builds the exact conditional portfolio-loss distribution by Bernoulli
//! recursion, and integrates the conditional tranche loss over the factor.
//! This is the recursion described by Andersen, Sidenius and Basu, "All Your
//! Hedges in One Basket", Risk, November 2003, pages 67-72, and implemented by
//! QuantLib's `RecursiveGaussLossModel`.
//!
//! Unlike [`super::SyntheticCdo`], which deliberately retains its large
//! homogeneous-pool (LHP) model, this engine keeps every reference name's
//! survival curve, exposure, recovery, and factor loading.

use crate::core::PricingError;
use crate::math::{gauss_legendre_integrate, normal_cdf, normal_inv_cdf, normal_pdf};

use super::{CdoTranche, SurvivalCurve};

const FACTOR_LOWER: f64 = -8.0;
const FACTOR_UPPER: f64 = 8.0;
const MIN_FACTOR_INTEGRATION_POINTS: usize = 32;
const MAX_INITIAL_FACTOR_INTEGRATION_POINTS: usize = 2_048;
const MAX_FACTOR_INTEGRATION_POINTS: usize = 4_096;
const FACTOR_CONVERGENCE_TOLERANCE: f64 = 2.0e-12;
const MAX_LOSS_STATES: usize = 1_000_001;

/// One reference name in a finite heterogeneous synthetic-CDO pool.
#[derive(Debug, Clone, PartialEq)]
pub struct CdoReferenceName {
    /// Exposure before recovery. Positive exposures are normalized internally
    /// to portfolio weights.
    pub notional: f64,
    /// Deterministic recovery rate in `[0, 1)`.
    pub recovery_rate: f64,
    /// Loading `beta_i` on the common standard-normal factor. The asset
    /// correlation of names `i` and `j` is `beta_i * beta_j`.
    pub factor_loading: f64,
    /// Marginal survival-probability term structure.
    pub survival_curve: SurvivalCurve,
}

/// Base correlations for the two equity tranches used to synthesize an
/// attachment-detachment tranche.
///
/// Both values are asset correlations, not factor loadings. They represent a
/// single maturity slice and are held fixed across the contract's payment
/// dates, matching FinancePy's `CDSTranche.value_bc` convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaseCorrelationPair {
    /// Base correlation at the tranche attachment point. Ignored when the
    /// attachment is zero.
    pub attachment_correlation: f64,
    /// Base correlation at the tranche detachment point.
    pub detachment_correlation: f64,
}

/// Finite heterogeneous-pool one-factor Gaussian-copula CDO model.
#[derive(Debug, Clone, PartialEq)]
pub struct HeterogeneousSyntheticCdo {
    /// Reference names. Exposures are normalized to sum to one.
    pub names: Vec<CdoReferenceName>,
    /// Flat continuously compounded risk-free rate.
    pub risk_free_rate: f64,
    /// Contract maturity in years.
    pub maturity: f64,
    /// Coupon payment frequency per year.
    pub payment_freq: usize,
    /// Loss-distribution unit as a fraction of portfolio notional.
    ///
    /// Every name's normalized exposure times loss-given-default must be an
    /// integer multiple of this value. Requiring commensurability prevents a
    /// hidden loss-bucketing approximation: conditional recursion is exact on
    /// the declared finite loss grid.
    pub loss_unit: f64,
    /// Minimum Gauss-Legendre order for common-factor integration on
    /// `[-8, 8]`.
    ///
    /// The order is rounded up to a power of two and doubled automatically
    /// until consecutive tranche-loss estimates agree. Pricing returns a
    /// numerical error rather than an unchecked value if the 4096-point cap is
    /// reached without convergence. Inputs must lie in `[32, 2048]` so at
    /// least one finer grid is always available.
    pub factor_integration_points: usize,
}

impl HeterogeneousSyntheticCdo {
    /// Unconditional expected portfolio loss as a fraction of portfolio
    /// notional.
    pub fn portfolio_expected_loss(&self, t: f64) -> Result<f64, PricingError> {
        self.validate_model()?;
        validate_horizon(t)?;
        if t <= 0.0 {
            return Ok(0.0);
        }

        let total_notional = self.total_notional();
        Ok(self
            .names
            .iter()
            .map(|name| {
                let weight = name.notional / total_notional;
                let default_probability =
                    (1.0 - name.survival_curve.survival_prob(t)).clamp(0.0, 1.0);
                weight * (1.0 - name.recovery_rate) * default_probability
            })
            .sum())
    }

    /// Expected tranche loss as a fraction of tranche notional at horizon
    /// `t`, using each name's configured factor loading.
    pub fn expected_loss_fraction(
        &self,
        tranche: &CdoTranche,
        t: f64,
    ) -> Result<f64, PricingError> {
        let loadings = self
            .names
            .iter()
            .map(|name| name.factor_loading)
            .collect::<Vec<_>>();
        self.expected_loss_fraction_with_loadings(tranche, t, &loadings)
    }

    /// Expected tranche loss as a fraction of tranche notional under a base
    /// correlation pair.
    ///
    /// The `[A, D]` loss is constructed as `EL[0,D](rho_D) -
    /// EL[0,A](rho_A)`. An inconsistent pair that produces a negative or
    /// above-width expected loss is rejected rather than silently clamped.
    pub fn expected_loss_fraction_base_correlation(
        &self,
        tranche: &CdoTranche,
        t: f64,
        correlations: BaseCorrelationPair,
    ) -> Result<f64, PricingError> {
        self.validate_model()?;
        validate_tranche(tranche)?;
        correlations.validate()?;
        validate_horizon(t)?;
        if t <= 0.0 {
            return Ok(0.0);
        }

        let equity_detachment = self.expected_equity_loss_pool_fraction(
            tranche.detachment,
            t,
            correlations.detachment_correlation,
        )?;
        let equity_attachment = if tranche.attachment <= 0.0 {
            0.0
        } else {
            self.expected_equity_loss_pool_fraction(
                tranche.attachment,
                t,
                correlations.attachment_correlation,
            )?
        };
        let pool_loss = equity_detachment - equity_attachment;
        let width = tranche.width();
        let numerical_tolerance = 2.0e-12;
        if pool_loss < -numerical_tolerance || pool_loss > width + numerical_tolerance {
            return Err(PricingError::InvalidInput(format!(
                "base-correlation pair implies tranche expected loss {pool_loss}, outside [0, {width}]"
            )));
        }
        Ok(pool_loss.clamp(0.0, width) / width)
    }

    /// Expected tranche loss in currency units at horizon `t`.
    pub fn expected_tranche_loss(&self, tranche: &CdoTranche, t: f64) -> Result<f64, PricingError> {
        Ok(tranche.notional * self.expected_loss_fraction(tranche, t)?)
    }

    /// Expected tranche loss in currency units under base correlation.
    pub fn expected_tranche_loss_base_correlation(
        &self,
        tranche: &CdoTranche,
        t: f64,
        correlations: BaseCorrelationPair,
    ) -> Result<f64, PricingError> {
        Ok(tranche.notional
            * self.expected_loss_fraction_base_correlation(tranche, t, correlations)?)
    }

    /// Protection-leg PV using each name's configured factor loading.
    pub fn protection_leg_pv(&self, tranche: &CdoTranche) -> Result<f64, PricingError> {
        self.protection_leg_pv_with(tranche, |t| self.expected_loss_fraction(tranche, t))
    }

    /// Premium-leg PV for `spread` using each name's configured factor
    /// loading.
    pub fn premium_leg_pv(&self, tranche: &CdoTranche, spread: f64) -> Result<f64, PricingError> {
        self.premium_leg_pv_with(tranche, spread, |t| self.expected_loss_fraction(tranche, t))
    }

    /// Fair running spread using each name's configured factor loading.
    pub fn fair_spread(&self, tranche: &CdoTranche) -> Result<f64, PricingError> {
        let protection = self.protection_leg_pv(tranche)?;
        let annuity = self.premium_leg_pv(tranche, 1.0)?;
        if annuity <= 1.0e-14 {
            return Err(PricingError::NumericalError(
                "heterogeneous CDO premium annuity is zero".to_string(),
            ));
        }
        Ok(protection / annuity)
    }

    /// Protection-buyer NPV using the tranche's running spread.
    pub fn npv(&self, tranche: &CdoTranche) -> Result<f64, PricingError> {
        Ok(self.protection_leg_pv(tranche)? - self.premium_leg_pv(tranche, tranche.spread)?)
    }

    /// Protection-leg PV under a base-correlation pair.
    pub fn protection_leg_pv_base_correlation(
        &self,
        tranche: &CdoTranche,
        correlations: BaseCorrelationPair,
    ) -> Result<f64, PricingError> {
        self.protection_leg_pv_with(tranche, |t| {
            self.expected_loss_fraction_base_correlation(tranche, t, correlations)
        })
    }

    /// Premium-leg PV under a base-correlation pair.
    pub fn premium_leg_pv_base_correlation(
        &self,
        tranche: &CdoTranche,
        spread: f64,
        correlations: BaseCorrelationPair,
    ) -> Result<f64, PricingError> {
        self.premium_leg_pv_with(tranche, spread, |t| {
            self.expected_loss_fraction_base_correlation(tranche, t, correlations)
        })
    }

    /// Fair running spread under a base-correlation pair.
    pub fn fair_spread_base_correlation(
        &self,
        tranche: &CdoTranche,
        correlations: BaseCorrelationPair,
    ) -> Result<f64, PricingError> {
        let protection = self.protection_leg_pv_base_correlation(tranche, correlations)?;
        let annuity = self.premium_leg_pv_base_correlation(tranche, 1.0, correlations)?;
        if annuity <= 1.0e-14 {
            return Err(PricingError::NumericalError(
                "base-correlation CDO premium annuity is zero".to_string(),
            ));
        }
        Ok(protection / annuity)
    }

    /// Protection-buyer NPV under a base-correlation pair.
    pub fn npv_base_correlation(
        &self,
        tranche: &CdoTranche,
        correlations: BaseCorrelationPair,
    ) -> Result<f64, PricingError> {
        Ok(
            self.protection_leg_pv_base_correlation(tranche, correlations)?
                - self.premium_leg_pv_base_correlation(tranche, tranche.spread, correlations)?,
        )
    }

    fn expected_equity_loss_pool_fraction(
        &self,
        detachment: f64,
        t: f64,
        asset_correlation: f64,
    ) -> Result<f64, PricingError> {
        let equity = CdoTranche {
            attachment: 0.0,
            detachment,
            notional: detachment,
            spread: 0.0,
        };
        let loading = asset_correlation.sqrt();
        let loadings = vec![loading; self.names.len()];
        Ok(detachment * self.expected_loss_fraction_with_loadings(&equity, t, &loadings)?)
    }

    fn expected_loss_fraction_with_loadings(
        &self,
        tranche: &CdoTranche,
        t: f64,
        loadings: &[f64],
    ) -> Result<f64, PricingError> {
        self.validate_model()?;
        validate_tranche(tranche)?;
        validate_horizon(t)?;
        if loadings.len() != self.names.len() {
            return Err(PricingError::InvalidInput(format!(
                "factor-loading count {} must equal name count {}",
                loadings.len(),
                self.names.len()
            )));
        }
        if loadings
            .iter()
            .any(|beta| !beta.is_finite() || beta.abs() >= 1.0)
        {
            return Err(PricingError::InvalidInput(
                "factor loadings must be finite and have absolute value < 1".to_string(),
            ));
        }
        if t <= 0.0 {
            return Ok(0.0);
        }

        let loss_units = self.loss_units()?;
        let total_loss_units = loss_units.iter().sum::<usize>();
        let thresholds = self
            .names
            .iter()
            .map(|name| {
                let q = (1.0 - name.survival_curve.survival_prob(t)).clamp(0.0, 1.0);
                if q <= 0.0 {
                    f64::NEG_INFINITY
                } else if q >= 1.0 {
                    f64::INFINITY
                } else {
                    refined_normal_inv_cdf(q)
                }
            })
            .collect::<Vec<_>>();
        let width = tranche.width();
        let attachment = tranche.attachment;
        let loss_unit = self.loss_unit;

        let integrand = |factor: f64| {
            let conditional_probabilities = thresholds
                .iter()
                .zip(loadings.iter())
                .map(|(&threshold, &beta)| {
                    if threshold == f64::NEG_INFINITY {
                        0.0
                    } else if threshold == f64::INFINITY {
                        1.0
                    } else {
                        normal_cdf((threshold - beta * factor) / (1.0 - beta * beta).sqrt())
                            .clamp(0.0, 1.0)
                    }
                })
                .collect::<Vec<_>>();
            let conditional_loss = conditional_tranche_loss(
                &conditional_probabilities,
                &loss_units,
                total_loss_units,
                loss_unit,
                attachment,
                width,
            );
            conditional_loss * normal_pdf(factor)
        };

        let expected_pool_loss = self.integrate_factor_converged(integrand, width)?;
        if !expected_pool_loss.is_finite() {
            return Err(PricingError::NumericalError(
                "heterogeneous CDO expected loss is non-finite".to_string(),
            ));
        }
        Ok((expected_pool_loss / width).clamp(0.0, 1.0))
    }

    fn integrate_factor_converged<F>(
        &self,
        integrand: F,
        tranche_width: f64,
    ) -> Result<f64, PricingError>
    where
        F: Fn(f64) -> f64,
    {
        let mut order = self.factor_integration_points.next_power_of_two();
        let mut previous = gauss_legendre_integrate(&integrand, FACTOR_LOWER, FACTOR_UPPER, order)
            .map_err(|error| {
                PricingError::NumericalError(format!(
                    "heterogeneous CDO factor integration failed at order {order}: {error:?}"
                ))
            })?;
        let mut previous_grid_converged = false;

        loop {
            let next_order = order
                .checked_mul(2)
                .unwrap_or(MAX_FACTOR_INTEGRATION_POINTS)
                .min(MAX_FACTOR_INTEGRATION_POINTS);
            let current = gauss_legendre_integrate(
                &integrand,
                FACTOR_LOWER,
                FACTOR_UPPER,
                next_order,
            )
            .map_err(|error| {
                PricingError::NumericalError(format!(
                    "heterogeneous CDO factor integration failed at order {next_order}: {error:?}"
                ))
            })?;
            let normalized_difference = (current - previous).abs() / tranche_width;
            let current_grid_converged = normalized_difference <= FACTOR_CONVERGENCE_TOLERANCE;
            // Away from the cap, require agreement across two successive
            // refinements. This prevents an accidental match on one pair of
            // global rules from being mistaken for convergence. At the cap,
            // the final fine/coarse comparison is the last available check.
            if current_grid_converged
                && (previous_grid_converged || next_order >= MAX_FACTOR_INTEGRATION_POINTS)
            {
                return Ok(current);
            }
            if next_order >= MAX_FACTOR_INTEGRATION_POINTS {
                return Err(PricingError::NumericalError(format!(
                    "heterogeneous CDO factor integration did not converge: orders {order} and {next_order} differ by {normalized_difference} in tranche-loss fraction (tolerance {FACTOR_CONVERGENCE_TOLERANCE})"
                )));
            }
            order = next_order;
            previous = current;
            previous_grid_converged = current_grid_converged;
        }
    }

    fn protection_leg_pv_with<F>(
        &self,
        tranche: &CdoTranche,
        mut expected_loss_fraction: F,
    ) -> Result<f64, PricingError>
    where
        F: FnMut(f64) -> Result<f64, PricingError>,
    {
        self.validate_model()?;
        validate_tranche(tranche)?;
        let mut pv_unit = 0.0;
        let mut previous_time = 0.0;
        let mut previous_loss = 0.0;
        for time in self.payment_times() {
            let current_loss = expected_loss_fraction(time)?;
            if current_loss + 2.0e-12 < previous_loss {
                return Err(PricingError::NumericalError(
                    "tranche expected loss decreases between payment dates".to_string(),
                ));
            }
            let incremental_loss = (current_loss - previous_loss).max(0.0);
            pv_unit += self.discount_factor(0.5 * (previous_time + time)) * incremental_loss;
            previous_time = time;
            previous_loss = current_loss;
        }
        Ok(tranche.notional * pv_unit)
    }

    fn premium_leg_pv_with<F>(
        &self,
        tranche: &CdoTranche,
        spread: f64,
        mut expected_loss_fraction: F,
    ) -> Result<f64, PricingError>
    where
        F: FnMut(f64) -> Result<f64, PricingError>,
    {
        self.validate_model()?;
        validate_tranche(tranche)?;
        if !spread.is_finite() || spread < 0.0 {
            return Err(PricingError::InvalidInput(
                "CDO running spread must be finite and non-negative".to_string(),
            ));
        }

        let mut annuity_unit = 0.0;
        let mut previous_time = 0.0;
        let mut previous_loss = 0.0;
        for time in self.payment_times() {
            let current_loss = expected_loss_fraction(time)?;
            if current_loss + 2.0e-12 < previous_loss {
                return Err(PricingError::NumericalError(
                    "tranche expected loss decreases between payment dates".to_string(),
                ));
            }
            let outstanding = (1.0 - 0.5 * (previous_loss + current_loss)).clamp(0.0, 1.0);
            annuity_unit += (time - previous_time) * self.discount_factor(time) * outstanding;
            previous_time = time;
            previous_loss = current_loss;
        }
        Ok(tranche.notional * spread * annuity_unit)
    }

    fn loss_units(&self) -> Result<Vec<usize>, PricingError> {
        let total_notional = self.total_notional();
        let mut total_units = 0usize;
        let units = self
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let loss_amount =
                    name.notional / total_notional * (1.0 - name.recovery_rate);
                let units_exact = loss_amount / self.loss_unit;
                let units_rounded = units_exact.round();
                let tolerance = 2.0e-10 * units_exact.abs().max(1.0);
                if (units_exact - units_rounded).abs() > tolerance || units_rounded < 1.0 {
                    return Err(PricingError::InvalidInput(format!(
                        "name {index} loss {loss_amount} is not a positive integer multiple of loss_unit {}",
                        self.loss_unit
                    )));
                }
                if units_rounded > usize::MAX as f64 {
                    return Err(PricingError::InvalidInput(format!(
                        "name {index} loss-unit count is too large"
                    )));
                }
                let count = units_rounded as usize;
                total_units = total_units.checked_add(count).ok_or_else(|| {
                    PricingError::InvalidInput("portfolio loss grid overflows usize".to_string())
                })?;
                Ok(count)
            })
            .collect::<Result<Vec<_>, PricingError>>()?;
        let state_count = total_units.checked_add(1).ok_or_else(|| {
            PricingError::InvalidInput("portfolio loss state count overflows usize".to_string())
        })?;
        if state_count > MAX_LOSS_STATES {
            return Err(PricingError::InvalidInput(format!(
                "portfolio loss grid has {} states; maximum is {MAX_LOSS_STATES}",
                state_count
            )));
        }
        Ok(units)
    }

    fn validate_model(&self) -> Result<(), PricingError> {
        if self.names.is_empty() {
            return Err(PricingError::InvalidInput(
                "heterogeneous CDO must contain at least one name".to_string(),
            ));
        }
        if !self.risk_free_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "risk-free rate must be finite".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "CDO maturity must be finite and positive".to_string(),
            ));
        }
        if self.payment_freq == 0 {
            return Err(PricingError::InvalidInput(
                "CDO payment frequency must be positive".to_string(),
            ));
        }
        if !self.loss_unit.is_finite() || self.loss_unit <= 0.0 || self.loss_unit > 1.0 {
            return Err(PricingError::InvalidInput(
                "loss_unit must be finite and in (0, 1]".to_string(),
            ));
        }
        if self.factor_integration_points < MIN_FACTOR_INTEGRATION_POINTS
            || self.factor_integration_points > MAX_INITIAL_FACTOR_INTEGRATION_POINTS
        {
            return Err(PricingError::InvalidInput(format!(
                "factor integration points must be in [{MIN_FACTOR_INTEGRATION_POINTS}, {MAX_INITIAL_FACTOR_INTEGRATION_POINTS}]"
            )));
        }
        for (index, name) in self.names.iter().enumerate() {
            if !name.notional.is_finite() || name.notional <= 0.0 {
                return Err(PricingError::InvalidInput(format!(
                    "name {index} notional must be finite and positive"
                )));
            }
            if !name.recovery_rate.is_finite() || !(0.0..1.0).contains(&name.recovery_rate) {
                return Err(PricingError::InvalidInput(format!(
                    "name {index} recovery must be finite and in [0, 1)"
                )));
            }
            if !name.factor_loading.is_finite() || name.factor_loading.abs() >= 1.0 {
                return Err(PricingError::InvalidInput(format!(
                    "name {index} factor loading must be finite with absolute value < 1"
                )));
            }
            validate_survival_curve(index, &name.survival_curve)?;
        }
        if !self.total_notional().is_finite() {
            return Err(PricingError::InvalidInput(
                "portfolio notional sum must be finite".to_string(),
            ));
        }
        Ok(())
    }

    fn total_notional(&self) -> f64 {
        self.names.iter().map(|name| name.notional).sum()
    }

    fn payment_times(&self) -> Vec<f64> {
        let dt = 1.0 / self.payment_freq as f64;
        let mut time = 0.0;
        let mut times = Vec::new();
        while time + dt < self.maturity - 1.0e-12 {
            time += dt;
            times.push(time);
        }
        times.push(self.maturity);
        times
    }

    fn discount_factor(&self, t: f64) -> f64 {
        (-self.risk_free_rate * t.max(0.0)).exp()
    }
}

impl BaseCorrelationPair {
    fn validate(self) -> Result<(), PricingError> {
        for (name, correlation) in [
            ("attachment", self.attachment_correlation),
            ("detachment", self.detachment_correlation),
        ] {
            if !correlation.is_finite() || !(0.0..1.0).contains(&correlation) {
                return Err(PricingError::InvalidInput(format!(
                    "{name} base correlation must be finite and in [0, 1)"
                )));
            }
        }
        Ok(())
    }
}

fn validate_tranche(tranche: &CdoTranche) -> Result<(), PricingError> {
    if !tranche.attachment.is_finite()
        || !tranche.detachment.is_finite()
        || tranche.attachment < 0.0
        || tranche.detachment <= tranche.attachment
        || tranche.detachment > 1.0
    {
        return Err(PricingError::InvalidInput(
            "tranche attachment/detachment must satisfy 0 <= A < D <= 1".to_string(),
        ));
    }
    if !tranche.notional.is_finite() || tranche.notional <= 0.0 {
        return Err(PricingError::InvalidInput(
            "tranche notional must be finite and positive".to_string(),
        ));
    }
    if !tranche.spread.is_finite() || tranche.spread < 0.0 {
        return Err(PricingError::InvalidInput(
            "tranche spread must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn validate_horizon(t: f64) -> Result<(), PricingError> {
    if !t.is_finite() {
        return Err(PricingError::InvalidInput(
            "CDO loss horizon must be finite".to_string(),
        ));
    }
    Ok(())
}

fn validate_survival_curve(name_index: usize, curve: &SurvivalCurve) -> Result<(), PricingError> {
    if curve.tenors.is_empty() {
        return Err(PricingError::InvalidInput(format!(
            "name {name_index} survival curve must contain at least one tenor"
        )));
    }

    let mut previous_tenor = 0.0;
    let mut previous_survival = 1.0;
    for (node_index, &(tenor, survival)) in curve.tenors.iter().enumerate() {
        if !tenor.is_finite() || tenor <= 0.0 {
            return Err(PricingError::InvalidInput(format!(
                "name {name_index} survival curve node {node_index} tenor must be finite and positive"
            )));
        }
        if tenor <= previous_tenor {
            return Err(PricingError::InvalidInput(format!(
                "name {name_index} survival curve tenors must be strictly increasing"
            )));
        }
        if !survival.is_finite() || survival <= 0.0 || survival > 1.0 {
            return Err(PricingError::InvalidInput(format!(
                "name {name_index} survival curve node {node_index} probability must be finite and in (0, 1]"
            )));
        }
        if survival > previous_survival {
            return Err(PricingError::InvalidInput(format!(
                "name {name_index} survival probabilities must be non-increasing"
            )));
        }
        previous_tenor = tenor;
        previous_survival = survival;
    }
    Ok(())
}

/// Refine the library's fast rational inverse with Newton steps against the
/// accurate normal CDF. Credit-tail thresholds feed every factor node, so the
/// raw few-e-9 inverse error would otherwise be visible in tranche ETLs.
fn refined_normal_inv_cdf(probability: f64) -> f64 {
    let mut quantile = normal_inv_cdf(probability);
    for _ in 0..2 {
        quantile -= (normal_cdf(quantile) - probability) / normal_pdf(quantile);
    }
    quantile
}

fn conditional_tranche_loss(
    conditional_default_probabilities: &[f64],
    loss_units: &[usize],
    total_loss_units: usize,
    loss_unit: f64,
    attachment: f64,
    width: f64,
) -> f64 {
    let mut distribution = vec![0.0; total_loss_units + 1];
    distribution[0] = 1.0;
    let mut reachable = 0usize;

    for (&probability, &name_loss_units) in conditional_default_probabilities
        .iter()
        .zip(loss_units.iter())
    {
        for state in (0..=reachable).rev() {
            let mass = distribution[state];
            distribution[state] = mass * (1.0 - probability);
            distribution[state + name_loss_units] += mass * probability;
        }
        reachable += name_loss_units;
    }

    distribution
        .iter()
        .take(reachable + 1)
        .enumerate()
        .map(|(state, &probability)| {
            let portfolio_loss = state as f64 * loss_unit;
            probability * (portfolio_loss - attachment).clamp(0.0, width)
        })
        .sum()
}
