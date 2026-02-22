//! Credit analytics for Cdo.
//!
//! Module openferric::credit::cdo provides pricing helpers and model utilities for credit products.

use crate::math::{gauss_legendre_integrate, normal_cdf, normal_inv_cdf, normal_pdf};

/// Synthetic CDO tranche definition.
#[derive(Debug, Clone, PartialEq)]
pub struct CdoTranche {
    /// Attachment point as fraction of portfolio notional.
    pub attachment: f64,
    /// Detachment point as fraction of portfolio notional.
    pub detachment: f64,
    /// Tranche notional.
    pub notional: f64,
    /// Running spread in decimal (e.g. 0.01 for 100 bps).
    pub spread: f64,
}

/// Large homogeneous pool (LHP) synthetic CDO model.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticCdo {
    /// Number of names in the reference pool.
    pub num_names: usize,
    /// Flat pool spread proxy used as a constant hazard level in this LHP approximation.
    pub pool_spread: f64,
    /// Recovery rate.
    pub recovery_rate: f64,
    /// One-factor Gaussian correlation.
    pub correlation: f64,
    /// Flat risk-free rate.
    pub risk_free_rate: f64,
    /// Contract maturity in years.
    pub maturity: f64,
    /// Coupon payment frequency per year.
    pub payment_freq: usize,
}

impl CdoTranche {
    pub fn width(&self) -> f64 {
        self.detachment - self.attachment
    }

    fn is_valid(&self) -> bool {
        self.notional > 0.0
            && self.spread >= 0.0
            && self.attachment >= 0.0
            && self.detachment > self.attachment
            && self.detachment <= 1.0
    }

    /// Expected tranche loss as fraction of tranche notional at horizon `t`.
    pub fn expected_loss_fraction(
        &self,
        default_probability: f64,
        recovery_rate: f64,
        correlation: f64,
    ) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }
        let width = self.width();
        let loss_pool = expected_tranche_loss_pool_fraction(
            self.attachment,
            self.detachment,
            default_probability,
            recovery_rate,
            correlation,
        );
        (loss_pool / width).clamp(0.0, 1.0)
    }
}

impl SyntheticCdo {
    pub fn hazard_rate(&self) -> f64 {
        self.pool_spread.max(0.0)
    }

    pub fn default_probability(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let h = self.hazard_rate();
        (1.0 - (-h * t).exp()).clamp(0.0, 1.0)
    }

    /// Expected portfolio loss as fraction of portfolio notional.
    pub fn portfolio_expected_loss(&self, t: f64) -> f64 {
        (1.0 - self.recovery_rate).max(0.0) * self.default_probability(t)
    }

    /// Expected tranche loss in currency units at horizon `t`.
    pub fn expected_tranche_loss(&self, tranche: &CdoTranche, t: f64) -> f64 {
        tranche.notional
            * tranche.expected_loss_fraction(
                self.default_probability(t),
                self.recovery_rate,
                self.correlation,
            )
    }

    pub fn protection_leg_pv(&self, tranche: &CdoTranche) -> f64 {
        if !self.is_valid() || !tranche.is_valid() {
            return 0.0;
        }
        let mut pv_unit = 0.0;
        let mut prev_t = 0.0;
        let mut prev_loss = 0.0;
        for t in self.payment_times() {
            let current_loss = tranche.expected_loss_fraction(
                self.default_probability(t),
                self.recovery_rate,
                self.correlation,
            );
            let incremental_loss = (current_loss - prev_loss).max(0.0);
            let t_mid = 0.5 * (prev_t + t);
            pv_unit += self.discount_factor(t_mid) * incremental_loss;
            prev_t = t;
            prev_loss = current_loss;
        }
        tranche.notional * pv_unit
    }

    pub fn premium_leg_pv(&self, tranche: &CdoTranche, spread: f64) -> f64 {
        if !self.is_valid() || !tranche.is_valid() || spread < 0.0 {
            return 0.0;
        }
        let mut annuity_unit = 0.0;
        let mut prev_t = 0.0;
        let mut prev_loss = 0.0;
        for t in self.payment_times() {
            let dt = t - prev_t;
            let current_loss = tranche.expected_loss_fraction(
                self.default_probability(t),
                self.recovery_rate,
                self.correlation,
            );
            let outstanding = (1.0 - 0.5 * (prev_loss + current_loss)).clamp(0.0, 1.0);
            annuity_unit += dt * self.discount_factor(t) * outstanding;
            prev_t = t;
            prev_loss = current_loss;
        }
        tranche.notional * spread * annuity_unit
    }

    pub fn fair_spread(&self, tranche: &CdoTranche) -> f64 {
        if !self.is_valid() || !tranche.is_valid() {
            return 0.0;
        }
        let protection = self.protection_leg_pv(tranche);
        let annuity_at_1bp = self.premium_leg_pv(tranche, 1.0);
        if annuity_at_1bp <= 1.0e-14 {
            0.0
        } else {
            protection / annuity_at_1bp
        }
    }

    pub fn npv(&self, tranche: &CdoTranche) -> f64 {
        self.protection_leg_pv(tranche) - self.premium_leg_pv(tranche, tranche.spread)
    }

    fn payment_times(&self) -> Vec<f64> {
        if self.maturity <= 0.0 || self.payment_freq == 0 {
            return vec![];
        }
        let dt = 1.0 / self.payment_freq as f64;
        let mut t = 0.0;
        let mut times = Vec::new();
        while t + dt < self.maturity - 1.0e-12 {
            t += dt;
            times.push(t);
        }
        times.push(self.maturity);
        times
    }

    fn discount_factor(&self, t: f64) -> f64 {
        (-self.risk_free_rate.max(0.0) * t.max(0.0)).exp()
    }

    fn is_valid(&self) -> bool {
        self.num_names > 0
            && self.pool_spread >= 0.0
            && (0.0..1.0).contains(&self.recovery_rate)
            && self.correlation >= 0.0
            && self.correlation < 1.0
            && self.maturity > 0.0
            && self.payment_freq > 0
    }
}

/// Vasicek CDF for portfolio loss fraction under the LHP approximation.
pub fn vasicek_portfolio_loss_cdf(
    loss_fraction: f64,
    default_probability: f64,
    recovery_rate: f64,
    correlation: f64,
) -> f64 {
    if loss_fraction <= 0.0 {
        return 0.0;
    }

    let lgd = (1.0 - recovery_rate).clamp(0.0, 1.0);
    if lgd <= 0.0 {
        return 1.0;
    }
    if loss_fraction >= lgd {
        return 1.0;
    }

    let q = default_probability.clamp(1.0e-12, 1.0 - 1.0e-12);
    let rho = correlation.clamp(0.0, 0.999_999);
    if rho <= 1.0e-12 {
        let deterministic_loss = lgd * q;
        return if loss_fraction < deterministic_loss {
            0.0
        } else {
            1.0
        };
    }

    let x = normal_inv_cdf((loss_fraction / lgd).clamp(1.0e-12, 1.0 - 1.0e-12));
    let k = normal_inv_cdf(q);
    let arg = ((1.0 - rho).sqrt() * x - k) / rho.sqrt();
    normal_cdf(arg).clamp(0.0, 1.0)
}

fn expected_tranche_loss_pool_fraction(
    attachment: f64,
    detachment: f64,
    default_probability: f64,
    recovery_rate: f64,
    correlation: f64,
) -> f64 {
    let width = (detachment - attachment).max(0.0);
    if width <= 0.0 {
        return 0.0;
    }

    let q = default_probability.clamp(0.0, 1.0);
    if q <= 0.0 {
        return 0.0;
    }

    let lgd = (1.0 - recovery_rate).clamp(0.0, 1.0);
    if lgd <= 0.0 {
        return 0.0;
    }

    let rho = correlation.clamp(0.0, 0.999_999);
    if rho <= 1.0e-12 {
        let deterministic_loss = lgd * q;
        return (deterministic_loss - attachment).clamp(0.0, width);
    }

    let k = normal_inv_cdf(q.clamp(1.0e-12, 1.0 - 1.0e-12));
    let sqrt_rho = rho.sqrt();
    let sqrt_one_minus_rho = (1.0 - rho).sqrt();

    let integrand = |m: f64| {
        let cond_default_prob = normal_cdf((k - sqrt_rho * m) / sqrt_one_minus_rho);
        let loss = lgd * cond_default_prob;
        let tranche_loss = (loss - attachment).clamp(0.0, width);
        tranche_loss * normal_pdf(m)
    };

    let expected_loss = gauss_legendre_integrate(integrand, -8.0, 8.0, 96).unwrap_or_else(|_| {
        // Coarse fallback in the unlikely case quadrature setup fails.
        let n = 2_000usize;
        let a = -8.0;
        let b = 8.0;
        let h = (b - a) / n as f64;
        (0..n)
            .map(|i| {
                let x = a + (i as f64 + 0.5) * h;
                integrand(x)
            })
            .sum::<f64>()
            * h
    });

    expected_loss.clamp(0.0, width)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn tranche_losses_and_spreads_have_expected_ordering() {
        let cdo = SyntheticCdo {
            num_names: 125,
            pool_spread: 0.01,
            recovery_rate: 0.4,
            correlation: 0.30,
            risk_free_rate: 0.05,
            maturity: 5.0,
            payment_freq: 4,
        };

        let equity = CdoTranche {
            attachment: 0.0,
            detachment: 0.03,
            notional: 0.03,
            spread: 0.0,
        };
        let mezz = CdoTranche {
            attachment: 0.03,
            detachment: 0.07,
            notional: 0.04,
            spread: 0.0,
        };
        let senior = CdoTranche {
            attachment: 0.07,
            detachment: 1.0,
            notional: 0.93,
            spread: 0.0,
        };

        let t = cdo.maturity;
        let lf_equity = equity.expected_loss_fraction(
            cdo.default_probability(t),
            cdo.recovery_rate,
            cdo.correlation,
        );
        let lf_mezz = mezz.expected_loss_fraction(
            cdo.default_probability(t),
            cdo.recovery_rate,
            cdo.correlation,
        );
        let lf_senior = senior.expected_loss_fraction(
            cdo.default_probability(t),
            cdo.recovery_rate,
            cdo.correlation,
        );

        assert!(lf_equity > lf_mezz);
        assert!(lf_mezz > lf_senior);

        let el_equity = cdo.expected_tranche_loss(&equity, t);
        let el_mezz = cdo.expected_tranche_loss(&mezz, t);
        let el_senior = cdo.expected_tranche_loss(&senior, t);

        let spread_equity_bps = cdo.fair_spread(&equity) * 1.0e4;
        let spread_mezz_bps = cdo.fair_spread(&mezz) * 1.0e4;
        let spread_senior_bps = cdo.fair_spread(&senior) * 1.0e4;
        assert!(spread_equity_bps > spread_mezz_bps);
        assert!(spread_mezz_bps > spread_senior_bps);
        assert!(
            (500.0..=1700.0).contains(&spread_equity_bps),
            "equity spread: {spread_equity_bps:.2} bps"
        );
        assert!(
            (100.0..=450.0).contains(&spread_mezz_bps),
            "mezz spread: {spread_mezz_bps:.2} bps"
        );
        assert!(
            (10.0..=60.0).contains(&spread_senior_bps),
            "senior spread: {spread_senior_bps:.2} bps"
        );

        let tranche_sum = el_equity + el_mezz + el_senior;
        let portfolio_el = cdo.portfolio_expected_loss(t);
        assert_relative_eq!(tranche_sum, portfolio_el, epsilon = 4.0e-3);
    }

    #[test]
    fn vasicek_cdf_is_monotone_in_loss() {
        let q = 0.08;
        let recovery = 0.4;
        let rho = 0.3;
        let l1 = 0.01;
        let l2 = 0.04;
        let l3 = 0.10;

        let c1 = vasicek_portfolio_loss_cdf(l1, q, recovery, rho);
        let c2 = vasicek_portfolio_loss_cdf(l2, q, recovery, rho);
        let c3 = vasicek_portfolio_loss_cdf(l3, q, recovery, rho);
        assert!(c1 <= c2 && c2 <= c3);
    }
}
