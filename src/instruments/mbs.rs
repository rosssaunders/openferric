//! Module `instruments::mbs`.
//!
//! Implements MBS abstractions and re-exports used by adjacent pricing/model modules.
//! In this module, [`MbsPassThrough::coupon_rate`] is the pool's gross mortgage
//! coupon/WAC: scheduled amortization and refinancing incentive use that gross
//! rate, while pass-through interest is paid at `coupon_rate - servicing_fee`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `PsaModel`, `ConstantCpr`, `PrepaymentModel`,
//! `RateIncentivePrepayment`, `MbsPassThrough`, and `MbsCashflow` define the
//! core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.

/// PSA (Public Securities Association) prepayment model.
///
/// The standard benchmark ramps CPR linearly from 0% to 6% over months 1–30,
/// then holds at 6%. The `psa_speed` multiplier scales the curve
/// (1.0 = 100% PSA, 2.0 = 200% PSA, etc.).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PsaModel {
    pub psa_speed: f64,
}

impl PsaModel {
    /// Largest valid PSA multiplier. At this boundary the plateau CPR is 100%.
    pub const MAX_SPEED: f64 = 1.0 / 0.06;

    /// Validates that the complete PSA path has CPR in `[0, 1]`.
    pub fn validate(&self) -> Result<(), String> {
        if !self.psa_speed.is_finite() || !(0.0..=Self::MAX_SPEED).contains(&self.psa_speed) {
            return Err(format!(
                "psa_speed must be finite and in [0, {}]",
                Self::MAX_SPEED
            ));
        }
        Ok(())
    }

    /// Annual Conditional Prepayment Rate at the given month (1-indexed).
    pub fn cpr(&self, month: u32) -> f64 {
        // Express the plateau as speed / max_speed so the valid upper
        // boundary evaluates to exactly 100% CPR in binary64. Computing
        // 0.06 * max_speed leaves a one-ulp gap whose twelfth root produces a
        // materially incorrect SMM below 100%.
        let plateau_cpr = self.psa_speed / Self::MAX_SPEED;
        if month <= 30 {
            plateau_cpr * f64::from(month) / 30.0
        } else {
            plateau_cpr
        }
    }

    /// Single Monthly Mortality rate: SMM = 1 - (1 - CPR)^(1/12).
    pub fn smm(&self, month: u32) -> f64 {
        let cpr = self.cpr(month);
        1.0 - (1.0 - cpr).powf(1.0 / 12.0)
    }
}

/// Constant CPR prepayment model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstantCpr {
    pub annual_cpr: f64,
}

impl ConstantCpr {
    pub fn smm(&self) -> f64 {
        1.0 - (1.0 - self.annual_cpr).powf(1.0 / 12.0)
    }
}

/// Transparent refinancing-incentive prepayment model.
///
/// This uses the deterministic Office of Thrift Supervision refinancing
/// coefficients and the seasoning/seasonality factors shown by MathWorks in
/// its modified Richard--Roll example. It combines refinancing incentive,
/// seasoning, and calendar seasonality:
///
/// `CPR = refinancing_incentive * seasoning * seasonality`.
///
/// It deliberately does not claim to be a calibrated borrower model or the
/// full Richard--Roll model: in particular, it has no burnout factor. The
/// explicit refinancing-rate scenario supplied to [`MbsPassThrough`] controls
/// the rate response. MathWorks' example feeds its pass-through `CouponRate`
/// into the incentive formula; OpenFerric instead uses the gross WAC stored in
/// [`MbsPassThrough::coupon_rate`], consistently with this module's amortization
/// and servicing-fee convention.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RateIncentivePrepayment {
    /// Calendar month of the loan's first scheduled payment (January = 1).
    pub first_payment_month: u8,
}

impl RateIncentivePrepayment {
    /// Ginnie Mae 30-year single-family seasonality multipliers used by the
    /// published OTS/modified Richard--Roll example, January through December.
    pub const SEASONALITY: [f64; 12] = [
        0.94, 0.76, 0.73, 0.96, 0.98, 0.92, 0.99, 1.10, 1.18, 1.21, 1.23, 0.97,
    ];

    /// Creates a model and validates the calendar-month convention.
    pub fn new(first_payment_month: u8) -> Result<Self, String> {
        let model = Self {
            first_payment_month,
        };
        model.validate()?;
        Ok(model)
    }

    /// Validates model fields.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=12).contains(&self.first_payment_month) {
            return Err("first_payment_month must be in 1..=12".to_string());
        }
        Ok(())
    }

    /// Annual refinancing-incentive CPR before seasoning and seasonality.
    ///
    /// Both rates are decimal annual rates.  `mortgage_coupon_rate` is the
    /// MBS gross coupon/WAC and `refinancing_rate` is the contemporaneous
    /// mortgage rate available to borrowers.
    pub fn refinancing_incentive(
        mortgage_coupon_rate: f64,
        refinancing_rate: f64,
    ) -> Result<f64, String> {
        if !mortgage_coupon_rate.is_finite() || mortgage_coupon_rate < 0.0 {
            return Err("mortgage_coupon_rate must be finite and >= 0".to_string());
        }
        if !refinancing_rate.is_finite() || refinancing_rate <= 0.0 {
            return Err("refinancing_rate must be finite and > 0".to_string());
        }

        let ratio = mortgage_coupon_rate / refinancing_rate;
        Ok(0.2406 - 0.1389 * (5.952 * (1.089 - ratio)).atan())
    }

    /// Annual CPR for a 1-indexed loan-age month and refinancing rate.
    pub fn cpr(
        &self,
        loan_age_month: u32,
        mortgage_coupon_rate: f64,
        refinancing_rate: f64,
    ) -> Result<f64, String> {
        self.validate()?;
        if loan_age_month == 0 {
            return Err("loan_age_month must be >= 1".to_string());
        }

        let refinancing = Self::refinancing_incentive(mortgage_coupon_rate, refinancing_rate)?;
        let seasoning = f64::from(loan_age_month.min(30)) / 30.0;
        let age_month_offset = ((loan_age_month - 1) % 12) as usize;
        let calendar_month = (usize::from(self.first_payment_month) - 1 + age_month_offset) % 12;
        let cpr = refinancing * seasoning * Self::SEASONALITY[calendar_month];

        if !(0.0..1.0).contains(&cpr) {
            return Err("rate-incentive CPR must be in [0, 1)".to_string());
        }
        Ok(cpr)
    }

    /// Single Monthly Mortality rate for a 1-indexed loan-age month.
    pub fn smm(
        &self,
        loan_age_month: u32,
        mortgage_coupon_rate: f64,
        refinancing_rate: f64,
    ) -> Result<f64, String> {
        let cpr = self.cpr(loan_age_month, mortgage_coupon_rate, refinancing_rate)?;
        Ok(1.0 - (1.0 - cpr).powf(1.0 / 12.0))
    }
}

impl Default for RateIncentivePrepayment {
    fn default() -> Self {
        Self {
            first_payment_month: 1,
        }
    }
}

/// Prepayment model selector.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PrepaymentModel {
    Psa(PsaModel),
    ConstantCpr(ConstantCpr),
}

impl PrepaymentModel {
    /// SMM for a given month (1-indexed, where month = age + period).
    pub fn smm(&self, month: u32) -> f64 {
        match self {
            PrepaymentModel::Psa(m) => m.smm(month),
            PrepaymentModel::ConstantCpr(m) => m.smm(),
        }
    }
}

/// MBS pass-through security.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MbsPassThrough {
    /// Current pool balance at [`Self::age`].
    pub original_balance: f64,
    /// Gross mortgage coupon / weighted-average coupon (WAC).
    ///
    /// This rate drives scheduled amortization and the refinancing incentive.
    /// Investor pass-through interest uses this rate less [`Self::servicing_fee`].
    pub coupon_rate: f64,
    /// Annual servicing strip deducted from the gross WAC for investor interest.
    pub servicing_fee: f64,
    /// Original mortgage term in months.
    pub original_term: u32,
    /// Current loan age in months.
    pub age: u32,
    /// PSA or constant-CPR assumption used by the legacy, rate-independent APIs.
    pub prepayment: PrepaymentModel,
}

/// A single month's cashflow from an MBS pass-through.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MbsCashflow {
    pub month: u32,
    pub interest: f64,
    pub scheduled_principal: f64,
    pub prepayment: f64,
    pub total_principal: f64,
    pub remaining_balance: f64,
    pub total_cashflow: f64,
}

impl MbsPassThrough {
    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), String> {
        if !self.original_balance.is_finite() || self.original_balance <= 0.0 {
            return Err("mbs original_balance must be finite and > 0".to_string());
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate < 0.0 {
            return Err("mbs coupon_rate must be finite and >= 0".to_string());
        }
        if !self.servicing_fee.is_finite() || self.servicing_fee < 0.0 {
            return Err("mbs servicing_fee must be finite and >= 0".to_string());
        }
        if self.servicing_fee > self.coupon_rate {
            return Err("mbs servicing_fee must be <= coupon_rate".to_string());
        }
        if self.original_term == 0 {
            return Err("mbs original_term must be > 0".to_string());
        }
        if self.age > self.original_term {
            return Err("mbs age must be <= original_term".to_string());
        }
        match &self.prepayment {
            PrepaymentModel::Psa(m) => {
                m.validate().map_err(|error| format!("mbs {error}"))?;
            }
            PrepaymentModel::ConstantCpr(m) => {
                if !m.annual_cpr.is_finite() || !(0.0..=1.0).contains(&m.annual_cpr) {
                    return Err("mbs annual_cpr must be finite and in [0, 1]".to_string());
                }
            }
        }
        Ok(())
    }

    fn cashflows_from_smm(
        &self,
        mut smm_for_month: impl FnMut(u32, u32) -> f64,
    ) -> Vec<MbsCashflow> {
        let net_coupon = self.coupon_rate - self.servicing_fee;
        let monthly_net = net_coupon / 12.0;
        let monthly_gross = self.coupon_rate / 12.0;
        let remaining_term = self.original_term.saturating_sub(self.age);

        // Compute the current balance after seasoning via standard amortization
        // We need to simulate from the start to get the correct current balance
        // accounting for prepayments that already occurred.
        // For simplicity, assume original_balance is the *current* balance at age.
        let mut balance = self.original_balance;
        let mut result = Vec::with_capacity(remaining_term as usize);

        for period in 1..=remaining_term {
            if balance <= 0.0 {
                break;
            }

            let month = self.age + period; // seasoning month for prepayment model
            let interest = balance * monthly_net;

            // Scheduled principal: standard amortization payment minus interest (at gross rate)
            let periods_left = remaining_term - period + 1;
            let scheduled_payment = if monthly_gross > 0.0 {
                // `1 - (1 + r)^-n` loses several digits for short pools or
                // small coupons.  The log1p/expm1 form is algebraically
                // identical and remains accurate as `r` approaches zero.
                let denominator = -(-f64::from(periods_left) * monthly_gross.ln_1p()).exp_m1();
                balance * monthly_gross / denominator
            } else {
                balance / periods_left as f64
            };
            let scheduled_principal = scheduled_payment - balance * monthly_gross;

            // Prepayment on remaining balance after scheduled principal
            let smm = smm_for_month(month, period);
            let prepayment = (balance - scheduled_principal) * smm;

            let total_principal = scheduled_principal + prepayment;
            balance -= total_principal;
            if balance < 1e-10 {
                balance = 0.0;
            }

            result.push(MbsCashflow {
                month: period,
                interest,
                scheduled_principal,
                prepayment,
                total_principal,
                remaining_balance: balance,
                total_cashflow: interest + total_principal,
            });
        }

        result
    }

    /// Generate the monthly cashflow schedule.
    pub fn cashflows(&self) -> Vec<MbsCashflow> {
        if self.validate().is_err() {
            return Vec::new();
        }

        self.cashflows_from_smm(|month, _period| self.prepayment.smm(month))
    }

    fn validate_rate_curve(
        rates: &[f64],
        remaining_term: usize,
        name: &str,
        require_positive: bool,
    ) -> Result<(), String> {
        let valid_length = if remaining_term == 0 {
            rates.is_empty() || rates.len() == 1
        } else {
            rates.len() == 1 || rates.len() == remaining_term
        };
        if !valid_length {
            return Err(format!(
                "{name} must contain one value or one value per remaining payment ({remaining_term})"
            ));
        }
        for &rate in rates {
            if !rate.is_finite() {
                return Err(format!("{name} values must be finite"));
            }
            if require_positive && rate <= 0.0 {
                return Err(format!("{name} values must be > 0"));
            }
            if !require_positive && 1.0 + rate / 12.0 <= 0.0 {
                return Err(format!(
                    "{name} values must produce a positive monthly discount base"
                ));
            }
        }
        Ok(())
    }

    fn curve_value(rates: &[f64], period: usize) -> f64 {
        if rates.len() == 1 {
            rates[0]
        } else {
            rates[period]
        }
    }

    /// Generate cashflows under an explicit monthly refinancing-rate scenario.
    ///
    /// `refinancing_rates` contains either one annual rate (broadcast to every
    /// remaining month) or one rate per remaining payment.  This method uses
    /// the MBS gross [`Self::coupon_rate`] as the mortgage coupon and replaces
    /// the instrument's embedded PSA/constant-CPR assumption for this scenario.
    pub fn cashflows_with_refinancing_rates(
        &self,
        model: &RateIncentivePrepayment,
        refinancing_rates: &[f64],
    ) -> Result<Vec<MbsCashflow>, String> {
        self.validate()?;
        model.validate()?;
        let remaining_term = self.original_term.saturating_sub(self.age) as usize;
        Self::validate_rate_curve(refinancing_rates, remaining_term, "refinancing_rates", true)?;
        if remaining_term == 0 {
            return Ok(Vec::new());
        }

        Ok(self.cashflows_from_smm(|month, period| {
            let rate = Self::curve_value(refinancing_rates, period as usize - 1);
            // Inputs were fully validated above, so this cannot fail.
            model
                .smm(month, self.coupon_rate, rate)
                .expect("validated rate-incentive prepayment inputs")
        }))
    }

    /// Present value of a pre-computed cashflow schedule at a flat yield.
    ///
    /// The prepayment-driven schedule does not depend on the discount rate,
    /// so re-discounting a cached schedule avoids rebuilding it in bump and
    /// root-finding loops.
    fn pv_of_cashflows(cfs: &[MbsCashflow], yield_rate: f64) -> f64 {
        let monthly_yield = yield_rate / 12.0;
        cfs.iter()
            .map(|cf| cf.total_cashflow * (1.0 + monthly_yield).powi(-(cf.month as i32)))
            .sum()
    }

    fn pv_of_cashflows_with_yield_curve(cfs: &[MbsCashflow], discount_yields: &[f64]) -> f64 {
        cfs.iter()
            .enumerate()
            .map(|(period, cf)| {
                let annual_yield = Self::curve_value(discount_yields, period);
                cf.total_cashflow * (1.0 + annual_yield / 12.0).powi(-(cf.month as i32))
            })
            .sum()
    }

    /// Dollar price given an annual nominal yield compounded monthly.
    ///
    /// Returns `NaN` for invalid instrument definitions (see [`Self::validate`]).
    pub fn price(&self, yield_rate: f64) -> f64 {
        if self.validate().is_err() || !yield_rate.is_finite() || yield_rate <= -12.0 {
            return f64::NAN;
        }
        Self::pv_of_cashflows(&self.cashflows(), yield_rate)
    }

    /// Price under explicit refinancing-rate and discount-yield scenarios.
    ///
    /// Each curve contains either one annual nominal rate (broadcast) or one
    /// rate per remaining monthly payment.  Discount-yield entry `i` is the
    /// annual nominal spot yield for payment month `i`, compounded monthly.
    pub fn price_with_rate_scenario(
        &self,
        model: &RateIncentivePrepayment,
        refinancing_rates: &[f64],
        discount_yields: &[f64],
    ) -> Result<f64, String> {
        let cfs = self.cashflows_with_refinancing_rates(model, refinancing_rates)?;
        let remaining_term = self.original_term.saturating_sub(self.age) as usize;
        Self::validate_rate_curve(discount_yields, remaining_term, "discount_yields", false)?;
        if remaining_term == 0 {
            return Ok(0.0);
        }
        Ok(Self::pv_of_cashflows_with_yield_curve(
            &cfs,
            discount_yields,
        ))
    }

    /// Weighted Average Life under an explicit refinancing-rate scenario.
    pub fn wal_with_refinancing_rates(
        &self,
        model: &RateIncentivePrepayment,
        refinancing_rates: &[f64],
    ) -> Result<f64, String> {
        let cfs = self.cashflows_with_refinancing_rates(model, refinancing_rates)?;
        let total_principal: f64 = cfs.iter().map(|c| c.total_principal).sum();
        if total_principal == 0.0 {
            return Ok(0.0);
        }
        let weighted: f64 = cfs
            .iter()
            .map(|c| c.total_principal * f64::from(c.month) / 12.0)
            .sum();
        Ok(weighted / total_principal)
    }

    /// Weighted Average Life in years.
    ///
    /// Returns `NaN` for invalid instrument definitions.
    pub fn wal(&self) -> f64 {
        if self.validate().is_err() {
            return f64::NAN;
        }
        let cfs = self.cashflows();
        let total_principal: f64 = cfs.iter().map(|c| c.total_principal).sum();
        if total_principal == 0.0 {
            return 0.0;
        }
        let weighted: f64 = cfs
            .iter()
            .map(|c| c.total_principal * c.month as f64 / 12.0)
            .sum();
        weighted / total_principal
    }

    /// Constant yield spread over a flat or monthly spot-yield term structure.
    /// `market_price` is a dollar price. `base_yields` contains one annual nominal
    /// yield or one per remaining payment, compounded monthly; an empty slice
    /// denotes a zero base curve. Every supplied curve point is used.
    /// Cashflows use the deterministic prepayment assumption, so this is a
    /// deterministic spread solve, not a stochastic option-adjusted valuation.
    ///
    /// Returns `NaN` for invalid instrument definitions.
    pub fn oas(&self, market_price: f64, base_yields: &[f64]) -> f64 {
        if self.validate().is_err() || !market_price.is_finite() || market_price <= 0.0 {
            return f64::NAN;
        }
        let base_yields = if base_yields.is_empty() {
            &[0.0]
        } else {
            base_yields
        };
        let remaining_term = self.original_term.saturating_sub(self.age) as usize;
        if Self::validate_rate_curve(base_yields, remaining_term, "base_yields", false).is_err() {
            return f64::NAN;
        }
        let cashflows = self.cashflows();
        if cashflows.is_empty() {
            return f64::NAN;
        }
        let price = |spread: f64| -> f64 {
            cashflows
                .iter()
                .enumerate()
                .map(|(index, cashflow)| {
                    let yield_rate = Self::curve_value(base_yields, index) + spread;
                    cashflow.total_cashflow
                        * (1.0 + yield_rate / 12.0).powi(-(cashflow.month as i32))
                })
                .sum()
        };
        let minimum_yield = base_yields.iter().copied().fold(f64::INFINITY, f64::min);
        let mut lower = -12.0 - minimum_yield + 1.0e-12;
        let mut upper = 1.0;
        for _ in 0..64 {
            if price(upper) <= market_price {
                break;
            }
            upper *= 2.0;
        }
        if price(lower) < market_price || price(upper) > market_price {
            return f64::NAN;
        }
        for _ in 0..200 {
            let middle = 0.5 * (lower + upper);
            if middle == lower || middle == upper {
                break;
            }
            if price(middle) > market_price {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        0.5 * (lower + upper)
    }

    /// Effective duration via numerical bump (parallel shift).
    ///
    /// Returns `NaN` for invalid instrument definitions.
    pub fn effective_duration(&self, yield_rate: f64) -> f64 {
        if self.validate().is_err() || !yield_rate.is_finite() || yield_rate - 0.0001 <= -12.0 {
            return f64::NAN;
        }
        let cfs = self.cashflows();
        let bump = 0.0001; // 1bp
        let p_up = Self::pv_of_cashflows(&cfs, yield_rate + bump);
        let p_down = Self::pv_of_cashflows(&cfs, yield_rate - bump);
        let p0 = Self::pv_of_cashflows(&cfs, yield_rate);
        (p_down - p_up) / (2.0 * bump * p0)
    }

    /// Effective duration for a parallel market-rate scenario shift.
    ///
    /// Both discount yields and refinancing rates are shifted, and the
    /// prepayment cashflows are rebuilt for each side of the bump.  This is a
    /// deterministic scenario duration, not a stochastic OAS model.
    pub fn effective_duration_with_rate_scenario(
        &self,
        model: &RateIncentivePrepayment,
        refinancing_rates: &[f64],
        discount_yields: &[f64],
        bump: f64,
    ) -> Result<f64, String> {
        if !bump.is_finite() || bump <= 0.0 {
            return Err("bump must be finite and > 0".to_string());
        }

        let p0 = self.price_with_rate_scenario(model, refinancing_rates, discount_yields)?;
        if !p0.is_finite() || p0 <= 0.0 {
            return Err("unshifted scenario price must be finite and > 0".to_string());
        }

        let refinancing_up: Vec<f64> = refinancing_rates.iter().map(|r| r + bump).collect();
        let refinancing_down: Vec<f64> = refinancing_rates.iter().map(|r| r - bump).collect();
        let discount_up: Vec<f64> = discount_yields.iter().map(|r| r + bump).collect();
        let discount_down: Vec<f64> = discount_yields.iter().map(|r| r - bump).collect();
        let p_up = self.price_with_rate_scenario(model, &refinancing_up, &discount_up)?;
        let p_down = self.price_with_rate_scenario(model, &refinancing_down, &discount_down)?;

        Ok((p_down - p_up) / (2.0 * bump * p0))
    }
}

/// Interest-Only strip.
pub struct IoStrip<'a> {
    pub mbs: &'a MbsPassThrough,
}

impl<'a> IoStrip<'a> {
    pub fn cashflows(&self) -> Vec<(u32, f64)> {
        self.mbs
            .cashflows()
            .iter()
            .map(|c| (c.month, c.interest))
            .collect()
    }

    pub fn price(&self, yield_rate: f64) -> f64 {
        if self.mbs.validate().is_err() || !yield_rate.is_finite() || yield_rate <= -12.0 {
            return f64::NAN;
        }
        let monthly_yield = yield_rate / 12.0;
        self.cashflows()
            .iter()
            .map(|(m, cf)| cf / (1.0 + monthly_yield).powi(*m as i32))
            .sum()
    }
}

/// Principal-Only strip.
pub struct PoStrip<'a> {
    pub mbs: &'a MbsPassThrough,
}

impl<'a> PoStrip<'a> {
    pub fn cashflows(&self) -> Vec<(u32, f64)> {
        self.mbs
            .cashflows()
            .iter()
            .map(|c| (c.month, c.total_principal))
            .collect()
    }

    pub fn price(&self, yield_rate: f64) -> f64 {
        if self.mbs.validate().is_err() || !yield_rate.is_finite() || yield_rate <= -12.0 {
            return f64::NAN;
        }
        let monthly_yield = yield_rate / 12.0;
        self.cashflows()
            .iter()
            .map(|(m, cf)| cf / (1.0 + monthly_yield).powi(*m as i32))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_psa_cpr_month1() {
        let psa = PsaModel { psa_speed: 1.0 };
        let cpr = psa.cpr(1);
        assert_relative_eq!(cpr, 0.002, epsilon = 1.0e-15);
    }

    #[test]
    fn test_psa_smm_month1() {
        let psa = PsaModel { psa_speed: 1.0 };
        let smm = psa.smm(1);
        assert_relative_eq!(smm, 0.000_166_819_639_945_581_24, epsilon = 1.0e-15);
    }

    #[test]
    fn test_psa_cpr_month15() {
        let psa = PsaModel { psa_speed: 1.0 };
        assert_relative_eq!(psa.cpr(15), 0.03, epsilon = 1.0e-15);
    }

    #[test]
    fn test_psa_smm_month15() {
        let psa = PsaModel { psa_speed: 1.0 };
        let smm = psa.smm(15);
        assert_relative_eq!(smm, 0.002_535_048_613_836_688, epsilon = 1.0e-15);
    }

    #[test]
    fn test_psa_cpr_month30_plus() {
        let psa = PsaModel { psa_speed: 1.0 };
        assert_relative_eq!(psa.cpr(30), 0.06, epsilon = 1.0e-15);
        assert_relative_eq!(psa.cpr(100), 0.06, epsilon = 1.0e-15);
    }

    #[test]
    fn test_psa_smm_month30() {
        let psa = PsaModel { psa_speed: 1.0 };
        let smm = psa.smm(30);
        assert_relative_eq!(smm, 0.005_143_012_831_822_946, epsilon = 1.0e-15);
    }

    #[test]
    fn test_200psa_month15() {
        let psa = PsaModel { psa_speed: 2.0 };
        assert_relative_eq!(psa.cpr(15), 0.06, epsilon = 1.0e-15);
    }

    #[test]
    fn test_300psa_month30() {
        let psa = PsaModel { psa_speed: 3.0 };
        assert_relative_eq!(psa.cpr(30), 0.18, epsilon = 1.0e-15);
    }

    #[test]
    fn psa_speed_boundary_is_finite_and_larger_speeds_are_rejected() {
        let boundary = PsaModel {
            psa_speed: PsaModel::MAX_SPEED,
        };
        assert!(boundary.validate().is_ok());
        assert_relative_eq!(boundary.cpr(30), 1.0, epsilon = 2.0e-16);
        assert_relative_eq!(boundary.smm(30), 1.0, epsilon = 2.0e-16);

        let boundary_mbs = make_mbs(PrepaymentModel::Psa(boundary));
        assert!(boundary_mbs.validate().is_ok());
        assert!(boundary_mbs.price(0.05).is_finite());
        assert!(
            boundary_mbs
                .cashflows()
                .iter()
                .all(|cashflow| cashflow.total_cashflow.is_finite())
        );

        let invalid = PsaModel { psa_speed: 20.0 };
        assert!(invalid.validate().is_err());
        let invalid_mbs = make_mbs(PrepaymentModel::Psa(invalid));
        assert!(invalid_mbs.validate().is_err());
        assert!(invalid_mbs.cashflows().is_empty());
        assert!(invalid_mbs.price(0.05).is_nan());
    }

    #[test]
    fn rate_incentive_matches_ots_reference_formula_exactly() {
        // Independently evaluated with 70-digit Decimal arithmetic.  The
        // formula and coefficients are the OTS values reproduced in the
        // MathWorks modified Richard--Roll example.
        let cases = [
            (0.8, 0.095_560_426_948_413_43),
            (1.0, 0.172_935_392_137_225_9),
            (1.089, 0.240_6),
            (1.2, 0.321_695_509_253_931_6),
            (1.5, 0.404_882_508_835_867_8),
        ];
        for (coupon_to_refi_ratio, expected) in cases {
            let actual =
                RateIncentivePrepayment::refinancing_incentive(coupon_to_refi_ratio, 1.0).unwrap();
            assert_relative_eq!(actual, expected, epsilon = 2.0e-16);
        }

        let model = RateIncentivePrepayment::default();
        assert_relative_eq!(
            model.cpr(30, 0.06, 0.045).unwrap(),
            0.345_104_608_673_138_1,
            epsilon = 3.0e-16
        );
        assert_relative_eq!(
            model.smm(30, 0.06, 0.045).unwrap(),
            0.034_658_460_837_010_67,
            epsilon = 3.0e-16
        );
    }

    fn make_mbs(prepayment: PrepaymentModel) -> MbsPassThrough {
        MbsPassThrough {
            original_balance: 100.0,
            coupon_rate: 0.06,
            servicing_fee: 0.0,
            original_term: 360,
            age: 0,
            prepayment,
        }
    }

    #[test]
    fn rate_incentive_flat_scenario_matches_decimal_cashflow_reference() {
        let mut mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        mbs.servicing_fee = 0.005;
        let model = RateIncentivePrepayment::default();

        let cfs = mbs
            .cashflows_with_refinancing_rates(&model, &[0.045])
            .unwrap();
        assert_eq!(cfs.len(), 360);
        assert_relative_eq!(
            cfs[0].scheduled_principal,
            0.099_550_525_152_752_4,
            epsilon = 2.0e-14
        );
        assert_relative_eq!(
            cfs[0].prepayment,
            0.098_379_959_297_010_2,
            epsilon = 2.0e-14
        );
        assert_relative_eq!(
            cfs[29].prepayment,
            2.010_471_950_656_109_7,
            epsilon = 3.0e-13
        );

        let price = mbs
            .price_with_rate_scenario(&model, &[0.045], &[0.05])
            .unwrap();
        let wal = mbs.wal_with_refinancing_rates(&model, &[0.045]).unwrap();
        let duration = mbs
            .effective_duration_with_rate_scenario(&model, &[0.045], &[0.05], 0.0001)
            .unwrap();

        assert_relative_eq!(price, 101.449_723_054_539_36, epsilon = 2.0e-12);
        assert_relative_eq!(wal, 3.248_112_193_095_725, epsilon = 2.0e-13);
        assert_relative_eq!(duration, 2.688_623_246_183_667_6, epsilon = 3.0e-11);
    }

    #[test]
    fn rate_incentive_nonflat_scenario_matches_decimal_reference() {
        let mut mbs = make_mbs(PrepaymentModel::ConstantCpr(ConstantCpr {
            annual_cpr: 0.0,
        }));
        mbs.servicing_fee = 0.005;
        let model = RateIncentivePrepayment::default();
        let refinancing_rates: Vec<f64> = (1..=360)
            .map(|month| {
                if month <= 120 {
                    0.065 - 0.025 * f64::from(month) / 120.0
                } else {
                    0.04 + 0.015 * f64::from(month - 120) / 240.0
                }
            })
            .collect();
        let discount_yields: Vec<f64> = (1..=360)
            .map(|month| 0.04 + 0.01 * f64::from(month) / 360.0)
            .collect();

        // Full independent Decimal recurrence using the same explicitly
        // documented rate paths (not values produced by this Rust engine).
        let price = mbs
            .price_with_rate_scenario(&model, &refinancing_rates, &discount_yields)
            .unwrap();
        let wal = mbs
            .wal_with_refinancing_rates(&model, &refinancing_rates)
            .unwrap();
        assert_relative_eq!(price, 105.347_972_378_988_2, epsilon = 3.0e-12);
        assert_relative_eq!(wal, 4.609_516_586_471_395, epsilon = 3.0e-13);
    }

    #[test]
    fn aged_pool_rollover_and_curve_indices_match_decimal_reference() {
        let mbs = MbsPassThrough {
            original_balance: 100.0,
            coupon_rate: 0.06,
            servicing_fee: 0.005,
            original_term: 15,
            age: 10,
            prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.0 }),
        };
        let model = RateIncentivePrepayment::default();
        let refinancing_rates = [0.07, 0.06, 0.05, 0.04, 0.03];
        let discount_yields = [0.03, 0.035, 0.04, 0.045, 0.05];

        // The first three remaining payments use loan-age months 11, 12 and
        // 13. With a January first-payment month, their calendar multipliers
        // are November, December and January respectively.
        assert_relative_eq!(
            model.cpr(11, 0.06, refinancing_rates[0]).unwrap(),
            0.049_391_648_067_763_375,
            epsilon = 2.0e-16
        );
        assert_relative_eq!(
            model.cpr(12, 0.06, refinancing_rates[1]).unwrap(),
            0.067_098_932_149_243_65,
            epsilon = 2.0e-16
        );
        assert_relative_eq!(
            model.cpr(13, 0.06, refinancing_rates[2]).unwrap(),
            0.131_037_304_102_768_13,
            epsilon = 3.0e-16
        );

        let cashflows = mbs
            .cashflows_with_refinancing_rates(&model, &refinancing_rates)
            .unwrap();
        assert_eq!(
            cashflows
                .iter()
                .map(|cashflow| cashflow.month)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        // Independent 80-digit Decimal recurrence. Distinct rates in every
        // remaining month make any age or rate-vector off-by-one visible.
        let expected_prepayments = [
            0.337_814_060_114_554_9,
            0.346_537_264_592_425_1,
            0.464_269_126_484_602_1,
            0.253_696_067_603_719_95,
            0.0,
        ];
        let expected_balances = [
            79.861_188_442_335_63,
            59.698_471_438_949_34,
            39.433_878_454_802_085,
            19.512_412_584_304_557,
            0.0,
        ];
        for ((cashflow, expected_prepayment), expected_balance) in cashflows
            .iter()
            .zip(expected_prepayments)
            .zip(expected_balances)
        {
            assert_relative_eq!(cashflow.prepayment, expected_prepayment, epsilon = 3.0e-14);
            assert_relative_eq!(
                cashflow.remaining_balance,
                expected_balance,
                epsilon = 8.0e-14
            );
        }
        assert_relative_eq!(
            cashflows[0].interest,
            0.458_333_333_333_333_3,
            epsilon = 2.0e-16
        );
        assert_relative_eq!(
            cashflows[0].scheduled_principal,
            19.800_997_497_549_81,
            epsilon = 2.0e-14
        );

        let price = mbs
            .price_with_rate_scenario(&model, &refinancing_rates, &discount_yields)
            .unwrap();
        let wal = mbs
            .wal_with_refinancing_rates(&model, &refinancing_rates)
            .unwrap();
        assert_relative_eq!(price, 100.291_494_624_406_17, epsilon = 3.0e-14);
        assert_relative_eq!(wal, 0.248_754_959_100_326_35, epsilon = 3.0e-15);
    }

    #[test]
    fn tiny_coupon_annuity_denominator_stays_stable() {
        let mbs = MbsPassThrough {
            original_balance: 100.0,
            coupon_rate: 1.2e-11,
            servicing_fee: 0.0,
            original_term: 2,
            age: 0,
            prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.0 }),
        };

        let cashflows = mbs.cashflows();
        let monthly_coupon = mbs.coupon_rate / 12.0;
        // For a two-payment annuity the first scheduled principal simplifies
        // exactly to B / (2 + r).  The direct power-difference formula loses
        // roughly 4.4e-3 here through cancellation.
        let expected_first_principal = mbs.original_balance / (2.0 + monthly_coupon);
        assert_relative_eq!(
            cashflows[0].scheduled_principal,
            expected_first_principal,
            epsilon = 2.0e-14
        );
        assert_relative_eq!(
            cashflows
                .iter()
                .map(|cashflow| cashflow.total_principal)
                .sum::<f64>(),
            mbs.original_balance,
            epsilon = 2.0e-14
        );
    }

    #[test]
    fn gross_wac_drives_amortization_and_incentive_while_interest_is_net() {
        let mbs = MbsPassThrough {
            original_balance: 100.0,
            coupon_rate: 0.06,
            servicing_fee: 0.005,
            original_term: 360,
            age: 0,
            prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.0 }),
        };
        let model = RateIncentivePrepayment::default();
        let cashflow = &mbs
            .cashflows_with_refinancing_rates(&model, &[0.045])
            .unwrap()[0];

        let gross_wac_smm = model.smm(1, 0.06, 0.045).unwrap();
        let pass_through_coupon_smm = model.smm(1, 0.055, 0.045).unwrap();
        assert_relative_eq!(
            cashflow.interest,
            100.0 * (0.06 - 0.005) / 12.0,
            epsilon = 2.0e-16
        );
        assert_relative_eq!(
            cashflow.prepayment,
            (100.0 - cashflow.scheduled_principal) * gross_wac_smm,
            epsilon = 2.0e-15
        );
        assert!(
            (cashflow.prepayment
                - (100.0 - cashflow.scheduled_principal) * pass_through_coupon_smm)
                .abs()
                > 0.01
        );
    }

    #[test]
    fn rate_incentive_scenario_validates_inputs_and_responds_to_rates() {
        assert!(RateIncentivePrepayment::new(0).is_err());
        assert!(RateIncentivePrepayment::new(13).is_err());
        let model = RateIncentivePrepayment::new(1).unwrap();
        assert!(model.cpr(0, 0.06, 0.05).is_err());
        assert!(model.cpr(1, 0.06, 0.0).is_err());

        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        assert!(mbs.cashflows_with_refinancing_rates(&model, &[]).is_err());
        assert!(
            mbs.cashflows_with_refinancing_rates(&model, &[0.05, 0.04])
                .is_err()
        );
        assert!(mbs.price_with_rate_scenario(&model, &[0.05], &[]).is_err());
        assert!(
            mbs.effective_duration_with_rate_scenario(&model, &[0.05], &[0.05], 0.0)
                .is_err()
        );

        let fast_wal = mbs.wal_with_refinancing_rates(&model, &[0.04]).unwrap();
        let slow_wal = mbs.wal_with_refinancing_rates(&model, &[0.075]).unwrap();
        assert!(fast_wal < slow_wal);
    }

    #[test]
    fn test_wal_no_prepayment() {
        let mbs = make_mbs(PrepaymentModel::ConstantCpr(ConstantCpr {
            annual_cpr: 0.0,
        }));
        let wal = mbs.wal();
        // Independent monthly-amortization spreadsheet reference: level 6%
        // mortgage payment, principal-weighted payment month / 12.
        assert_relative_eq!(wal, 19.306_364_842_498_26, epsilon = 1.0e-10);
    }

    #[test]
    fn test_wal_100psa() {
        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        let wal = mbs.wal();
        assert_relative_eq!(wal, 11.363_476_874_606_265, epsilon = 1.0e-10);
    }

    #[test]
    fn test_wal_300psa() {
        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 3.0 }));
        let wal = mbs.wal();
        assert_relative_eq!(wal, 5.736_289_578_189_329_5, epsilon = 1.0e-10);
    }

    #[test]
    fn test_no_prepayment_price_matches_amortizing_mortgage_reference() {
        let mbs = make_mbs(PrepaymentModel::ConstantCpr(ConstantCpr {
            annual_cpr: 0.0,
        }));

        // At the 6% mortgage rate, level-payment principal plus interest
        // discounts exactly to par. The 5% value is an independently generated
        // monthly cashflow reference.
        assert_relative_eq!(mbs.price(0.06), 100.0, epsilon = 3.0e-12);
        assert_relative_eq!(mbs.price(0.05), 111.685_241_326_278_59, epsilon = 1.0e-10);
    }

    #[test]
    fn io_and_po_strips_match_independent_decimal_cashflows() {
        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.5 }));
        let yield_rate = 0.05;
        let whole = mbs.price(yield_rate);
        let io = IoStrip { mbs: &mbs }.price(yield_rate);
        let po = PoStrip { mbs: &mbs }.price(yield_rate);

        // Independent 80-digit Decimal monthly amortization, 150% PSA/SMM,
        // and 5% nominal-yield discounting. Pin each strip separately so an
        // interest/principal allocation error cannot hide behind IO+PO parity.
        assert_relative_eq!(io, 40.158_863_626_459_05, epsilon = 3.0e-12);
        assert_relative_eq!(po, 66.534_280_311_284_13, epsilon = 3.0e-12);
        assert_relative_eq!(whole, 106.693_143_937_743_18, epsilon = 3.0e-12);

        // Supplemental cashflow partition identity.
        assert_relative_eq!(io + po, whole, epsilon = 1.0e-12);
    }

    #[test]
    fn test_effective_duration_positive() {
        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        let dur = mbs.effective_duration(0.06);
        assert_relative_eq!(dur, 7.364_726_342_852_562, epsilon = 1.0e-10);
    }

    #[test]
    fn test_higher_psa_shorter_duration() {
        let mbs1 = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        let mbs3 = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 3.0 }));
        let d1 = mbs1.effective_duration(0.06);
        let d3 = mbs3.effective_duration(0.06);
        assert!(
            d1 > d3,
            "Higher PSA should have shorter duration: {} vs {}",
            d1,
            d3
        );
    }

    #[test]
    fn test_validate_rejects_age_beyond_term_without_panicking() {
        let mut mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        mbs.original_term = 360;
        mbs.age = 400;

        assert!(mbs.validate().is_err());
        // Entry points must not panic (previously underflowed u32 in debug)
        // and must signal the invalid input.
        assert!(mbs.price(0.05).is_nan());
        assert!(mbs.wal().is_nan());
        assert!(mbs.effective_duration(0.05).is_nan());
        assert!(mbs.oas(100.0, &[0.05]).is_nan());
        assert!(mbs.cashflows().is_empty());
    }

    #[test]
    fn test_validate_rejects_bad_fields() {
        let base = make_mbs(PrepaymentModel::ConstantCpr(ConstantCpr {
            annual_cpr: 0.06,
        }));
        assert!(base.validate().is_ok());

        let mut bad = base.clone();
        bad.original_balance = f64::NAN;
        assert!(bad.validate().is_err());

        let mut bad = base.clone();
        bad.original_balance = -1.0;
        assert!(bad.validate().is_err());

        let mut bad = base.clone();
        bad.coupon_rate = f64::NAN;
        assert!(bad.validate().is_err());

        let mut bad = base.clone();
        bad.servicing_fee = bad.coupon_rate + 0.01;
        assert!(bad.validate().is_err());

        let mut bad = base.clone();
        bad.original_term = 0;
        assert!(bad.validate().is_err());

        let mut bad = base.clone();
        bad.prepayment = PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 1.5 });
        assert!(bad.validate().is_err());

        let mut bad = base;
        bad.prepayment = PrepaymentModel::Psa(PsaModel {
            psa_speed: f64::NAN,
        });
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_oas_roundtrip() {
        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.0 }));
        let base_rate = 0.05;
        let true_spread = 0.01;
        let market_price = mbs.price(base_rate + true_spread);
        let computed_oas = mbs.oas(market_price, &[base_rate]);
        assert_relative_eq!(computed_oas, true_spread, epsilon = 2.0e-9);
    }
}
