//! Module `instruments::mbs`.
//!
//! Implements mbs abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `PsaModel`, `ConstantCpr`, `PrepaymentModel`, `MbsPassThrough`, `MbsCashflow` define the core data contracts for this module.
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
    /// Annual Conditional Prepayment Rate at the given month (1-indexed).
    pub fn cpr(&self, month: u32) -> f64 {
        let base_cpr = if month <= 30 {
            0.06 * (month as f64) / 30.0
        } else {
            0.06
        };
        base_cpr * self.psa_speed
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
    pub original_balance: f64,
    pub coupon_rate: f64,
    pub servicing_fee: f64,
    pub original_term: u32,
    pub age: u32,
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
                if !m.psa_speed.is_finite() || m.psa_speed < 0.0 {
                    return Err("mbs psa_speed must be finite and >= 0".to_string());
                }
            }
            PrepaymentModel::ConstantCpr(m) => {
                if !m.annual_cpr.is_finite() || !(0.0..=1.0).contains(&m.annual_cpr) {
                    return Err("mbs annual_cpr must be finite and in [0, 1]".to_string());
                }
            }
        }
        Ok(())
    }

    /// Generate the monthly cashflow schedule.
    pub fn cashflows(&self) -> Vec<MbsCashflow> {
        if self.validate().is_err() {
            return Vec::new();
        }

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
                balance * monthly_gross / (1.0 - (1.0 + monthly_gross).powi(-(periods_left as i32)))
            } else {
                balance / periods_left as f64
            };
            let scheduled_principal = scheduled_payment - balance * monthly_gross;

            // Prepayment on remaining balance after scheduled principal
            let smm = self.prepayment.smm(month);
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

    /// Price given a constant discount rate (yield), as a fraction of current balance.
    ///
    /// Returns `NaN` for invalid instrument definitions (see [`Self::validate`]).
    pub fn price(&self, yield_rate: f64) -> f64 {
        if self.validate().is_err() {
            return f64::NAN;
        }
        Self::pv_of_cashflows(&self.cashflows(), yield_rate)
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

    /// Option-Adjusted Spread via Newton's method.
    /// `market_price` is the dollar price, `base_yields` is a flat or term-structure
    /// of monthly discount rates. For simplicity we use `base_yields[0]` as the flat rate.
    ///
    /// Returns `NaN` for invalid instrument definitions.
    pub fn oas(&self, market_price: f64, base_yields: &[f64]) -> f64 {
        if self.validate().is_err() {
            return f64::NAN;
        }
        let base_rate = if base_yields.is_empty() {
            0.0
        } else {
            base_yields[0]
        };

        // The schedule is rate-independent: compute it once and re-discount
        // inside the Newton iteration instead of rebuilding it per evaluation.
        let cfs = self.cashflows();

        // Find spread s such that price(base_rate + s) = market_price
        let mut spread = 0.01; // initial guess
        for _ in 0..200 {
            let p = Self::pv_of_cashflows(&cfs, base_rate + spread);
            let dp = Self::pv_of_cashflows(&cfs, base_rate + spread + 0.0001);
            let deriv = (dp - p) / 0.0001;
            if deriv.abs() < 1e-20 {
                break;
            }
            let new_spread = spread - (p - market_price) / deriv;
            if (new_spread - spread).abs() < 1e-10 {
                spread = new_spread;
                break;
            }
            spread = new_spread;
        }
        spread
    }

    /// Effective duration via numerical bump (parallel shift).
    ///
    /// Returns `NaN` for invalid instrument definitions.
    pub fn effective_duration(&self, yield_rate: f64) -> f64 {
        if self.validate().is_err() {
            return f64::NAN;
        }
        let cfs = self.cashflows();
        let bump = 0.0001; // 1bp
        let p_up = Self::pv_of_cashflows(&cfs, yield_rate + bump);
        let p_down = Self::pv_of_cashflows(&cfs, yield_rate - bump);
        let p0 = Self::pv_of_cashflows(&cfs, yield_rate);
        (p_down - p_up) / (2.0 * bump * p0)
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
    fn test_io_po_equals_whole() {
        let mbs = make_mbs(PrepaymentModel::Psa(PsaModel { psa_speed: 1.5 }));
        let yield_rate = 0.05;
        let whole = mbs.price(yield_rate);
        let io = IoStrip { mbs: &mbs }.price(yield_rate);
        let po = PoStrip { mbs: &mbs }.price(yield_rate);
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
