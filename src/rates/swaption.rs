//! Module `rates::swaption`.
//!
//! Implements swaption abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `Swaption` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
use crate::core::OptionType;
use crate::engines::analytic::black_scholes::bs_price;
use crate::rates::YieldCurve;

/// European swaption on a forward-starting fixed-for-floating swap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swaption {
    pub notional: f64,
    pub strike: f64,
    pub option_expiry: f64,
    pub swap_tenor: f64,
    pub is_payer: bool,
}

impl Swaption {
    /// Swap annuity factor `A = sum(DF_i * delta_i)` for annual fixed payments.
    pub fn annuity_factor(&self, curve: &YieldCurve) -> f64 {
        if !self.option_expiry.is_finite()
            || !self.swap_tenor.is_finite()
            || self.option_expiry < 0.0
            || self.swap_tenor <= 0.0
        {
            return 0.0;
        }

        let start = self.option_expiry;
        let end = start + self.swap_tenor;
        let mut prev = start;
        let mut annuity = 0.0;

        loop {
            let next = (prev + 1.0).min(end);
            if next <= prev {
                break;
            }

            let delta = next - prev;
            annuity += delta * curve.discount_factor(next);

            if next >= end - 1.0e-12 {
                break;
            }
            prev = next;
        }

        annuity
    }

    /// Forward par swap rate for the underlying forward-starting swap.
    pub fn forward_swap_rate(&self, curve: &YieldCurve) -> f64 {
        if self.option_expiry < 0.0 || self.swap_tenor <= 0.0 {
            return f64::NAN;
        }

        let start = self.option_expiry;
        let end = start + self.swap_tenor;
        let annuity = self.annuity_factor(curve);
        if annuity <= 0.0 {
            return f64::NAN;
        }

        let df_start = curve.discount_factor(start);
        let df_end = curve.discount_factor(end);
        (df_start - df_end) / annuity
    }

    /// Black-76 payer/receiver swaption price.
    pub fn price(&self, curve: &YieldCurve, vol: f64) -> f64 {
        if !self.notional.is_finite()
            || self.notional < 0.0
            || !self.strike.is_finite()
            || self.strike < 0.0
            || !vol.is_finite()
            || vol < 0.0
        {
            return f64::NAN;
        }

        let annuity = self.annuity_factor(curve);
        if annuity <= 0.0 {
            return f64::NAN;
        }

        let forward = self.forward_swap_rate(curve);
        if !forward.is_finite() || forward < 0.0 {
            return f64::NAN;
        }

        if self.notional == 0.0 || (forward == 0.0 && self.strike == 0.0) {
            return 0.0;
        }
        let option_type = if self.is_payer {
            OptionType::Call
        } else {
            OptionType::Put
        };
        self.notional
            * annuity
            * bs_price(
                option_type,
                forward,
                self.strike,
                0.0,
                0.0,
                vol,
                self.option_expiry,
            )
    }

    /// Implied Black volatility from market swaption price.
    pub fn implied_vol(&self, market_price: f64, curve: &YieldCurve) -> f64 {
        if !market_price.is_finite() || market_price < 0.0 {
            return f64::NAN;
        }

        let intrinsic = self.price(curve, 0.0);
        if !intrinsic.is_finite() {
            return f64::NAN;
        }
        if market_price < intrinsic {
            return f64::NAN;
        }
        if market_price == intrinsic {
            return 0.0;
        }

        let mut lo = 0.0;
        let mut hi = 5.0;
        let mut flo = self.price(curve, lo) - market_price;
        let fhi = self.price(curve, hi) - market_price;

        if !flo.is_finite() || !fhi.is_finite() || flo * fhi > 0.0 {
            return f64::NAN;
        }

        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let fm = self.price(curve, mid) - market_price;
            if !fm.is_finite() {
                return f64::NAN;
            }
            if fm.abs() <= 1.0e-10 {
                return mid;
            }

            if flo * fm <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
                flo = fm;
            }
        }

        0.5 * (lo + hi)
    }
}
