//! Module `rates::futures`.
//!
//! Implements futures abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `Future`, `InterestRateFutureQuote` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
/// Cost-of-carry futures contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Future {
    pub underlying_spot: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub storage_cost: f64,
    pub convenience_yield: f64,
    pub expiry: f64,
}

impl Future {
    /// Theoretical futures price under continuous carrying costs.
    ///
    /// `F = S * exp((r - q + u - y) * T)`
    pub fn theoretical_price(&self) -> f64 {
        if self.underlying_spot <= 0.0 || self.expiry < 0.0 {
            return f64::NAN;
        }

        let carry =
            self.risk_free_rate - self.dividend_yield + self.storage_cost - self.convenience_yield;
        self.underlying_spot * (carry * self.expiry).exp()
    }

    /// Spot-futures basis, defined as `spot - futures`.
    pub fn basis(&self) -> f64 {
        self.underlying_spot - self.theoretical_price()
    }

    /// Implied repo rate from a market futures price.
    pub fn implied_repo_rate(&self, market_price: f64) -> f64 {
        if self.underlying_spot <= 0.0 || market_price <= 0.0 || self.expiry <= 0.0 {
            return f64::NAN;
        }

        (market_price / self.underlying_spot).ln() / self.expiry + self.dividend_yield
            - self.storage_cost
            + self.convenience_yield
    }
}

/// Eurodollar/SOFR futures quote conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterestRateFutureQuote;

impl InterestRateFutureQuote {
    /// Converts annualized decimal rate to quoted futures price.
    pub fn price_from_rate(rate: f64) -> f64 {
        100.0 - 100.0 * rate
    }

    /// Converts quoted futures price to annualized decimal rate.
    pub fn rate_from_price(price: f64) -> f64 {
        (100.0 - price) / 100.0
    }

    /// Convexity adjustment approximation: `0.5 * sigma^2 * T1 * T2`.
    pub fn convexity_adjustment(vol: f64, t1: f64, t2: f64) -> f64 {
        if vol < 0.0 || t1 < 0.0 || t2 < 0.0 {
            return f64::NAN;
        }
        0.5 * vol * vol * t1 * t2
    }

    /// Converts a futures-implied rate into an adjusted forward rate.
    pub fn forward_rate_from_futures_rate(futures_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
        futures_rate - Self::convexity_adjustment(vol, t1, t2)
    }

    /// Converts a forward rate into a convexity-adjusted futures-implied rate.
    pub fn futures_rate_from_forward_rate(forward_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
        forward_rate + Self::convexity_adjustment(vol, t1, t2)
    }
}
