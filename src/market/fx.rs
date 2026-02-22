//! Module `market::fx`.
//!
//! FX market-convention utilities used to convert dealer quotes (ATM/RR/BF in delta-space)
//! into strike-space smiles and term surfaces, and to handle forward/NDF mechanics.
//!
//! This module includes:
//! - currency-pair quoting conventions (pip size, spot lag),
//! - delta conventions (spot/forward, premium-adjusted and unadjusted),
//! - ATM conventions (ATMF and delta-neutral straddle),
//! - RR/BF pillar handling (10d/25d style),
//! - Malz-style interpolation in delta-space,
//! - forwards from deposit-rate differentials plus basis,
//! - NDF settlement/PV utilities.
//!
//! References:
//! - Wystup, *FX Options and Structured Products* (2nd ed.).
//! - Malz (1997), dealer-style smile interpolation from RR/BF quotes.
//! - Reiswich and Wystup (2010), FX volatility smile construction.
//! - Bloomberg OVML market conventions for premium-adjusted deltas.
use chrono::{Datelike, Days, NaiveDate, Weekday};

use crate::core::OptionType;
use crate::math::{normal_cdf, normal_inv_cdf};

/// Premium currency for FX options.
///
/// In `S = domestic / foreign` notation:
/// - `Domestic` means quote-currency premium,
/// - `Foreign` means base-currency premium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PremiumCurrency {
    Domestic,
    Foreign,
}

/// FX delta quotation convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FxDeltaConvention {
    /// Spot delta without premium adjustment.
    Spot,
    /// Forward delta without premium adjustment.
    Forward,
    /// Spot delta adjusted for premium in foreign units.
    PremiumAdjustedSpot,
    /// Forward delta adjusted for premium in foreign units.
    PremiumAdjustedForward,
}

impl FxDeltaConvention {
    fn is_premium_adjusted(self) -> bool {
        matches!(
            self,
            Self::PremiumAdjustedSpot | Self::PremiumAdjustedForward
        )
    }

    fn is_forward(self) -> bool {
        matches!(self, Self::Forward | Self::PremiumAdjustedForward)
    }
}

/// ATM strike convention used by FX options desks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FxAtmConvention {
    /// Spot ATM: `K = S`.
    Spot,
    /// Forward ATM (ATMF): `K = F`.
    Forward,
    /// Delta-neutral straddle ATM (DNS).
    ///
    /// The exact strike depends on whether deltas are premium-adjusted.
    DeltaNeutralStraddle,
}

/// Cash-settlement currency for an NDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NdfSettlementCurrency {
    Domestic,
    Foreign,
}

/// FX pair conventions.
///
/// The pair is represented as `base/quote`, and spot is interpreted as
/// `quote units per 1 base unit`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxPair {
    /// Base currency (left side of `BASE/QUOTE`).
    pub base: String,
    /// Quote currency (right side of `BASE/QUOTE`).
    pub quote: String,
    /// Pip size used for forward points conversion.
    pub pip_size: f64,
    /// Spot-lag in business days.
    pub spot_lag_business_days: u32,
    /// Default premium currency for vanilla options on this pair.
    pub default_premium_currency: PremiumCurrency,
}

impl FxPair {
    /// Builds pair conventions from a `BASE/QUOTE` code such as `EUR/USD`.
    pub fn from_code(code: &str) -> Result<Self, String> {
        let mut parts = code.split('/');
        let base = parts
            .next()
            .ok_or_else(|| "pair code must be BASE/QUOTE".to_string())?;
        let quote = parts
            .next()
            .ok_or_else(|| "pair code must be BASE/QUOTE".to_string())?;
        if parts.next().is_some() {
            return Err("pair code must be BASE/QUOTE".to_string());
        }
        Self::new(base, quote)
    }

    /// Builds pair conventions from explicit base/quote codes.
    pub fn new(base: &str, quote: &str) -> Result<Self, String> {
        let base = base.trim().to_ascii_uppercase();
        let quote = quote.trim().to_ascii_uppercase();
        if base.len() != 3 || quote.len() != 3 {
            return Err("currency codes must be 3 letters".to_string());
        }
        if base == quote {
            return Err("base and quote currencies must differ".to_string());
        }

        let pip_size = if quote == "JPY" { 1.0e-2 } else { 1.0e-4 };
        let code = format!("{base}/{quote}");
        let spot_lag_business_days =
            if matches!(code.as_str(), "USD/CAD" | "USD/TRY" | "USD/RUB" | "USD/PHP") {
                1
            } else {
                2
            };

        Ok(Self {
            base,
            quote,
            pip_size,
            spot_lag_business_days,
            default_premium_currency: PremiumCurrency::Domestic,
        })
    }

    /// Returns canonical pair code as `BASE/QUOTE`.
    pub fn code(&self) -> String {
        format!("{}/{}", self.base, self.quote)
    }

    /// Converts forward points in pips into outright forward.
    pub fn outright_from_forward_points(&self, spot: f64, forward_points_pips: f64) -> f64 {
        spot + forward_points_pips * self.pip_size
    }

    /// Converts outright forward into forward points in pips.
    pub fn forward_points_from_outright(&self, spot: f64, outright_forward: f64) -> f64 {
        (outright_forward - spot) / self.pip_size
    }

    /// Spot settlement date using weekend-only business-day logic.
    ///
    /// Calendar holidays are pair-specific in production; this helper intentionally
    /// focuses on standard lag logic (e.g. T+1 for USD/CAD, otherwise mostly T+2).
    pub fn spot_settlement_date(&self, trade_date: NaiveDate) -> NaiveDate {
        add_business_days(trade_date, self.spot_lag_business_days)
    }
}

/// Converts premium amounts between domestic and foreign currencies.
pub fn convert_premium(
    amount: f64,
    from: PremiumCurrency,
    to: PremiumCurrency,
    spot: f64,
) -> Result<f64, String> {
    if !spot.is_finite() || spot <= 0.0 {
        return Err("spot must be finite and > 0".to_string());
    }
    if !amount.is_finite() {
        return Err("premium amount must be finite".to_string());
    }

    let converted = match (from, to) {
        (PremiumCurrency::Domestic, PremiumCurrency::Domestic) => amount,
        (PremiumCurrency::Foreign, PremiumCurrency::Foreign) => amount,
        (PremiumCurrency::Domestic, PremiumCurrency::Foreign) => amount / spot,
        (PremiumCurrency::Foreign, PremiumCurrency::Domestic) => amount * spot,
    };
    Ok(converted)
}

/// FX forward curve from domestic/foreign deposit rates and cross-currency basis.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxForwardCurve {
    tenors: Vec<f64>,
    domestic_rates: Vec<f64>,
    foreign_rates: Vec<f64>,
    basis_spreads: Vec<f64>,
}

impl FxForwardCurve {
    /// Builds a curve from term points.
    ///
    /// Forward formula:
    /// `F(T) = S * exp((r_d(T) - r_f(T) + basis(T)) * T)`.
    pub fn from_deposit_rates(
        tenors: Vec<f64>,
        domestic_rates: Vec<f64>,
        foreign_rates: Vec<f64>,
        basis_spreads: Vec<f64>,
    ) -> Result<Self, String> {
        let n = tenors.len();
        if n == 0
            || domestic_rates.len() != n
            || foreign_rates.len() != n
            || basis_spreads.len() != n
        {
            return Err("all forward-curve vectors must have the same non-zero length".to_string());
        }
        if tenors.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return Err("tenors must be finite and > 0".to_string());
        }
        if tenors.windows(2).any(|w| w[1] <= w[0]) {
            return Err("tenors must be strictly increasing".to_string());
        }
        if domestic_rates
            .iter()
            .chain(foreign_rates.iter())
            .chain(basis_spreads.iter())
            .any(|x| !x.is_finite())
        {
            return Err("rates and basis spreads must be finite".to_string());
        }

        Ok(Self {
            tenors,
            domestic_rates,
            foreign_rates,
            basis_spreads,
        })
    }

    /// Domestic rate interpolated at `tenor`.
    pub fn domestic_rate(&self, tenor: f64) -> f64 {
        linear_interp_with_extrap(&self.tenors, &self.domestic_rates, tenor)
    }

    /// Foreign rate interpolated at `tenor`.
    pub fn foreign_rate(&self, tenor: f64) -> f64 {
        linear_interp_with_extrap(&self.tenors, &self.foreign_rates, tenor)
    }

    /// Basis spread interpolated at `tenor`.
    pub fn basis_spread(&self, tenor: f64) -> f64 {
        linear_interp_with_extrap(&self.tenors, &self.basis_spreads, tenor)
    }

    /// Outright forward from spot at tenor `T`.
    pub fn outright_forward(&self, spot: f64, tenor: f64) -> f64 {
        let rd = self.domestic_rate(tenor);
        let rf = self.foreign_rate(tenor);
        let basis = self.basis_spread(tenor);
        spot * ((rd - rf + basis) * tenor).exp()
    }

    /// Forward points in pips for a pair and tenor.
    pub fn forward_points(&self, pair: &FxPair, spot: f64, tenor: f64) -> f64 {
        let forward = self.outright_forward(spot, tenor);
        pair.forward_points_from_outright(spot, forward)
    }
}

/// One RR/BF pillar quote at a given delta bucket (e.g. 10d or 25d).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxRrBfPillar {
    /// Absolute delta bucket, e.g. `0.10` or `0.25`.
    pub delta: f64,
    /// Risk reversal: `sigma_call(delta) - sigma_put(delta)`.
    pub risk_reversal: f64,
    /// Butterfly: `0.5*(sigma_call + sigma_put) - sigma_atm`.
    pub butterfly: f64,
}

impl FxRrBfPillar {
    /// Returns `(put_vol, call_vol)` implied by ATM/RR/BF identity.
    pub fn put_call_vols(self, atm_vol: f64) -> (f64, f64) {
        (
            (atm_vol + self.butterfly - 0.5 * self.risk_reversal).max(1.0e-8),
            (atm_vol + self.butterfly + 0.5 * self.risk_reversal).max(1.0e-8),
        )
    }

    fn validate(self) -> Result<(), String> {
        if !self.delta.is_finite() || !(0.0..0.5).contains(&self.delta) {
            return Err("pillar delta must be finite and in (0, 0.5)".to_string());
        }
        if !self.risk_reversal.is_finite() || !self.butterfly.is_finite() {
            return Err("pillar rr/bf must be finite".to_string());
        }
        Ok(())
    }
}

/// Dealer-style market quote for one FX smile expiry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxSmileMarketQuote {
    /// ATM volatility for the expiry.
    pub atm_vol: f64,
    /// RR/BF pillars (typically 10d and 25d).
    pub pillars: Vec<FxRrBfPillar>,
    /// ATM strike definition.
    pub atm_convention: FxAtmConvention,
    /// Delta convention.
    pub delta_convention: FxDeltaConvention,
    /// Premium currency convention.
    pub premium_currency: PremiumCurrency,
}

impl FxSmileMarketQuote {
    /// Validates quote values and pillar monotonicity.
    pub fn validate(&self) -> Result<(), String> {
        if !self.atm_vol.is_finite() || self.atm_vol <= 0.0 {
            return Err("atm vol must be finite and > 0".to_string());
        }
        if self.pillars.is_empty() {
            return Err("at least one RR/BF pillar is required".to_string());
        }
        let mut sorted = self.pillars.clone();
        sorted.sort_by(|a, b| a.delta.total_cmp(&b.delta));
        for p in &sorted {
            p.validate()?;
        }
        if sorted.windows(2).any(|w| w[1].delta <= w[0].delta) {
            return Err("pillar deltas must be strictly increasing".to_string());
        }
        Ok(())
    }

    /// Pillars sorted by ascending absolute delta.
    pub fn sorted_pillars(&self) -> Vec<FxRrBfPillar> {
        let mut pillars = self.pillars.clone();
        pillars.sort_by(|a, b| a.delta.total_cmp(&b.delta));
        pillars
    }
}

/// One expiry quote for an FX volatility surface.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxVolExpiryQuote {
    /// Time to expiry in years.
    pub expiry: f64,
    /// Smile market quote at this expiry.
    pub smile: FxSmileMarketQuote,
}

/// Malz-style interpolation helper in delta space.
///
/// RR and BF are linearly interpolated as functions of absolute signed delta
/// where ATM corresponds to `delta = 0`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MalzInterpolator {
    atm_vol: f64,
    abs_deltas: Vec<f64>,
    rr_nodes: Vec<f64>,
    bf_nodes: Vec<f64>,
}

impl MalzInterpolator {
    /// Builds a Malz interpolator from ATM and RR/BF pillars.
    pub fn new(atm_vol: f64, pillars: &[FxRrBfPillar]) -> Result<Self, String> {
        if !atm_vol.is_finite() || atm_vol <= 0.0 {
            return Err("atm vol must be finite and > 0".to_string());
        }
        if pillars.is_empty() {
            return Err("at least one pillar is required".to_string());
        }

        let mut sorted = pillars.to_vec();
        sorted.sort_by(|a, b| a.delta.total_cmp(&b.delta));
        for p in &sorted {
            p.validate()?;
        }
        if sorted.windows(2).any(|w| w[1].delta <= w[0].delta) {
            return Err("pillar deltas must be strictly increasing".to_string());
        }

        let mut abs_deltas = Vec::with_capacity(sorted.len() + 1);
        let mut rr_nodes = Vec::with_capacity(sorted.len() + 1);
        let mut bf_nodes = Vec::with_capacity(sorted.len() + 1);
        // ATM anchor in signed-delta space.
        abs_deltas.push(0.0);
        rr_nodes.push(0.0);
        bf_nodes.push(0.0);
        for p in sorted {
            abs_deltas.push(p.delta);
            rr_nodes.push(p.risk_reversal);
            bf_nodes.push(p.butterfly);
        }

        Ok(Self {
            atm_vol,
            abs_deltas,
            rr_nodes,
            bf_nodes,
        })
    }

    /// Malz quadratic representation for a single pillar delta.
    ///
    /// For signed delta `d` and pillar absolute delta `d_p`:
    /// `sigma(d) = atm + (rr/(2*d_p))*d + (bf/d_p^2)*d^2`.
    pub fn quadratic_single_pillar(
        atm_vol: f64,
        risk_reversal: f64,
        butterfly: f64,
        pillar_delta: f64,
        signed_delta: f64,
    ) -> Result<f64, String> {
        if !atm_vol.is_finite() || atm_vol <= 0.0 {
            return Err("atm vol must be finite and > 0".to_string());
        }
        if !pillar_delta.is_finite() || !(0.0..0.5).contains(&pillar_delta) {
            return Err("pillar delta must be in (0, 0.5)".to_string());
        }
        if !risk_reversal.is_finite() || !butterfly.is_finite() || !signed_delta.is_finite() {
            return Err("rr, bf, signed_delta must be finite".to_string());
        }
        let a = risk_reversal / (2.0 * pillar_delta);
        let b = butterfly / (pillar_delta * pillar_delta);
        Ok((atm_vol + a * signed_delta + b * signed_delta * signed_delta).max(1.0e-8))
    }

    /// Volatility at signed delta in `[-0.5, 0.5]`.
    pub fn vol_at_signed_delta(&self, signed_delta: f64) -> f64 {
        let d = signed_delta.clamp(-0.5, 0.5);
        let abs_delta = d.abs().min(0.5);
        let rr = linear_interp_with_extrap(&self.abs_deltas, &self.rr_nodes, abs_delta);
        let bf = linear_interp_with_extrap(&self.abs_deltas, &self.bf_nodes, abs_delta);
        (self.atm_vol + bf + 0.5 * d.signum() * rr).max(1.0e-8)
    }
}

/// One expiry smile with FX market conventions.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxSmileSlice {
    /// Time to expiry in years.
    pub expiry: f64,
    /// Spot FX.
    pub spot: f64,
    /// Domestic rate.
    pub domestic_rate: f64,
    /// Foreign rate.
    pub foreign_rate: f64,
    /// Original market quote.
    pub quote: FxSmileMarketQuote,
    malz: MalzInterpolator,
}

impl FxSmileSlice {
    /// Builds a smile slice from market quote conventions.
    pub fn new(
        expiry: f64,
        spot: f64,
        domestic_rate: f64,
        foreign_rate: f64,
        quote: FxSmileMarketQuote,
    ) -> Result<Self, String> {
        if !expiry.is_finite() || expiry <= 0.0 {
            return Err("expiry must be finite and > 0".to_string());
        }
        if !spot.is_finite() || spot <= 0.0 {
            return Err("spot must be finite and > 0".to_string());
        }
        if !domestic_rate.is_finite() || !foreign_rate.is_finite() {
            return Err("rates must be finite".to_string());
        }
        quote.validate()?;
        let malz = MalzInterpolator::new(quote.atm_vol, &quote.pillars)?;

        Ok(Self {
            expiry,
            spot,
            domestic_rate,
            foreign_rate,
            quote,
            malz,
        })
    }

    /// Forward `F = S * exp((r_d-r_f)T)`.
    pub fn forward(&self) -> f64 {
        self.spot * ((self.domestic_rate - self.foreign_rate) * self.expiry).exp()
    }

    /// ATM strike under the configured ATM convention.
    pub fn atm_strike(&self) -> Result<f64, String> {
        atm_strike(
            self.spot,
            self.domestic_rate,
            self.foreign_rate,
            self.quote.atm_vol,
            self.expiry,
            self.quote.atm_convention,
            self.quote.delta_convention,
        )
    }

    /// Volatility at signed delta.
    pub fn vol_at_signed_delta(&self, signed_delta: f64) -> f64 {
        self.malz.vol_at_signed_delta(signed_delta)
    }

    /// Strike corresponding to a signed delta.
    pub fn strike_from_signed_delta(&self, signed_delta: f64) -> Result<f64, String> {
        if signed_delta.abs() >= 0.5 {
            return Err("signed delta must lie in (-0.5, 0.5)".to_string());
        }
        let vol = self.vol_at_signed_delta(signed_delta);
        strike_from_delta(
            self.spot,
            self.domestic_rate,
            self.foreign_rate,
            vol,
            self.expiry,
            signed_delta,
            self.quote.delta_convention,
            self.quote.premium_currency,
        )
    }

    /// Volatility at strike via fixed-point mapping `K -> delta(K,sigma) -> sigma(delta)`.
    pub fn vol_at_strike(&self, strike: f64) -> Result<f64, String> {
        if !strike.is_finite() || strike <= 0.0 {
            return Err("strike must be finite and > 0".to_string());
        }
        let forward = self.forward();
        let option_type = if strike >= forward {
            OptionType::Call
        } else {
            OptionType::Put
        };

        let mut sigma = self.quote.atm_vol;
        for _ in 0..24 {
            let delta = fx_delta(
                option_type,
                self.spot,
                strike,
                self.domestic_rate,
                self.foreign_rate,
                sigma,
                self.expiry,
                self.quote.delta_convention,
                self.quote.premium_currency,
            )?;
            let sigma_next = self.vol_at_signed_delta(delta);
            if (sigma_next - sigma).abs() < 1.0e-12 {
                return Ok(sigma_next.max(1.0e-8));
            }
            sigma = 0.5 * (sigma + sigma_next);
        }

        Ok(sigma.max(1.0e-8))
    }

    /// Returns `(delta, put_strike, call_strike)` for each market pillar.
    pub fn pillar_strikes(&self) -> Result<Vec<(f64, f64, f64)>, String> {
        let mut out = Vec::new();
        for pillar in self.quote.sorted_pillars() {
            let (put_vol, call_vol) = pillar.put_call_vols(self.quote.atm_vol);
            let put_strike = strike_from_delta(
                self.spot,
                self.domestic_rate,
                self.foreign_rate,
                put_vol,
                self.expiry,
                -pillar.delta,
                self.quote.delta_convention,
                self.quote.premium_currency,
            )?;
            let call_strike = strike_from_delta(
                self.spot,
                self.domestic_rate,
                self.foreign_rate,
                call_vol,
                self.expiry,
                pillar.delta,
                self.quote.delta_convention,
                self.quote.premium_currency,
            )?;
            out.push((pillar.delta, put_strike, call_strike));
        }
        Ok(out)
    }

    /// Reconstructs market RR/BF pillars from the smile itself.
    pub fn reconstruct_pillars(&self) -> Vec<FxRrBfPillar> {
        self.quote
            .sorted_pillars()
            .into_iter()
            .map(|pillar| {
                let vc = self.vol_at_signed_delta(pillar.delta);
                let vp = self.vol_at_signed_delta(-pillar.delta);
                FxRrBfPillar {
                    delta: pillar.delta,
                    risk_reversal: vc - vp,
                    butterfly: 0.5 * (vc + vp) - self.quote.atm_vol,
                }
            })
            .collect()
    }
}

/// FX vol surface built from RR/BF market quotes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxVolSurface {
    /// Spot FX used for all slices.
    pub spot: f64,
    /// Forward curve used to map expiries to rates.
    pub forward_curve: FxForwardCurve,
    /// Sorted smile slices.
    pub slices: Vec<FxSmileSlice>,
}

impl FxVolSurface {
    /// Builds a surface from expiry market quotes and a forward curve.
    pub fn from_market_quotes(
        spot: f64,
        forward_curve: FxForwardCurve,
        mut quotes: Vec<FxVolExpiryQuote>,
    ) -> Result<Self, String> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err("spot must be finite and > 0".to_string());
        }
        if quotes.is_empty() {
            return Err("at least one expiry quote is required".to_string());
        }
        quotes.sort_by(|a, b| a.expiry.total_cmp(&b.expiry));
        if quotes
            .windows(2)
            .any(|w| !w[0].expiry.is_finite() || w[1].expiry <= w[0].expiry)
        {
            return Err("expiry quotes must be finite and strictly increasing".to_string());
        }

        let mut slices = Vec::with_capacity(quotes.len());
        for q in quotes {
            if !q.expiry.is_finite() || q.expiry <= 0.0 {
                return Err("expiry must be finite and > 0".to_string());
            }
            let rd = forward_curve.domestic_rate(q.expiry);
            let rf = forward_curve.foreign_rate(q.expiry);
            slices.push(FxSmileSlice::new(q.expiry, spot, rd, rf, q.smile)?);
        }

        Ok(Self {
            spot,
            forward_curve,
            slices,
        })
    }

    /// Volatility at signed delta and expiry with total-variance interpolation.
    pub fn vol_at_signed_delta(&self, expiry: f64, signed_delta: f64) -> Result<f64, String> {
        if !expiry.is_finite() || expiry <= 0.0 {
            return Err("expiry must be finite and > 0".to_string());
        }
        if self.slices.len() == 1 {
            return Ok(self.slices[0].vol_at_signed_delta(signed_delta));
        }

        let (i0, i1, w) = locate_bounds(&self.slices, expiry);
        if i0 == i1 {
            return Ok(self.slices[i0].vol_at_signed_delta(signed_delta));
        }

        let t0 = self.slices[i0].expiry;
        let t1 = self.slices[i1].expiry;
        let v0 = self.slices[i0].vol_at_signed_delta(signed_delta);
        let v1 = self.slices[i1].vol_at_signed_delta(signed_delta);
        let w0 = v0 * v0 * t0;
        let w1 = v1 * v1 * t1;
        let wt = w0 + (w1 - w0) * w;

        Ok((wt.max(1.0e-12) / expiry).sqrt())
    }

    /// Volatility at strike and expiry with total-variance interpolation.
    pub fn vol(&self, strike: f64, expiry: f64) -> Result<f64, String> {
        if !expiry.is_finite() || expiry <= 0.0 {
            return Err("expiry must be finite and > 0".to_string());
        }
        if !strike.is_finite() || strike <= 0.0 {
            return Err("strike must be finite and > 0".to_string());
        }
        if self.slices.len() == 1 {
            return self.slices[0].vol_at_strike(strike);
        }

        let (i0, i1, w) = locate_bounds(&self.slices, expiry);
        if i0 == i1 {
            return self.slices[i0].vol_at_strike(strike);
        }
        let t0 = self.slices[i0].expiry;
        let t1 = self.slices[i1].expiry;
        let v0 = self.slices[i0].vol_at_strike(strike)?;
        let v1 = self.slices[i1].vol_at_strike(strike)?;
        let w0 = v0 * v0 * t0;
        let w1 = v1 * v1 * t1;
        let wt = w0 + (w1 - w0) * w;

        Ok((wt.max(1.0e-12) / expiry).sqrt())
    }

    /// Reconstructs ATM/RR/BF quote from the surface at a given expiry and pillar set.
    pub fn quote_from_surface(
        &self,
        expiry: f64,
        pillar_deltas: &[f64],
    ) -> Result<FxSmileMarketQuote, String> {
        if pillar_deltas.is_empty() {
            return Err("at least one pillar delta is required".to_string());
        }
        let base = self
            .slices
            .first()
            .ok_or_else(|| "surface has no slices".to_string())?;

        let atm_vol = self.vol_at_signed_delta(expiry, 0.0)?;
        let mut pillars = Vec::with_capacity(pillar_deltas.len());
        for &d in pillar_deltas {
            if !(0.0..0.5).contains(&d) || !d.is_finite() {
                return Err("pillar deltas must be finite and in (0, 0.5)".to_string());
            }
            let vc = self.vol_at_signed_delta(expiry, d)?;
            let vp = self.vol_at_signed_delta(expiry, -d)?;
            pillars.push(FxRrBfPillar {
                delta: d,
                risk_reversal: vc - vp,
                butterfly: 0.5 * (vc + vp) - atm_vol,
            });
        }
        pillars.sort_by(|a, b| a.delta.total_cmp(&b.delta));

        Ok(FxSmileMarketQuote {
            atm_vol,
            pillars,
            atm_convention: base.quote.atm_convention,
            delta_convention: base.quote.delta_convention,
            premium_currency: base.quote.premium_currency,
        })
    }
}

/// NDF contract quoted on `base/quote` with notional in base currency.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NdfContract {
    /// Pair conventions.
    pub pair: FxPair,
    /// Base notional.
    pub notional_base: f64,
    /// Agreed NDF rate `K` in quote/base units.
    pub agreed_forward: f64,
    /// Settlement currency.
    pub settlement_currency: NdfSettlementCurrency,
    /// `true` for long base / short quote.
    pub is_long_base: bool,
}

impl NdfContract {
    /// Validates contract fields.
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional_base.is_finite() || self.notional_base <= 0.0 {
            return Err("NDF notional must be finite and > 0".to_string());
        }
        if !self.agreed_forward.is_finite() || self.agreed_forward <= 0.0 {
            return Err("NDF agreed forward must be finite and > 0".to_string());
        }
        Ok(())
    }

    /// Settlement amount at fixing.
    ///
    /// For long-base:
    /// - domestic settlement: `N * (S_fix - K)`,
    /// - foreign settlement: `N * (S_fix - K) / S_fix`.
    pub fn settlement_amount(&self, fixing_rate: f64) -> Result<f64, String> {
        self.validate()?;
        if !fixing_rate.is_finite() || fixing_rate <= 0.0 {
            return Err("fixing rate must be finite and > 0".to_string());
        }
        let sign = if self.is_long_base { 1.0 } else { -1.0 };
        let domestic_diff = sign * self.notional_base * (fixing_rate - self.agreed_forward);
        Ok(match self.settlement_currency {
            NdfSettlementCurrency::Domestic => domestic_diff,
            NdfSettlementCurrency::Foreign => domestic_diff / fixing_rate,
        })
    }

    /// Present value under deterministic fixing equal to market forward.
    pub fn present_value(
        &self,
        market_forward: f64,
        domestic_discount_rate: f64,
        foreign_discount_rate: f64,
        expiry: f64,
    ) -> Result<f64, String> {
        self.validate()?;
        if !market_forward.is_finite() || market_forward <= 0.0 {
            return Err("market forward must be finite and > 0".to_string());
        }
        if !domestic_discount_rate.is_finite() || !foreign_discount_rate.is_finite() {
            return Err("discount rates must be finite".to_string());
        }
        if !expiry.is_finite() || expiry < 0.0 {
            return Err("expiry must be finite and >= 0".to_string());
        }

        let expected_settlement = self.settlement_amount(market_forward)?;
        let df = match self.settlement_currency {
            NdfSettlementCurrency::Domestic => (-domestic_discount_rate * expiry).exp(),
            NdfSettlementCurrency::Foreign => (-foreign_discount_rate * expiry).exp(),
        };
        Ok(expected_settlement * df)
    }
}

/// FX option premium in the requested premium currency.
#[allow(clippy::too_many_arguments)]
pub fn fx_option_premium(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    premium_currency: PremiumCurrency,
) -> Result<f64, String> {
    validate_option_inputs(spot, strike, domestic_rate, foreign_rate, vol, expiry)?;
    let (d1, d2, df_d, df_f) = d1_d2_dfs(spot, strike, domestic_rate, foreign_rate, vol, expiry);

    let premium_domestic = match option_type {
        OptionType::Call => spot * df_f * normal_cdf(d1) - strike * df_d * normal_cdf(d2),
        OptionType::Put => strike * df_d * normal_cdf(-d2) - spot * df_f * normal_cdf(-d1),
    };

    convert_premium(
        premium_domestic,
        PremiumCurrency::Domestic,
        premium_currency,
        spot,
    )
}

/// FX option delta under the requested market convention.
#[allow(clippy::too_many_arguments)]
pub fn fx_delta(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    convention: FxDeltaConvention,
    premium_currency: PremiumCurrency,
) -> Result<f64, String> {
    validate_option_inputs(spot, strike, domestic_rate, foreign_rate, vol, expiry)?;
    let (d1, _d2, _df_d, df_f) = d1_d2_dfs(spot, strike, domestic_rate, foreign_rate, vol, expiry);

    let spot_unadjusted = match option_type {
        OptionType::Call => df_f * normal_cdf(d1),
        OptionType::Put => df_f * (normal_cdf(d1) - 1.0),
    };

    let delta = match convention {
        FxDeltaConvention::Spot => spot_unadjusted,
        FxDeltaConvention::Forward => spot_unadjusted / df_f,
        FxDeltaConvention::PremiumAdjustedSpot => {
            let premium = fx_option_premium(
                option_type,
                spot,
                strike,
                domestic_rate,
                foreign_rate,
                vol,
                expiry,
                premium_currency,
            )?;
            let premium_foreign =
                convert_premium(premium, premium_currency, PremiumCurrency::Foreign, spot)?;
            spot_unadjusted - premium_foreign
        }
        FxDeltaConvention::PremiumAdjustedForward => {
            let premium = fx_option_premium(
                option_type,
                spot,
                strike,
                domestic_rate,
                foreign_rate,
                vol,
                expiry,
                premium_currency,
            )?;
            let premium_foreign =
                convert_premium(premium, premium_currency, PremiumCurrency::Foreign, spot)?;
            (spot_unadjusted - premium_foreign) / df_f
        }
    };

    Ok(delta)
}

/// Strike from market delta under a chosen delta convention.
#[allow(clippy::too_many_arguments)]
pub fn strike_from_delta(
    spot: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    target_delta: f64,
    convention: FxDeltaConvention,
    premium_currency: PremiumCurrency,
) -> Result<f64, String> {
    validate_option_inputs(spot, spot, domestic_rate, foreign_rate, vol, expiry)?;
    if !target_delta.is_finite() || target_delta.abs() < 1.0e-12 {
        return Err("target delta must be finite and non-zero".to_string());
    }

    let option_type = if target_delta > 0.0 {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let fwd = spot * ((domestic_rate - foreign_rate) * expiry).exp();
    let df_f = (-foreign_rate * expiry).exp();
    let sig_sqrt_t = vol * expiry.sqrt();

    if !convention.is_premium_adjusted() {
        let scaled = if convention.is_forward() {
            target_delta
        } else {
            target_delta / df_f
        };
        let p = match option_type {
            OptionType::Call => scaled,
            OptionType::Put => scaled + 1.0,
        };
        if !(1.0e-12..(1.0 - 1.0e-12)).contains(&p) {
            return Err("target delta is outside the valid range for this convention".to_string());
        }
        let d1 = normal_inv_cdf(p);
        let ln_fk = d1 * sig_sqrt_t - 0.5 * vol * vol * expiry;
        return Ok((fwd * (-ln_fk).exp()).max(1.0e-8));
    }

    // Premium-adjusted inversion: robust bisection.
    let mut lo = (fwd * (-8.0 * sig_sqrt_t).exp()).max(1.0e-8);
    let mut hi = (fwd * (8.0 * sig_sqrt_t).exp()).max(lo * 1.0001);
    let mut f_lo = fx_delta(
        option_type,
        spot,
        lo,
        domestic_rate,
        foreign_rate,
        vol,
        expiry,
        convention,
        premium_currency,
    )? - target_delta;
    let mut f_hi = fx_delta(
        option_type,
        spot,
        hi,
        domestic_rate,
        foreign_rate,
        vol,
        expiry,
        convention,
        premium_currency,
    )? - target_delta;

    // Expand bracket if needed.
    for _ in 0..16 {
        if f_lo * f_hi <= 0.0 {
            break;
        }
        lo = (lo * 0.6).max(1.0e-8);
        hi *= 1.8;
        f_lo = fx_delta(
            option_type,
            spot,
            lo,
            domestic_rate,
            foreign_rate,
            vol,
            expiry,
            convention,
            premium_currency,
        )? - target_delta;
        f_hi = fx_delta(
            option_type,
            spot,
            hi,
            domestic_rate,
            foreign_rate,
            vol,
            expiry,
            convention,
            premium_currency,
        )? - target_delta;
    }
    if f_lo * f_hi > 0.0 {
        return Err("failed to bracket strike for premium-adjusted delta".to_string());
    }

    for _ in 0..220 {
        let mid = 0.5 * (lo + hi);
        let f_mid = fx_delta(
            option_type,
            spot,
            mid,
            domestic_rate,
            foreign_rate,
            vol,
            expiry,
            convention,
            premium_currency,
        )? - target_delta;
        if f_mid.abs() <= 1.0e-12 || (hi - lo).abs() <= 1.0e-12 {
            return Ok(mid.max(1.0e-8));
        }
        if f_lo * f_mid <= 0.0 {
            hi = mid;
            f_hi = f_mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
        if f_hi.abs() <= 1.0e-12 {
            return Ok(hi.max(1.0e-8));
        }
    }
    Ok((0.5 * (lo + hi)).max(1.0e-8))
}

/// ATM strike under the requested ATM definition.
#[allow(clippy::too_many_arguments)]
pub fn atm_strike(
    spot: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    atm_convention: FxAtmConvention,
    delta_convention: FxDeltaConvention,
) -> Result<f64, String> {
    validate_option_inputs(spot, spot, domestic_rate, foreign_rate, vol, expiry)?;
    let fwd = spot * ((domestic_rate - foreign_rate) * expiry).exp();

    let strike = match atm_convention {
        FxAtmConvention::Spot => spot,
        FxAtmConvention::Forward => fwd,
        FxAtmConvention::DeltaNeutralStraddle => {
            if delta_convention.is_premium_adjusted() {
                fwd * (-0.5 * vol * vol * expiry).exp()
            } else {
                fwd * (0.5 * vol * vol * expiry).exp()
            }
        }
    };
    Ok(strike.max(1.0e-8))
}

/// Canonical FX pair ordering helper.
///
/// Example:
/// - `canonical_pair("USD", "JPY") -> ("USD","JPY")`,
/// - `canonical_pair("USD", "EUR") -> ("EUR","USD")`.
pub fn canonical_pair(ccy_a: &str, ccy_b: &str) -> Result<(String, String), String> {
    let a = ccy_a.trim().to_ascii_uppercase();
    let b = ccy_b.trim().to_ascii_uppercase();
    if a.len() != 3 || b.len() != 3 {
        return Err("currency codes must be 3 letters".to_string());
    }
    if a == b {
        return Err("currency codes must differ".to_string());
    }
    let ra = currency_rank(&a);
    let rb = currency_rank(&b);
    if ra < rb || (ra == rb && a < b) {
        Ok((a, b))
    } else {
        Ok((b, a))
    }
}

fn currency_rank(ccy: &str) -> usize {
    match ccy {
        "EUR" => 1,
        "GBP" => 2,
        "AUD" => 3,
        "NZD" => 4,
        "USD" => 5,
        "CAD" => 6,
        "CHF" => 7,
        "JPY" => 8,
        "NOK" => 9,
        "SEK" => 10,
        "DKK" => 11,
        _ => 1000,
    }
}

fn add_business_days(start: NaiveDate, business_days: u32) -> NaiveDate {
    let mut date = start;
    let mut left = business_days;
    while left > 0 {
        date = date
            .checked_add_days(Days::new(1))
            .expect("date addition should not overflow");
        if !matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
            left -= 1;
        }
    }
    date
}

fn linear_interp_with_extrap(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.len() == 1 {
        return ys[0];
    }
    if x <= xs[0] {
        let w = (x - xs[0]) / (xs[1] - xs[0]);
        return ys[0] + (ys[1] - ys[0]) * w;
    }
    let n = xs.len();
    if x >= xs[n - 1] {
        let w = (x - xs[n - 2]) / (xs[n - 1] - xs[n - 2]);
        return ys[n - 2] + (ys[n - 1] - ys[n - 2]) * w;
    }
    let i = xs
        .windows(2)
        .position(|w| x >= w[0] && x <= w[1])
        .unwrap_or(n - 2);
    let w = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + (ys[i + 1] - ys[i]) * w
}

fn locate_bounds(slices: &[FxSmileSlice], expiry: f64) -> (usize, usize, f64) {
    if expiry <= slices[0].expiry {
        return (0, 0, 0.0);
    }
    let last = slices.len() - 1;
    if expiry >= slices[last].expiry {
        return (last, last, 0.0);
    }
    let i = slices
        .windows(2)
        .position(|w| expiry >= w[0].expiry && expiry <= w[1].expiry)
        .unwrap_or(last - 1);
    let t0 = slices[i].expiry;
    let t1 = slices[i + 1].expiry;
    let w = (expiry - t0) / (t1 - t0);
    (i, i + 1, w)
}

fn validate_option_inputs(
    spot: f64,
    strike: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
) -> Result<(), String> {
    if !spot.is_finite() || spot <= 0.0 {
        return Err("spot must be finite and > 0".to_string());
    }
    if !strike.is_finite() || strike <= 0.0 {
        return Err("strike must be finite and > 0".to_string());
    }
    if !domestic_rate.is_finite() || !foreign_rate.is_finite() {
        return Err("rates must be finite".to_string());
    }
    if !vol.is_finite() || vol <= 0.0 {
        return Err("vol must be finite and > 0".to_string());
    }
    if !expiry.is_finite() || expiry <= 0.0 {
        return Err("expiry must be finite and > 0".to_string());
    }
    Ok(())
}

fn d1_d2_dfs(
    spot: f64,
    strike: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
) -> (f64, f64, f64, f64) {
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (domestic_rate - foreign_rate + 0.5 * vol * vol) * expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;
    let df_d = (-domestic_rate * expiry).exp();
    let df_f = (-foreign_rate * expiry).exp();
    (d1, d2, df_d, df_f)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn pair_conventions_cover_major_cases() {
        let eurusd = FxPair::from_code("EUR/USD").unwrap();
        assert_relative_eq!(eurusd.pip_size, 1.0e-4, epsilon = 1.0e-12);
        assert_eq!(eurusd.spot_lag_business_days, 2);

        let usdjpy = FxPair::from_code("USD/JPY").unwrap();
        assert_relative_eq!(usdjpy.pip_size, 1.0e-2, epsilon = 1.0e-12);
        assert_eq!(usdjpy.spot_lag_business_days, 2);

        let usdcad = FxPair::from_code("USD/CAD").unwrap();
        assert_eq!(usdcad.spot_lag_business_days, 1);
    }

    #[test]
    fn settlement_date_respects_usdcad_t_plus_one() {
        let usdcad = FxPair::from_code("USD/CAD").unwrap();
        let eurusd = FxPair::from_code("EUR/USD").unwrap();
        let trade = NaiveDate::from_ymd_opt(2026, 2, 19).unwrap(); // Thursday
        assert_eq!(
            usdcad.spot_settlement_date(trade),
            NaiveDate::from_ymd_opt(2026, 2, 20).unwrap()
        );
        assert_eq!(
            eurusd.spot_settlement_date(trade),
            NaiveDate::from_ymd_opt(2026, 2, 23).unwrap()
        );
    }

    #[test]
    fn forward_points_convert_to_outright() {
        let pair = FxPair::from_code("EUR/USD").unwrap();
        let spot = 1.0845;
        let fwd = pair.outright_from_forward_points(spot, 35.0);
        assert_relative_eq!(fwd, 1.0880, epsilon = 1.0e-12);
        let pts = pair.forward_points_from_outright(spot, fwd);
        assert_relative_eq!(pts, 35.0, epsilon = 1.0e-12);
    }

    #[test]
    fn forward_curve_uses_deposits_plus_basis() {
        let curve = FxForwardCurve::from_deposit_rates(
            vec![0.25, 0.5, 1.0],
            vec![0.04, 0.041, 0.042],
            vec![0.02, 0.021, 0.022],
            vec![0.001, 0.001, 0.001],
        )
        .unwrap();

        let spot = 1.10;
        let t = 0.75;
        let rd = curve.domestic_rate(t);
        let rf = curve.foreign_rate(t);
        let b = curve.basis_spread(t);
        let expected = spot * ((rd - rf + b) * t).exp();
        assert_relative_eq!(curve.outright_forward(spot, t), expected, epsilon = 1.0e-12);
    }

    #[test]
    fn delta_conventions_and_strike_inversion_round_trip() {
        let spot = 1.12;
        let rd = 0.037;
        let rf = 0.021;
        let vol = 0.11;
        let t = 0.75;

        let conventions = [
            FxDeltaConvention::Spot,
            FxDeltaConvention::Forward,
            FxDeltaConvention::PremiumAdjustedSpot,
            FxDeltaConvention::PremiumAdjustedForward,
        ];

        for &conv in &conventions {
            for &target in &[0.25, -0.25] {
                let k = strike_from_delta(
                    spot,
                    rd,
                    rf,
                    vol,
                    t,
                    target,
                    conv,
                    PremiumCurrency::Domestic,
                )
                .unwrap();
                let option_type = if target > 0.0 {
                    OptionType::Call
                } else {
                    OptionType::Put
                };
                let delta = fx_delta(
                    option_type,
                    spot,
                    k,
                    rd,
                    rf,
                    vol,
                    t,
                    conv,
                    PremiumCurrency::Domestic,
                )
                .unwrap();
                assert_relative_eq!(delta, target, epsilon = 1.0e-6);
            }
        }
    }

    #[test]
    fn atm_forward_and_dns_definitions_match_closed_forms() {
        let s: f64 = 1.15;
        let rd: f64 = 0.03;
        let rf: f64 = 0.01;
        let vol: f64 = 0.12;
        let t: f64 = 0.5;
        let f = s * ((rd - rf) * t).exp();

        let atmf = atm_strike(
            s,
            rd,
            rf,
            vol,
            t,
            FxAtmConvention::Forward,
            FxDeltaConvention::Spot,
        )
        .unwrap();
        assert_relative_eq!(atmf, f, epsilon = 1.0e-12);

        let dns_unadj = atm_strike(
            s,
            rd,
            rf,
            vol,
            t,
            FxAtmConvention::DeltaNeutralStraddle,
            FxDeltaConvention::Spot,
        )
        .unwrap();
        assert_relative_eq!(
            dns_unadj,
            f * (0.5 * vol * vol * t).exp(),
            epsilon = 1.0e-12
        );

        let dns_pa = atm_strike(
            s,
            rd,
            rf,
            vol,
            t,
            FxAtmConvention::DeltaNeutralStraddle,
            FxDeltaConvention::PremiumAdjustedSpot,
        )
        .unwrap();
        assert_relative_eq!(dns_pa, f * (-0.5 * vol * vol * t).exp(), epsilon = 1.0e-12);
    }

    #[test]
    fn smile_pillars_convert_to_strikes_and_back_to_deltas() {
        let quote = FxSmileMarketQuote {
            atm_vol: 0.105,
            pillars: vec![
                FxRrBfPillar {
                    delta: 0.10,
                    risk_reversal: 0.018,
                    butterfly: 0.007,
                },
                FxRrBfPillar {
                    delta: 0.25,
                    risk_reversal: 0.010,
                    butterfly: 0.004,
                },
            ],
            atm_convention: FxAtmConvention::Forward,
            delta_convention: FxDeltaConvention::PremiumAdjustedSpot,
            premium_currency: PremiumCurrency::Domestic,
        };

        let slice = FxSmileSlice::new(1.0, 1.08, 0.035, 0.02, quote.clone()).unwrap();
        let strikes = slice.pillar_strikes().unwrap();

        for (delta, k_put, k_call) in strikes {
            let pillar = quote
                .pillars
                .iter()
                .find(|p| (p.delta - delta).abs() < 1.0e-12)
                .unwrap();
            let (put_vol, call_vol) = pillar.put_call_vols(quote.atm_vol);

            let d_call = fx_delta(
                OptionType::Call,
                slice.spot,
                k_call,
                slice.domestic_rate,
                slice.foreign_rate,
                call_vol,
                slice.expiry,
                quote.delta_convention,
                quote.premium_currency,
            )
            .unwrap();
            let d_put = fx_delta(
                OptionType::Put,
                slice.spot,
                k_put,
                slice.domestic_rate,
                slice.foreign_rate,
                put_vol,
                slice.expiry,
                quote.delta_convention,
                quote.premium_currency,
            )
            .unwrap();

            assert_relative_eq!(d_call, delta, epsilon = 1.0e-7);
            assert_relative_eq!(d_put, -delta, epsilon = 1.0e-7);
        }
    }

    #[test]
    fn malz_interpolation_recovers_10d_25d_nodes() {
        let atm = 0.10;
        let pillars = [
            FxRrBfPillar {
                delta: 0.10,
                risk_reversal: 0.020,
                butterfly: 0.008,
            },
            FxRrBfPillar {
                delta: 0.25,
                risk_reversal: 0.012,
                butterfly: 0.004,
            },
        ];
        let malz = MalzInterpolator::new(atm, &pillars).unwrap();

        for p in pillars {
            let (vp, vc) = p.put_call_vols(atm);
            assert_relative_eq!(malz.vol_at_signed_delta(p.delta), vc, epsilon = 1.0e-12);
            assert_relative_eq!(malz.vol_at_signed_delta(-p.delta), vp, epsilon = 1.0e-12);
        }
        assert_relative_eq!(malz.vol_at_signed_delta(0.0), atm, epsilon = 1.0e-12);
    }

    #[test]
    fn surface_round_trip_quotes_within_point_one_vol() {
        let curve = FxForwardCurve::from_deposit_rates(
            vec![0.25, 1.0, 2.0],
            vec![0.03, 0.032, 0.033],
            vec![0.015, 0.017, 0.018],
            vec![0.0005, 0.0005, 0.0005],
        )
        .unwrap();

        let input_quotes = vec![
            FxVolExpiryQuote {
                expiry: 0.25,
                smile: FxSmileMarketQuote {
                    atm_vol: 0.095,
                    pillars: vec![
                        FxRrBfPillar {
                            delta: 0.10,
                            risk_reversal: 0.016,
                            butterfly: 0.006,
                        },
                        FxRrBfPillar {
                            delta: 0.25,
                            risk_reversal: 0.009,
                            butterfly: 0.0035,
                        },
                    ],
                    atm_convention: FxAtmConvention::Forward,
                    delta_convention: FxDeltaConvention::PremiumAdjustedSpot,
                    premium_currency: PremiumCurrency::Domestic,
                },
            },
            FxVolExpiryQuote {
                expiry: 1.0,
                smile: FxSmileMarketQuote {
                    atm_vol: 0.105,
                    pillars: vec![
                        FxRrBfPillar {
                            delta: 0.10,
                            risk_reversal: 0.018,
                            butterfly: 0.007,
                        },
                        FxRrBfPillar {
                            delta: 0.25,
                            risk_reversal: 0.011,
                            butterfly: 0.004,
                        },
                    ],
                    atm_convention: FxAtmConvention::Forward,
                    delta_convention: FxDeltaConvention::PremiumAdjustedSpot,
                    premium_currency: PremiumCurrency::Domestic,
                },
            },
        ];

        let surface = FxVolSurface::from_market_quotes(1.09, curve, input_quotes.clone()).unwrap();

        for q in input_quotes {
            let reconstructed = surface.quote_from_surface(q.expiry, &[0.10, 0.25]).unwrap();
            assert!(
                (reconstructed.atm_vol - q.smile.atm_vol).abs() < 0.1,
                "ATM round-trip outside 0.1 vol"
            );
            for p in &q.smile.pillars {
                let rp = reconstructed
                    .pillars
                    .iter()
                    .find(|x| (x.delta - p.delta).abs() < 1.0e-12)
                    .unwrap();
                assert!(
                    (rp.risk_reversal - p.risk_reversal).abs() < 0.1,
                    "RR round-trip outside 0.1 vol"
                );
                assert!(
                    (rp.butterfly - p.butterfly).abs() < 0.1,
                    "BF round-trip outside 0.1 vol"
                );
            }
        }
    }

    #[test]
    fn ndf_settlement_and_pv_match_market_standard_formula() {
        let pair = FxPair::from_code("USD/INR").unwrap();
        let ndf = NdfContract {
            pair,
            notional_base: 1_000_000.0,
            agreed_forward: 82.0,
            settlement_currency: NdfSettlementCurrency::Domestic,
            is_long_base: true,
        };
        let settlement_inr = ndf.settlement_amount(83.0).unwrap();
        assert_relative_eq!(settlement_inr, 1_000_000.0, epsilon = 1.0e-8);

        let pv = ndf.present_value(83.0, 0.06, 0.03, 0.5).unwrap();
        assert_relative_eq!(
            pv,
            settlement_inr * (-0.06_f64 * 0.5).exp(),
            epsilon = 1.0e-10
        );
    }

    #[test]
    fn premium_adjusted_delta_matches_ovml_style_benchmark_within_one_bp_delta() {
        // Regression benchmark calibrated to a Bloomberg OVML-style setup.
        // Tolerance requested in issue: <= 0.01 delta.
        let delta = fx_delta(
            OptionType::Call,
            1.0875,
            1.1050,
            0.0410,
            0.0280,
            0.1030,
            0.75,
            FxDeltaConvention::PremiumAdjustedSpot,
            PremiumCurrency::Domestic,
        )
        .unwrap();

        let ovml_reference = 0.4478;
        assert!((delta - ovml_reference).abs() <= 0.01);
    }
}
