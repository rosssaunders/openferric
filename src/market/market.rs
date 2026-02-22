//! Module `market::market`.
//!
//! Implements market abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `VolSurfaceClone`, `VolSurface`, `VolSource`, `Market`, `MarketBuilder` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: choose this module when its API directly matches your instrument/model assumptions; otherwise use a more specialized engine module.

use std::any::Any;

use crate::core::PricingError;

/// Clone support for boxed volatility surface trait objects.
pub trait VolSurfaceClone {
    /// Clones the concrete surface behind the trait object.
    fn clone_box(&self) -> Box<dyn VolSurface>;
}

impl<T> VolSurfaceClone for T
where
    T: 'static + VolSurface + Clone,
{
    fn clone_box(&self) -> Box<dyn VolSurface> {
        Box::new(self.clone())
    }
}

/// Volatility surface abstraction used by pricing engines.
pub trait VolSurface: std::fmt::Debug + Send + Sync + VolSurfaceClone + Any {
    /// Returns implied volatility for a given strike and expiry.
    fn vol(&self, strike: f64, expiry: f64) -> f64;
}

impl Clone for Box<dyn VolSurface> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl VolSurface for crate::vol::surface::VolSurface {
    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        crate::vol::surface::VolSurface::vol(self, strike, expiry)
    }
}

/// Serializable sampled volatility surface using bilinear interpolation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampledVolSurface {
    pub strikes: Vec<f64>,
    pub expiries: Vec<f64>,
    pub vols: Vec<Vec<f64>>,
}

impl SampledVolSurface {
    /// Creates a sampled surface from explicit grids.
    pub fn new(strikes: Vec<f64>, expiries: Vec<f64>, vols: Vec<Vec<f64>>) -> Result<Self, String> {
        if strikes.len() < 2 || expiries.len() < 2 {
            return Err("sampled surface requires >= 2 strikes and >= 2 expiries".to_string());
        }
        if strikes.windows(2).any(|w| w[1] <= w[0]) {
            return Err("sampled surface strikes must be strictly increasing".to_string());
        }
        if expiries.windows(2).any(|w| w[1] <= w[0]) {
            return Err("sampled surface expiries must be strictly increasing".to_string());
        }
        if vols.len() != expiries.len() {
            return Err("sampled surface row count must match expiries".to_string());
        }
        if vols.iter().any(|row| row.len() != strikes.len()) {
            return Err("sampled surface each row must match strike count".to_string());
        }
        if vols
            .iter()
            .flat_map(|row| row.iter())
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("sampled surface vols must be finite and > 0".to_string());
        }

        Ok(Self {
            strikes,
            expiries,
            vols,
        })
    }

    fn default_strikes(spot: f64) -> Vec<f64> {
        const M: [f64; 17] = [
            0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.1,
        ];
        M.iter().map(|m| (spot * m).max(1.0e-8)).collect()
    }

    fn default_expiries() -> Vec<f64> {
        vec![1.0 / 52.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    }

    fn locate_bounds(grid: &[f64], x: f64) -> (usize, usize, f64) {
        if x <= grid[0] {
            return (0, 0, 0.0);
        }
        let last = grid.len() - 1;
        if x >= grid[last] {
            return (last, last, 0.0);
        }

        let mut lo = 0usize;
        for i in 0..last {
            if x >= grid[i] && x <= grid[i + 1] {
                lo = i;
                break;
            }
        }

        let hi = lo + 1;
        let w = (x - grid[lo]) / (grid[hi] - grid[lo]);
        (lo, hi, w)
    }

    /// Samples a trait-object surface onto a fixed strike/expiry grid.
    pub fn from_surface(surface: &dyn VolSurface, spot: f64) -> Self {
        let strikes = Self::default_strikes(spot.max(1.0e-8));
        let expiries = Self::default_expiries();
        let mut vols = Vec::with_capacity(expiries.len());

        for &expiry in &expiries {
            let mut row = Vec::with_capacity(strikes.len());
            for &strike in &strikes {
                let v = surface.vol(strike, expiry);
                row.push(if v.is_finite() { v.max(1.0e-8) } else { 1.0e-8 });
            }
            vols.push(row);
        }

        Self {
            strikes,
            expiries,
            vols,
        }
    }

    /// Bilinear volatility lookup.
    pub fn vol(&self, strike: f64, expiry: f64) -> f64 {
        let (ei0, ei1, ew) = Self::locate_bounds(&self.expiries, expiry.max(self.expiries[0]));
        let (si0, si1, sw) = Self::locate_bounds(&self.strikes, strike.max(self.strikes[0]));

        if ei0 == ei1 && si0 == si1 {
            return self.vols[ei0][si0];
        }
        if ei0 == ei1 {
            let v0 = self.vols[ei0][si0];
            let v1 = self.vols[ei0][si1];
            return v0 + (v1 - v0) * sw;
        }
        if si0 == si1 {
            let v0 = self.vols[ei0][si0];
            let v1 = self.vols[ei1][si0];
            return v0 + (v1 - v0) * ew;
        }

        let v00 = self.vols[ei0][si0];
        let v01 = self.vols[ei0][si1];
        let v10 = self.vols[ei1][si0];
        let v11 = self.vols[ei1][si1];

        let v0 = v00 + (v01 - v00) * sw;
        let v1 = v10 + (v11 - v10) * sw;
        v0 + (v1 - v0) * ew
    }
}

/// Volatility source for a market snapshot.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VolSource {
    /// Constant volatility.
    Flat(f64),
    /// Parametric SVI surface.
    Parametric(crate::vol::surface::VolSurface),
    /// Sampled volatility grid.
    Sampled(SampledVolSurface),
}

impl VolSource {
    /// Returns a volatility value for the requested strike and expiry.
    pub fn vol(&self, strike: f64, expiry: f64) -> f64 {
        match self {
            Self::Flat(v) => *v,
            Self::Parametric(surface) => surface.vol(strike, expiry),
            Self::Sampled(surface) => surface.vol(strike, expiry),
        }
    }
}

/// Market snapshot used by all pricing engines.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Market {
    /// Spot price.
    pub spot: f64,
    /// Continuously compounded risk-free rate.
    pub rate: f64,
    /// Continuously compounded dividend yield.
    pub dividend_yield: f64,
    /// Volatility source.
    pub vol: VolSource,
    /// Optional date string for bindings/interoperability.
    pub reference_date: Option<String>,
}

impl Market {
    /// Starts a market builder.
    #[inline]
    pub fn builder() -> MarketBuilder {
        MarketBuilder::default()
    }

    /// Returns spot price.
    #[inline]
    pub fn spot(&self) -> f64 {
        self.spot
    }

    /// Returns risk-free rate.
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Returns dividend yield.
    #[inline]
    pub fn dividend(&self) -> f64 {
        self.dividend_yield
    }

    /// Resolves volatility for strike and expiry.
    #[inline]
    pub fn vol(&self, strike: f64, expiry: f64) -> f64 {
        self.vol_for(strike, expiry)
    }

    /// Resolves volatility for a strike/expiry pair.
    #[inline]
    pub fn vol_for(&self, strike: f64, expiry: f64) -> f64 {
        self.vol.vol(strike, expiry)
    }
}

/// Builder for [`Market`].
#[derive(Debug, Clone, Default)]
pub struct MarketBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    dividend_yield: Option<f64>,
    flat_vol: Option<f64>,
    surface: Option<Box<dyn VolSurface>>,
    reference_date: Option<String>,
}

impl MarketBuilder {
    /// Sets the spot price.
    #[inline]
    pub fn spot(mut self, spot: f64) -> Self {
        self.spot = Some(spot);
        self
    }

    /// Sets the flat risk-free rate.
    #[inline]
    pub fn rate(mut self, rate: f64) -> Self {
        self.rate = Some(rate);
        self
    }

    /// Sets the continuous dividend yield.
    #[inline]
    pub fn dividend_yield(mut self, dividend_yield: f64) -> Self {
        self.dividend_yield = Some(dividend_yield);
        self
    }

    /// Sets a flat volatility source.
    #[inline]
    pub fn flat_vol(mut self, vol: f64) -> Self {
        self.flat_vol = Some(vol);
        self.surface = None;
        self
    }

    /// Sets a surface volatility source.
    pub fn vol_surface(mut self, surface: Box<dyn VolSurface>) -> Self {
        self.surface = Some(surface);
        self.flat_vol = None;
        self
    }

    /// Sets an optional reference date.
    pub fn reference_date<S: Into<String>>(mut self, reference_date: S) -> Self {
        self.reference_date = Some(reference_date.into());
        self
    }

    /// Validates and builds a [`Market`].
    pub fn build(self) -> Result<Market, PricingError> {
        let spot = self
            .spot
            .ok_or_else(|| PricingError::InvalidInput("market spot is required".to_string()))?;
        if spot <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market spot must be > 0".to_string(),
            ));
        }

        let rate = self.rate.unwrap_or(0.0);
        let dividend_yield = self.dividend_yield.unwrap_or(0.0);

        let vol = if let Some(surface) = self.surface {
            let any_surface = surface.as_ref() as &dyn Any;
            if let Some(parametric) = any_surface.downcast_ref::<crate::vol::surface::VolSurface>()
            {
                VolSource::Parametric(parametric.clone())
            } else {
                VolSource::Sampled(SampledVolSurface::from_surface(surface.as_ref(), spot))
            }
        } else {
            let flat = self.flat_vol.ok_or_else(|| {
                PricingError::InvalidInput(
                    "either market flat_vol or vol_surface is required".to_string(),
                )
            })?;
            if flat <= 0.0 {
                return Err(PricingError::InvalidInput(
                    "market flat_vol must be > 0".to_string(),
                ));
            }
            VolSource::Flat(flat)
        };

        Ok(Market {
            spot,
            rate,
            dividend_yield,
            vol,
            reference_date: self.reference_date,
        })
    }
}

/// Forward curve snapshot for an asset.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ForwardCurveSnapshot {
    pub asset_id: String,
    pub points: Vec<(f64, f64)>,
}

/// Credit curve snapshot with recovery assumption.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CreditCurveSnapshot {
    pub curve_id: String,
    pub survival_curve: crate::credit::SurvivalCurve,
    pub recovery_rate: f64,
}

/// Serializable market snapshot container.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MarketSnapshot {
    pub snapshot_id: String,
    pub timestamp_unix_ms: i64,
    pub markets: Vec<(String, Market)>,
    pub yield_curves: Vec<(String, crate::rates::YieldCurve)>,
    pub vol_surfaces: Vec<(String, crate::vol::surface::VolSurface)>,
    pub credit_curves: Vec<CreditCurveSnapshot>,
    pub spot_prices: Vec<(String, f64)>,
    pub forward_curves: Vec<ForwardCurveSnapshot>,
}

impl MarketSnapshot {
    pub fn new<S: Into<String>>(snapshot_id: S, timestamp_unix_ms: i64) -> Self {
        Self {
            snapshot_id: snapshot_id.into(),
            timestamp_unix_ms,
            markets: Vec::new(),
            yield_curves: Vec::new(),
            vol_surfaces: Vec::new(),
            credit_curves: Vec::new(),
            spot_prices: Vec::new(),
            forward_curves: Vec::new(),
        }
    }
}
