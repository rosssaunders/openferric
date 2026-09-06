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

use super::dividends::DividendSchedule;

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
///
/// Grid validation is cached: the first successful [`validate`] marks the
/// surface so later calls (one per engine `price` boundary) are O(1) instead
/// of rescanning the whole grid. The token is skipped by serde and reset by
/// `clone`, so deserialized, hand-built, and cloned-then-modified surfaces
/// are always re-checked. Mutating the public fields of an already-validated
/// surface in place is not detected — build a new surface (or clone first)
/// instead.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SampledVolSurface {
    pub strikes: Vec<f64>,
    pub expiries: Vec<f64>,
    pub vols: Vec<Vec<f64>>,
    #[serde(skip)]
    validated: std::sync::atomic::AtomicBool,
}

impl Clone for SampledVolSurface {
    fn clone(&self) -> Self {
        // Reset the token: bump-and-reprice flows clone then mutate, and a
        // carried token would let the mutation skip validation.
        Self {
            strikes: self.strikes.clone(),
            expiries: self.expiries.clone(),
            vols: self.vols.clone(),
            validated: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl PartialEq for SampledVolSurface {
    fn eq(&self, other: &Self) -> bool {
        self.strikes == other.strikes && self.expiries == other.expiries && self.vols == other.vols
    }
}

impl SampledVolSurface {
    /// Validates a sampled surface, including values that may have entered
    /// through deserialization rather than [`SampledVolSurface::new`].
    pub fn validate(&self) -> Result<(), String> {
        use std::sync::atomic::Ordering;
        if self.validated.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.validate_grid()?;
        // Racing validators recompute the same deterministic result; Relaxed
        // is enough for a monotonic set-once flag.
        self.validated.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn validate_grid(&self) -> Result<(), String> {
        if self.strikes.len() < 2 || self.expiries.len() < 2 {
            return Err("sampled surface requires >= 2 strikes and >= 2 expiries".to_string());
        }
        if self
            .strikes
            .iter()
            .any(|strike| !strike.is_finite() || *strike <= 0.0)
        {
            return Err("sampled surface strikes must be finite and > 0".to_string());
        }
        if self
            .expiries
            .iter()
            .any(|expiry| !expiry.is_finite() || *expiry <= 0.0)
        {
            return Err("sampled surface expiries must be finite and > 0".to_string());
        }
        if self.strikes.windows(2).any(|w| w[1] <= w[0]) {
            return Err("sampled surface strikes must be strictly increasing".to_string());
        }
        if self.expiries.windows(2).any(|w| w[1] <= w[0]) {
            return Err("sampled surface expiries must be strictly increasing".to_string());
        }
        if self.vols.len() != self.expiries.len() {
            return Err("sampled surface row count must match expiries".to_string());
        }
        if self.vols.iter().any(|row| row.len() != self.strikes.len()) {
            return Err("sampled surface each row must match strike count".to_string());
        }
        if self
            .vols
            .iter()
            .flatten()
            .any(|vol| !vol.is_finite() || *vol <= 0.0)
        {
            return Err("sampled surface vols must be finite and > 0".to_string());
        }
        Ok(())
    }

    /// Creates a sampled surface from explicit grids.
    pub fn new(strikes: Vec<f64>, expiries: Vec<f64>, vols: Vec<Vec<f64>>) -> Result<Self, String> {
        let surface = Self {
            strikes,
            expiries,
            vols,
            validated: std::sync::atomic::AtomicBool::new(false),
        };
        surface.validate()?;
        Ok(surface)
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

        // First index with grid value > x, minus one: O(log n) bracket lookup.
        let lo = grid.partition_point(|g| *g <= x) - 1;
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
                // Preserve the source value so MarketBuilder/Market::validate
                // rejects non-finite and non-positive market data rather than
                // silently replacing it with an arbitrary volatility.
                row.push(v);
            }
            vols.push(row);
        }

        Self {
            strikes,
            expiries,
            vols,
            validated: std::sync::atomic::AtomicBool::new(false),
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

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Flat(vol) if !vol.is_finite() || *vol <= 0.0 => {
                Err("market flat volatility must be finite and > 0".to_string())
            }
            Self::Flat(_) => Ok(()),
            Self::Parametric(surface) => surface.validate(),
            Self::Sampled(surface) => surface.validate(),
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
    /// Deterministic discrete dividend schedule.
    #[serde(default)]
    pub dividend_schedule: DividendSchedule,
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

    /// Validates the complete snapshot.
    ///
    /// Public fields and serde support allow callers to construct a `Market`
    /// without using [`MarketBuilder`], so pricing engines call this method at
    /// their public boundaries rather than relying on builder-only checks.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.spot.is_finite() || self.spot <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market spot must be finite and > 0".to_string(),
            ));
        }
        if !self.rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "market rate must be finite".to_string(),
            ));
        }
        if !self.dividend_yield.is_finite() {
            return Err(PricingError::InvalidInput(
                "market dividend_yield must be finite".to_string(),
            ));
        }
        self.dividend_schedule
            .validate()
            .map_err(PricingError::InvalidInput)?;
        self.vol.validate().map_err(PricingError::InvalidInput)
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

    /// Returns deterministic discrete dividend schedule.
    #[inline]
    pub fn dividends(&self) -> &DividendSchedule {
        &self.dividend_schedule
    }

    /// Returns `true` if the market carries a non-empty discrete schedule.
    #[inline]
    pub fn has_discrete_dividends(&self) -> bool {
        !self.dividend_schedule.is_empty()
    }

    pub(crate) fn require_continuous_dividends(&self, maturity: f64) -> Result<(), PricingError> {
        if self
            .dividend_schedule
            .events_until(maturity)
            .next()
            .is_some()
        {
            return Err(PricingError::InvalidInput(
                "selected pricing engine does not support discrete dividends before maturity"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Forward price at maturity under continuous yield + discrete schedule.
    #[inline]
    pub fn forward_price(&self, maturity: f64) -> f64 {
        self.dividend_schedule
            .forward_price(self.spot, self.rate, self.dividend_yield, maturity)
    }

    /// Prepaid-forward equivalent spot at maturity.
    #[inline]
    pub fn prepaid_forward_spot(&self, maturity: f64) -> f64 {
        self.dividend_schedule.prepaid_forward_spot(
            self.spot,
            self.rate,
            self.dividend_yield,
            maturity,
        )
    }

    /// Effective continuous dividend yield equivalent at maturity.
    #[inline]
    pub fn effective_dividend_yield(&self, maturity: f64) -> f64 {
        self.dividend_schedule.effective_dividend_yield(
            self.spot,
            self.rate,
            self.dividend_yield,
            maturity,
        )
    }

    /// Escrowed spot adjustment using discrete dividends only (`q=0`).
    #[inline]
    pub fn escrowed_spot_adjustment(&self, maturity: f64) -> f64 {
        self.dividend_schedule
            .escrowed_spot_adjustment(self.spot, self.rate, maturity)
    }

    /// Escrowed-model tradable spot `S*(0)` for an option expiring at `maturity`.
    ///
    /// Early-exercise engines diffuse this component with carry
    /// `rate - dividend_yield` (no discrete-dividend smear) and reconstruct
    /// the observed spot for exercise payoffs via
    /// [`Market::escrowed_reconstruction`].
    #[inline]
    pub fn escrowed_spot(&self, maturity: f64) -> f64 {
        self.dividend_schedule
            .escrowed_spot(self.spot, self.rate, self.dividend_yield, maturity)
    }

    /// Escrowed-model reconstruction coefficients `(P(t), A(t))` at `time`.
    ///
    /// Observed spot is `S(t) = (S*(t) + A(t)) / P(t)`; see
    /// [`crate::market::DividendSchedule::escrowed_reconstruction`].
    #[inline]
    pub fn escrowed_reconstruction(&self, time: f64, maturity: f64) -> (f64, f64) {
        self.dividend_schedule.escrowed_reconstruction(
            time,
            self.rate,
            self.dividend_yield,
            maturity,
        )
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

    /// Resolves volatility and rejects invalid query coordinates or a
    /// non-finite/non-positive surface result.
    ///
    /// A surface can be structurally valid yet overflow for an extreme query,
    /// so pricing boundaries should use this checked form instead of assuming
    /// [`Market::validate`] proves every possible surface evaluation is finite.
    pub fn checked_vol_for(&self, strike: f64, expiry: f64) -> Result<f64, PricingError> {
        if !strike.is_finite() || strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "volatility strike must be finite and > 0".to_string(),
            ));
        }
        if !expiry.is_finite() || expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "volatility expiry must be finite and >= 0".to_string(),
            ));
        }

        let vol = self.vol_for(strike, expiry);
        if !vol.is_finite() || vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility query must return a finite value > 0".to_string(),
            ));
        }
        Ok(vol)
    }
}

/// Builder for [`Market`].
#[derive(Debug, Clone, Default)]
pub struct MarketBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    dividend_yield: Option<f64>,
    dividend_schedule: Option<DividendSchedule>,
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

    /// Sets the deterministic discrete dividend schedule.
    #[inline]
    pub fn dividend_schedule(mut self, dividend_schedule: DividendSchedule) -> Self {
        self.dividend_schedule = Some(dividend_schedule);
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
        if !spot.is_finite() || spot <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market spot must be finite and > 0".to_string(),
            ));
        }

        let rate = self.rate.unwrap_or(0.0);
        if !rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "market rate must be finite".to_string(),
            ));
        }
        let dividend_yield = self.dividend_yield.unwrap_or(0.0);
        if !dividend_yield.is_finite() {
            return Err(PricingError::InvalidInput(
                "market dividend_yield must be finite".to_string(),
            ));
        }
        let dividend_schedule = self.dividend_schedule.unwrap_or_default();
        dividend_schedule
            .validate()
            .map_err(PricingError::InvalidInput)?;

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
            if !flat.is_finite() || flat <= 0.0 {
                return Err(PricingError::InvalidInput(
                    "market flat_vol must be finite and > 0".to_string(),
                ));
            }
            VolSource::Flat(flat)
        };

        let market = Market {
            spot,
            rate,
            dividend_yield,
            dividend_schedule,
            vol,
            reference_date: self.reference_date,
        };
        market.validate()?;
        Ok(market)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct NonFiniteVolSurface;

    impl VolSurface for NonFiniteVolSurface {
        fn vol(&self, _strike: f64, _expiry: f64) -> f64 {
            f64::NAN
        }
    }

    #[derive(Debug, Clone)]
    struct ConstantTraitObjectSurface(f64);

    impl VolSurface for ConstantTraitObjectSurface {
        fn vol(&self, _strike: f64, _expiry: f64) -> f64 {
            self.0
        }
    }

    #[test]
    fn market_builder_rejects_non_finite_inputs() {
        let cases: Vec<(&str, MarketBuilder)> = vec![
            ("NaN spot", Market::builder().spot(f64::NAN).flat_vol(0.2)),
            (
                "infinite spot",
                Market::builder().spot(f64::INFINITY).flat_vol(0.2),
            ),
            (
                "NaN flat_vol",
                Market::builder().spot(100.0).flat_vol(f64::NAN),
            ),
            (
                "NaN rate",
                Market::builder().spot(100.0).rate(f64::NAN).flat_vol(0.2),
            ),
            (
                "NaN dividend_yield",
                Market::builder()
                    .spot(100.0)
                    .dividend_yield(f64::NAN)
                    .flat_vol(0.2),
            ),
        ];

        for (label, builder) in cases {
            assert!(builder.build().is_err(), "{label} must be rejected");
        }
    }

    #[test]
    fn market_builder_accepts_valid_inputs() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .unwrap();
        assert_eq!(market.spot(), 100.0);
    }

    #[test]
    fn direct_market_validation_rejects_all_non_finite_scalar_fields() {
        let valid = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .unwrap();

        for invalid in [
            Market {
                spot: f64::NAN,
                ..valid.clone()
            },
            Market {
                spot: f64::INFINITY,
                ..valid.clone()
            },
            Market {
                rate: f64::NEG_INFINITY,
                ..valid.clone()
            },
            Market {
                dividend_yield: f64::NAN,
                ..valid.clone()
            },
            Market {
                vol: VolSource::Flat(f64::INFINITY),
                ..valid.clone()
            },
        ] {
            assert!(invalid.validate().is_err(), "{invalid:?} must be rejected");
        }
    }

    #[test]
    fn deserialized_sampled_surface_shape_and_finiteness_are_revalidated() {
        let json =
            r#"{"strikes":[90.0,100.0],"expiries":[0.5,1.0],"vols":[[0.2,0.21],[0.22,-0.23]]}"#;
        let surface: SampledVolSurface = serde_json::from_str(json).unwrap();
        assert!(surface.validate().is_err());
    }

    #[test]
    fn sampled_surface_validation_cache_is_reset_on_clone() {
        let surface = SampledVolSurface::new(
            vec![90.0, 100.0],
            vec![0.5, 1.0],
            vec![vec![0.2, 0.21], vec![0.22, 0.23]],
        )
        .unwrap();
        // Cached fast path stays Ok on repeat validation.
        assert!(surface.validate().is_ok());
        assert!(surface.validate().is_ok());

        // A clone must re-check, so mutations after cloning are caught.
        let mut mutated = surface.clone();
        mutated.vols[1][0] = -0.5;
        assert!(mutated.validate().is_err());
    }

    #[test]
    fn market_builder_does_not_mask_non_finite_surface_outputs() {
        let result = Market::builder()
            .spot(100.0)
            .vol_surface(Box::new(NonFiniteVolSurface))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn market_builder_does_not_clamp_non_positive_trait_object_surface_outputs() {
        for value in [0.0, -0.2] {
            let result = Market::builder()
                .spot(100.0)
                .vol_surface(Box::new(ConstantTraitObjectSurface(value)))
                .build();
            assert!(result.is_err(), "surface value {value} must be rejected");
        }
    }

    #[test]
    fn sampled_surface_locate_bounds_brackets_interior_and_edges() {
        let grid = [0.1, 0.5, 1.0, 2.0, 5.0];

        // Below and above the grid clamp to the edges.
        assert_eq!(SampledVolSurface::locate_bounds(&grid, 0.0), (0, 0, 0.0));
        assert_eq!(SampledVolSurface::locate_bounds(&grid, 9.0), (4, 4, 0.0));

        // Interior points bracket correctly with the right weight.
        let (lo, hi, w) = SampledVolSurface::locate_bounds(&grid, 0.75);
        assert_eq!((lo, hi), (1, 2));
        assert!((w - 0.5).abs() < 1.0e-12);

        // Exact interior grid points land on their own bracket start.
        let (lo, hi, w) = SampledVolSurface::locate_bounds(&grid, 1.0);
        assert_eq!((lo, hi), (2, 3));
        assert!(w.abs() < 1.0e-12);
    }
}
