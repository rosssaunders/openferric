//! Module `vol::builder`.
//!
//! Implements builder abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Gatheral (2006), Derman and Kani (1994), static-arbitrage constraints around total variance Eq. (2.2).
//!
//! Key types and purpose: `MarketOptionQuote`, `BuiltVolSurface`, `VolSurfaceBuilder` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
use crate::math::CubicSpline;
use crate::pricing::OptionType;
use crate::vol::forward::{
    AtmSkewTermStructure, ForwardVarianceCurve, ForwardVarianceSource, VixSettings, VixStyleIndex,
    vix_style_index_from_surface,
};
use crate::vol::implied::implied_vol;
use crate::vol::local_vol::{DupireLocalVol, ImpliedVolSurface as LocalVolSurface};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketOptionQuote {
    pub strike: f64,
    pub expiry: f64,
    pub price: f64,
    pub option_type: OptionType,
}

impl MarketOptionQuote {
    pub fn new(strike: f64, expiry: f64, price: f64, option_type: OptionType) -> Self {
        Self {
            strike,
            expiry,
            price,
            option_type,
        }
    }
}

#[derive(Debug, Clone)]
struct ExpirySlice {
    strike_spline: CubicSpline,
}

impl ExpirySlice {
    fn vol_at_strike(&self, strike: f64) -> f64 {
        self.strike_spline.interpolate(strike).max(1e-8)
    }
}

#[derive(Debug, Clone)]
pub struct BuiltVolSurface {
    spot: f64,
    rate: f64,
    expiries: Vec<f64>,
    slices: Vec<ExpirySlice>,
}

impl BuiltVolSurface {
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        if self.slices.is_empty() {
            return f64::NAN;
        }

        let t = expiry.max(1e-8);

        if self.slices.len() == 1 {
            return self.slices[0].vol_at_strike(strike);
        }

        if t <= self.expiries[0] {
            return self.slices[0].vol_at_strike(strike);
        }

        if t >= self.expiries[self.expiries.len() - 1] {
            return self.slices[self.slices.len() - 1].vol_at_strike(strike);
        }

        let i = self
            .expiries
            .windows(2)
            .position(|w| t >= w[0] && t <= w[1])
            .unwrap_or(self.expiries.len() - 2);

        let t0 = self.expiries[i];
        let t1 = self.expiries[i + 1];
        let v0 = self.slices[i].vol_at_strike(strike);
        let v1 = self.slices[i + 1].vol_at_strike(strike);

        // Linear interpolation in total variance.
        let w0 = v0 * v0 * t0;
        let w1 = v1 * v1 * t1;
        let w = w0 + (w1 - w0) * (t - t0) / (t1 - t0);

        (w.max(1e-12) / t).sqrt()
    }

    pub fn local_vol(&self, spot: f64, expiry: f64) -> f64 {
        // Borrow the surface (ImpliedVolSurface is implemented for references),
        // avoiding a deep clone per evaluation. The builder's flat-carry
        // convention is forward(T) = spot * exp(rate * T) with q = 0.
        DupireLocalVol::new(self, self.spot)
            .with_rates(self.rate, 0.0)
            .local_vol(spot, expiry)
    }

    /// Spot used when building the surface.
    pub fn spot(&self) -> f64 {
        self.spot
    }

    /// Continuously compounded rate used when building the surface.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Native expiry grid of the built surface.
    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    /// Forward level under a flat carry assumption implied by `spot` and `rate`.
    pub fn forward_price(&self, expiry: f64) -> f64 {
        self.spot * (self.rate * expiry.max(0.0)).exp()
    }

    /// Builds ATM forward-variance curve from this surface.
    pub fn forward_variance_curve(&self, expiries: &[f64]) -> Result<ForwardVarianceCurve, String> {
        ForwardVarianceCurve::from_surface(self, expiries)
    }

    /// Builds ATM skew term structure from this surface.
    pub fn atm_skew_term_structure(
        &self,
        expiries: &[f64],
    ) -> Result<AtmSkewTermStructure, String> {
        AtmSkewTermStructure::from_surface(self, expiries)
    }

    /// Computes a VIX-style index using this surface and its builder rate.
    pub fn vix_style_index(&self, settings: VixSettings) -> Result<VixStyleIndex, String> {
        vix_style_index_from_surface(self, self.rate, settings)
    }
}

impl LocalVolSurface for BuiltVolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        BuiltVolSurface::implied_vol(self, strike, expiry)
    }
}

impl ForwardVarianceSource for BuiltVolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        BuiltVolSurface::implied_vol(self, strike, expiry)
    }

    fn forward_price(&self, expiry: f64) -> f64 {
        BuiltVolSurface::forward_price(self, expiry)
    }

    fn expiries(&self) -> &[f64] {
        &self.expiries
    }
}

#[derive(Debug, Clone)]
pub struct VolSurfaceBuilder {
    spot: f64,
    rate: f64,
    quotes: Vec<MarketOptionQuote>,
    tol: f64,
    max_iter: usize,
}

impl VolSurfaceBuilder {
    pub fn new(spot: f64, rate: f64) -> Self {
        Self {
            spot,
            rate,
            quotes: Vec::new(),
            tol: 1e-10,
            max_iter: 100,
        }
    }

    pub fn from_quotes(spot: f64, rate: f64, quotes: Vec<MarketOptionQuote>) -> Self {
        Self {
            spot,
            rate,
            quotes,
            tol: 1e-10,
            max_iter: 100,
        }
    }

    pub fn with_solver_params(mut self, tol: f64, max_iter: usize) -> Self {
        self.tol = tol.max(1e-14);
        self.max_iter = max_iter.max(1);
        self
    }

    pub fn add_quote(mut self, quote: MarketOptionQuote) -> Self {
        self.quotes.push(quote);
        self
    }

    pub fn add_quotes(mut self, quotes: Vec<MarketOptionQuote>) -> Self {
        self.quotes.extend(quotes);
        self
    }

    pub fn build(&self) -> Result<BuiltVolSurface, String> {
        if !self.spot.is_finite() || self.spot <= 0.0 {
            return Err("spot must be finite and > 0".to_string());
        }
        if !self.rate.is_finite() {
            return Err("rate must be finite".to_string());
        }

        if self.quotes.is_empty() {
            return Err("quotes cannot be empty".to_string());
        }

        let mut sorted_quotes = self.quotes.clone();
        sorted_quotes.sort_by(|a, b| {
            a.expiry
                .total_cmp(&b.expiry)
                .then(a.strike.total_cmp(&b.strike))
        });

        let mut grouped: Vec<(f64, Vec<MarketOptionQuote>)> = Vec::new();

        for quote in sorted_quotes {
            if !quote.strike.is_finite() || quote.strike <= 0.0 {
                return Err("quote strike must be finite and > 0".to_string());
            }
            if !quote.expiry.is_finite() || quote.expiry <= 0.0 {
                return Err("quote expiry must be finite and > 0".to_string());
            }
            if !quote.price.is_finite() || quote.price <= 0.0 {
                return Err("quote price must be finite and > 0".to_string());
            }

            if let Some((t, bucket)) = grouped.last_mut()
                && (quote.expiry - *t).abs() <= 1e-12
            {
                bucket.push(quote);
            } else {
                grouped.push((quote.expiry, vec![quote]));
            }
        }

        let mut expiries = Vec::with_capacity(grouped.len());
        let mut slices = Vec::with_capacity(grouped.len());

        for (expiry, mut bucket) in grouped {
            bucket.sort_by(|a, b| a.strike.total_cmp(&b.strike));

            if bucket.len() < 2 {
                return Err(format!(
                    "each expiry must have at least two strikes (expiry={expiry})"
                ));
            }

            if bucket.windows(2).any(|w| w[1].strike <= w[0].strike) {
                return Err(format!(
                    "strikes must be strictly increasing per expiry (expiry={expiry})"
                ));
            }

            let mut strikes = Vec::with_capacity(bucket.len());
            let mut vols = Vec::with_capacity(bucket.len());

            for quote in bucket {
                let iv = implied_vol(
                    quote.option_type,
                    self.spot,
                    quote.strike,
                    self.rate,
                    quote.expiry,
                    quote.price,
                    self.tol,
                    self.max_iter,
                )
                .map_err(|err| {
                    format!(
                        "implied vol solve failed at expiry={}, strike={}: {}",
                        quote.expiry, quote.strike, err
                    )
                })?;

                strikes.push(quote.strike);
                vols.push(iv.max(1e-8));
            }

            let strike_spline = CubicSpline::new(strikes.clone(), vols.clone())
                .map_err(|_| format!("failed to build strike spline for expiry={expiry}"))?;

            expiries.push(expiry);
            slices.push(ExpirySlice { strike_spline });
        }

        Ok(BuiltVolSurface {
            spot: self.spot,
            rate: self.rate,
            expiries,
            slices,
        })
    }

    /// Builds a surface and immediately extracts an ATM forward-variance curve.
    pub fn build_with_forward_variance_curve(
        &self,
        expiries: &[f64],
    ) -> Result<(BuiltVolSurface, ForwardVarianceCurve), String> {
        let surface = self.build()?;
        let curve = surface.forward_variance_curve(expiries)?;
        Ok((surface, curve))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn independent_flat_quotes() -> Vec<MarketOptionQuote> {
        // SciPy 1.17.1 `special.ndtr` Black-Scholes premiums for S=100,
        // r=1%, q=0 and sigma=25% (same independently generated grid as the
        // flagship recovery test below, reduced to two strikes per slice).
        [
            (80.0, 0.5, 21.129_603_912_480_505),
            (120.0, 0.5, 1.595_979_359_423_136_4),
            (80.0, 1.0, 22.890_064_143_625_56),
            (120.0, 1.0, 3.947_154_079_494_567),
        ]
        .into_iter()
        .map(|(strike, expiry, price)| {
            MarketOptionQuote::new(strike, expiry, price, OptionType::Call)
        })
        .collect()
    }

    fn constant_slice(vol: f64) -> ExpirySlice {
        ExpirySlice {
            strike_spline: CubicSpline::new(vec![80.0, 120.0], vec![vol, vol]).unwrap(),
        }
    }

    #[test]
    fn builder_recovers_flat_surface_from_independent_scipy_prices() {
        let spot = 100.0;
        let rate = 0.01;
        let flat_vol = 0.25;

        // SciPy 1.17.1 `special.ndtr` Black-Scholes premiums, evaluated
        // independently from OpenFerric for S=100, r=1%, q=0 and sigma=25%.
        // Freezing the premiums avoids the former circular test which created
        // every quote with OpenFerric's own Black-Scholes function and then
        // accepted any recovered volatility within 1e-4.
        let scipy_quotes = [
            (80.0, 0.5, 21.129_603_912_480_505),
            (90.0, 0.5, 13.154_955_980_372_549),
            (100.0, 0.5, 7.277_812_513_480_185),
            (110.0, 0.5, 3.589_238_815_956_854),
            (120.0, 0.5, 1.595_979_359_423_136_4),
            (80.0, 1.0, 22.890_064_143_625_56),
            (90.0, 1.0, 15.830_989_589_188_69),
            (100.0, 1.0, 10.403_539_152_996_622),
            (110.0, 1.0, 6.533_446_175_893_907),
            (120.0, 1.0, 3.947_154_079_494_567),
            (80.0, 2.0, 26.111_311_844_732_15),
            (90.0, 2.0, 19.907_712_441_384_32),
            (100.0, 2.0, 14.904_755_419_137_21),
            (110.0, 2.0, 10.995_381_861_880_077),
            (120.0, 2.0, 8.016_944_922_105_399),
        ];
        let quotes = scipy_quotes
            .into_iter()
            .map(|(strike, expiry, price)| {
                MarketOptionQuote::new(strike, expiry, price, OptionType::Call)
            })
            .collect();

        let surface = VolSurfaceBuilder::from_quotes(spot, rate, quotes)
            .with_solver_params(1.0e-13, 100)
            .build()
            .unwrap();

        for &expiry in &[0.5, 0.75, 1.0, 1.5, 2.0] {
            for &strike in &[80.0, 85.0, 95.0, 100.0, 105.0, 115.0, 120.0] {
                let vol = surface.implied_vol(strike, expiry);
                assert!(
                    (vol - flat_vol).abs() <= 3.0e-13,
                    "strike={strike} expiry={expiry} recovered={vol:.17e}"
                );
            }
        }
    }

    #[test]
    fn built_surface_interpolates_total_variance_and_clamps_expiry_wings() {
        let surface = BuiltVolSurface {
            spot: 100.0,
            rate: 0.03,
            expiries: vec![0.5, 2.0],
            slices: vec![constant_slice(0.20), constant_slice(0.30)],
        };

        assert_eq!(surface.implied_vol(100.0, -1.0), 0.20);
        assert_eq!(surface.implied_vol(100.0, 0.5), 0.20);
        assert_eq!(surface.implied_vol(100.0, 2.0), 0.30);
        assert_eq!(surface.implied_vol(100.0, 3.0), 0.30);

        // At 1.25y the interpolation weight is one half. Total variance is
        // therefore (0.20^2*0.5 + 0.30^2*2.0)/2 = 0.10, and IV=sqrt(0.10/1.25).
        assert_relative_eq!(
            surface.implied_vol(100.0, 1.25),
            0.08_f64.sqrt(),
            epsilon = 4.0 * f64::EPSILON
        );

        let one_slice = BuiltVolSurface {
            spot: 100.0,
            rate: 0.0,
            expiries: vec![1.0],
            slices: vec![constant_slice(0.18)],
        };
        assert_eq!(one_slice.implied_vol(90.0, 10.0), 0.18);

        let empty = BuiltVolSurface {
            spot: 100.0,
            rate: 0.0,
            expiries: Vec::new(),
            slices: Vec::new(),
        };
        assert!(empty.implied_vol(100.0, 1.0).is_nan());
    }

    #[test]
    fn builder_chaining_and_analytics_bridges_cover_public_api() {
        let quotes = independent_flat_quotes();
        let builder = VolSurfaceBuilder::new(100.0, 0.01)
            .with_solver_params(0.0, 0)
            .add_quote(quotes[0])
            .add_quotes(quotes[1..].to_vec());
        assert_eq!(builder.tol, 1.0e-14);
        assert_eq!(builder.max_iter, 1);

        // One iteration is intentionally insufficient for the tight IV solve;
        // rebuilding with production solver parameters exercises the same
        // fluent API without weakening the reference tolerance.
        let builder = builder.with_solver_params(1.0e-13, 100);
        let (surface, curve) = builder
            .build_with_forward_variance_curve(&[0.5, 1.0])
            .unwrap();

        assert_eq!(surface.spot(), 100.0);
        assert_eq!(surface.rate(), 0.01);
        assert_eq!(surface.expiries(), &[0.5, 1.0]);
        assert_eq!(surface.forward_price(-1.0), 100.0);
        assert_relative_eq!(
            surface.forward_price(1.0),
            100.0 * 0.01_f64.exp(),
            epsilon = 8.0 * f64::EPSILON * 100.0
        );

        assert_relative_eq!(
            curve.forward_variance(0.5, 1.0).unwrap(),
            0.25_f64.powi(2),
            epsilon = 3.0e-13
        );
        let direct_curve = surface.forward_variance_curve(&[0.5, 1.0]).unwrap();
        assert_eq!(curve, direct_curve);

        let skew = surface.atm_skew_term_structure(&[0.5, 1.0]).unwrap();
        assert!(skew.points().iter().all(|p| p.skew.abs() <= 2.0e-13));

        let vix = surface
            .vix_style_index(VixSettings {
                target_days: 30.0,
                strike_count: 101,
                log_moneyness_span: 2.0,
            })
            .unwrap();
        assert!(vix.index.is_finite());
        assert!(vix.target_variance >= 0.0);

        let local = surface.local_vol(100.0, 0.75);
        // DupireLocalVol uses finite differences on strike/time; the measured
        // flat-surface differentiation residual is 2.0842e-7 in this grid.
        // The remaining 4.16e-8 is cross-platform libm/roundoff headroom.
        assert_relative_eq!(local, 0.25, epsilon = 2.5e-7);

        // Lock both trait bridges explicitly; these are used by generic local-
        // and forward-volatility consumers.
        assert_relative_eq!(
            LocalVolSurface::implied_vol(&surface, 100.0, 0.75),
            0.25,
            epsilon = 3.0e-13
        );
        assert_relative_eq!(
            ForwardVarianceSource::implied_vol(&surface, 100.0, 0.75),
            0.25,
            epsilon = 3.0e-13
        );
        assert_eq!(ForwardVarianceSource::expiries(&surface), &[0.5, 1.0]);
    }

    #[test]
    fn builder_rejects_nonfinite_and_structurally_invalid_inputs() {
        let valid = independent_flat_quotes();

        assert!(VolSurfaceBuilder::new(100.0, 0.01).build().is_err());
        for spot in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                VolSurfaceBuilder::from_quotes(spot, 0.01, valid.clone())
                    .build()
                    .is_err()
            );
        }
        assert!(
            VolSurfaceBuilder::from_quotes(100.0, f64::NAN, valid.clone())
                .build()
                .is_err()
        );

        let invalid_fields = [
            MarketOptionQuote::new(0.0, 1.0, 1.0, OptionType::Call),
            MarketOptionQuote::new(f64::NAN, 1.0, 1.0, OptionType::Call),
            MarketOptionQuote::new(100.0, 0.0, 1.0, OptionType::Call),
            MarketOptionQuote::new(100.0, f64::INFINITY, 1.0, OptionType::Call),
            MarketOptionQuote::new(100.0, 1.0, 0.0, OptionType::Call),
            MarketOptionQuote::new(100.0, 1.0, f64::NAN, OptionType::Call),
        ];
        for quote in invalid_fields {
            assert!(
                VolSurfaceBuilder::from_quotes(100.0, 0.01, vec![quote])
                    .build()
                    .is_err()
            );
        }

        assert!(
            VolSurfaceBuilder::from_quotes(100.0, 0.01, vec![valid[0]])
                .build()
                .unwrap_err()
                .contains("at least two strikes")
        );
        assert!(
            VolSurfaceBuilder::from_quotes(100.0, 0.01, vec![valid[0], valid[0]])
                .build()
                .unwrap_err()
                .contains("strictly increasing")
        );

        let impossible_premium = vec![
            MarketOptionQuote::new(80.0, 1.0, 150.0, OptionType::Call),
            MarketOptionQuote::new(120.0, 1.0, 150.0, OptionType::Call),
        ];
        assert!(
            VolSurfaceBuilder::from_quotes(100.0, 0.01, impossible_premium)
                .build()
                .unwrap_err()
                .contains("implied vol solve failed")
        );
    }
}
