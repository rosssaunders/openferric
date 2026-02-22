//! Market data container and volatility source abstractions.

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

    /// Dynamic downcast support used by serialization.
    fn as_any(&self) -> &dyn Any;
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Volatility source for a market snapshot.
#[derive(Debug, Clone)]
pub enum VolSource {
    /// Constant volatility.
    Flat(f64),
    /// Dynamic surface lookup.
    Surface(Box<dyn VolSurface>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SerializableVolSource {
    Flat {
        value: f64,
    },
    Surface {
        surface: crate::vol::surface::VolSurface,
    },
}

impl serde::Serialize for VolSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let repr = match self {
            Self::Flat(value) => SerializableVolSource::Flat { value: *value },
            Self::Surface(surface) => {
                let Some(surface) = surface
                    .as_any()
                    .downcast_ref::<crate::vol::surface::VolSurface>()
                else {
                    return Err(serde::ser::Error::custom(
                        "unable to serialize custom vol surface trait object",
                    ));
                };
                SerializableVolSource::Surface {
                    surface: surface.clone(),
                }
            }
        };
        repr.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for VolSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = SerializableVolSource::deserialize(deserializer)?;
        Ok(match repr {
            SerializableVolSource::Flat { value } => Self::Flat(value),
            SerializableVolSource::Surface { surface } => Self::Surface(Box::new(surface)),
        })
    }
}

impl VolSource {
    /// Returns a volatility value for the requested strike and expiry.
    pub fn vol(&self, strike: f64, expiry: f64) -> f64 {
        match self {
            Self::Flat(v) => *v,
            Self::Surface(surface) => surface.vol(strike, expiry),
        }
    }
}

/// Market snapshot used by all pricing engines.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
            VolSource::Surface(surface)
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
