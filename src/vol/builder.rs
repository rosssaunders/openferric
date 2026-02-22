use crate::math::CubicSpline;
use crate::pricing::OptionType;
use crate::vol::implied::implied_vol;
use crate::vol::local_vol::{DupireLocalVol, ImpliedVolSurface as LocalVolSurface};

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
        DupireLocalVol::new(self.clone(), self.spot).local_vol(spot, expiry)
    }
}

impl LocalVolSurface for BuiltVolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        BuiltVolSurface::implied_vol(self, strike, expiry)
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
        if self.spot <= 0.0 {
            return Err("spot must be > 0".to_string());
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
            if quote.strike <= 0.0 {
                return Err("quote strike must be > 0".to_string());
            }
            if quote.expiry <= 0.0 {
                return Err("quote expiry must be > 0".to_string());
            }
            if quote.price <= 0.0 {
                return Err("quote price must be > 0".to_string());
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
            expiries,
            slices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::european::black_scholes_price;
    use approx::assert_relative_eq;

    #[test]
    fn builder_recovers_flat_surface_from_bs_prices() {
        let spot = 100.0;
        let rate = 0.01;
        let flat_vol = 0.25;
        let expiries = [0.5, 1.0, 2.0];
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];

        let mut quotes = Vec::new();
        for &expiry in &expiries {
            for &strike in &strikes {
                let price =
                    black_scholes_price(OptionType::Call, spot, strike, rate, flat_vol, expiry);
                quotes.push(MarketOptionQuote::new(
                    strike,
                    expiry,
                    price,
                    OptionType::Call,
                ));
            }
        }

        let surface = VolSurfaceBuilder::from_quotes(spot, rate, quotes)
            .build()
            .unwrap();

        for &expiry in &[0.5, 0.75, 1.0, 1.5, 2.0] {
            for &strike in &[80.0, 85.0, 95.0, 100.0, 105.0, 115.0, 120.0] {
                let vol = surface.implied_vol(strike, expiry);
                assert_relative_eq!(vol, flat_vol, epsilon = 1e-4);
            }
        }
    }
}
