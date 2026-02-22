//! OpenFerric is a quantitative-finance library for option pricing, rates, volatility,
//! and risk analytics with both low-level numerical kernels and higher-level product APIs.
//!
//! The crate combines textbook models (Black-Scholes, Black-76, binomial/PDE/Monte Carlo),
//! volatility surface tooling (SABR/SVI/local-vol), rates/credit primitives, and portfolio
//! risk measures under one namespace.
//!
//! References used across modules include:
//! - Hull, *Options, Futures, and Other Derivatives* (11th ed.), notably Ch. 13, 19, 20, 24-26.
//! - Glasserman (2004) for Monte Carlo estimators.
//! - Carr and Madan (1999) for FFT pricing.
//! - Hagan et al. (2002) for SABR asymptotics.
//!
//! Numerical considerations:
//! - FFT/PDE modules expose discretization controls where truncation and stability trade off.
//! - MC modules expose path count and RNG controls; confidence intervals are sampling-driven.
//! - Vol and calibration modules enforce parameter bounds to avoid non-physical fits.
//!
//! When to use this crate vs alternatives:
//! - Use `openferric` when you want one Rust-native library spanning instruments, engines,
//!   volatility surfaces, and risk with reusable components.
//! - Use a narrower crate if you only need one isolated capability (for example, only random
//!   numbers or only vanilla-option closed forms) and want a smaller dependency surface.
//!
//! # Feature Flags
//! - `parallel`: enables Rayon-powered parallel Monte Carlo and grid workflows.
//! - `simd`: enables SIMD-specialized paths where available.
//! - `gpu`: enables `wgpu`-based GPU Monte Carlo engines.
//!
//! # Quick Start
//! Price a Black-Scholes call:
//! ```rust
//! use openferric::core::OptionType;
//! use openferric::pricing::european::black_scholes_price;
//!
//! let px = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);
//! assert!(px > 10.0 && px < 11.0);
//! ```
//!
//! Compute Greeks:
//! ```rust
//! use openferric::core::OptionType;
//! use openferric::pricing::european::black_scholes_greeks;
//!
//! let g = black_scholes_greeks(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);
//! assert!(g.delta > 0.0 && g.gamma > 0.0 && g.vega > 0.0);
//! ```
//!
//! Build a simple discount curve:
//! ```rust
//! use openferric::rates::YieldCurveBuilder;
//!
//! let yc = YieldCurveBuilder::from_deposits(&[(0.5, 0.04), (1.0, 0.045)]);
//! let df_1y = yc.discount_factor(1.0);
//! assert!(df_1y > 0.90 && df_1y < 1.0);
//! ```
//!
//! Work with day-count conventions:
//! ```rust
//! use chrono::NaiveDate;
//! use openferric::rates::{DayCountConvention, year_fraction};
//!
//! let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
//! let end = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
//! let yf = year_fraction(start, end, DayCountConvention::Act365Fixed);
//! assert!((yf - 1.0).abs() < 1.0e-8);
//! ```
//!
//! Invert implied volatility:
//! ```rust
//! use openferric::core::OptionType;
//! use openferric::pricing::european::black_scholes_price;
//! use openferric::vol::implied::implied_vol;
//!
//! let s = 100.0;
//! let k = 105.0;
//! let r = 0.02;
//! let t = 1.0;
//! let sigma_true = 0.25;
//! let market = black_scholes_price(OptionType::Call, s, k, r, sigma_true, t);
//! let sigma = implied_vol(OptionType::Call, s, k, r, t, market, 1.0e-12, 64).unwrap();
//! assert!((sigma - sigma_true).abs() < 1.0e-6);
//! ```
//!
//! Run a simple historical VaR:
//! ```rust
//! use openferric::risk::var::historical_var;
//!
//! let pnl = [-2.0, -1.5, 0.2, 0.4, 1.0, -0.8, 0.1];
//! let var_95 = historical_var(&pnl, 0.95);
//! assert!(var_95 >= 0.0);
//! ```

pub mod core;
pub mod credit;
pub mod engines;
pub mod instruments;
pub mod market;
pub mod math;
pub mod models;
pub mod rates;
pub mod risk;
pub mod vol;

// Legacy modules kept for compatibility with existing consumers.
pub mod greeks;
pub mod mc;
pub mod pricing;

/// Common imports for ergonomic usage.
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use crate::core::*;
    pub use crate::credit::*;
    pub use crate::engines::analytic::*;
    pub use crate::instruments::*;
    pub use crate::market::*;
    pub use crate::rates::*;
}
