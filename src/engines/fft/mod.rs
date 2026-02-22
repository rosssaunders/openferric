//! Module `engines::fft::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Carr and Madan (1999), Lewis (2001), Hull (11th ed.) Ch. 19, with FFT damping/inversion forms around Eq. (19.8).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: choose damping/aliasing controls (alpha, grid spacing, FFT size) to balance truncation error against oscillation near strikes.
//!
//! When to use: choose FFT-based routines for dense strike grids under characteristic-function models; use direct quadrature or Monte Carlo for sparse-strike or path-dependent products.

mod fft_core;

pub mod carr_madan;
pub mod char_fn;
pub mod frft;

pub use carr_madan::{
    CarrMadanContext, CarrMadanGreeksPoint, CarrMadanParams, DEFAULT_ALPHA, DEFAULT_ETA,
    DEFAULT_FFT_N, HIGH_RES_FFT_N, carr_madan_fft, carr_madan_fft_complex, carr_madan_fft_greeks,
    carr_madan_fft_strikes, heston_price_fft, interpolate_strike_prices, try_heston_price_fft,
};
pub use char_fn::{
    BlackScholesCharFn, CgmyCharFn, CharacteristicFunction, HestonCharFn, NigCharFn,
    VarianceGammaCharFn,
};
pub use frft::{carr_madan_frft_grid, carr_madan_price_at_strikes, frft};
