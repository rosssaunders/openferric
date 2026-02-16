//! FFT-based option pricing engines.

mod fft_core;

pub mod carr_madan;
pub mod char_fn;
pub mod frft;

pub use carr_madan::{
    CarrMadanGreeksPoint, CarrMadanParams, DEFAULT_ALPHA, DEFAULT_ETA, DEFAULT_FFT_N,
    carr_madan_fft, carr_madan_fft_greeks, carr_madan_fft_strikes, heston_price_fft,
    interpolate_strike_prices, try_heston_price_fft,
};
pub use char_fn::{
    BlackScholesCharFn, CgmyCharFn, CharacteristicFunction, HestonCharFn, VarianceGammaCharFn,
};
pub use frft::{carr_madan_frft_grid, carr_madan_price_at_strikes, frft};
