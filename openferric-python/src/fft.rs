use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;

type HestonFftCache =
    OnceLock<Mutex<Option<([u64; 12], Arc<openferric_core::engines::fft::CarrMadanContext>)>>>;

static HESTON_FFT_CACHE: HestonFftCache = OnceLock::new();

#[inline]
pub(crate) fn heston_fft_cache_key(
    spot: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    params: openferric_core::engines::fft::CarrMadanParams,
) -> [u64; 12] {
    [
        spot.to_bits(),
        expiry.to_bits(),
        rate.to_bits(),
        div_yield.to_bits(),
        v0.to_bits(),
        kappa.to_bits(),
        theta.to_bits(),
        sigma_v.to_bits(),
        rho.to_bits(),
        params.n as u64,
        params.eta.to_bits(),
        params.alpha.to_bits(),
    ]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn heston_fft_prices_cached(
    spot: f64,
    strikes: &[f64],
    expiry: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
) -> Option<Vec<(f64, f64)>> {
    use openferric_core::engines::fft::{CarrMadanContext, CarrMadanParams, HestonCharFn};

    let params = CarrMadanParams::default();
    let key = heston_fft_cache_key(
        spot, expiry, rate, div_yield, v0, kappa, theta, sigma_v, rho, params,
    );
    let cache = HESTON_FFT_CACHE.get_or_init(|| Mutex::new(None));
    if let Some(cached_ctx) = {
        let guard = cache.lock().ok()?;
        let result: Option<Arc<openferric_core::engines::fft::CarrMadanContext>> =
            match guard.as_ref() {
                Some((cached_key, cached_ctx)) if *cached_key == key => {
                    Some(Arc::clone(cached_ctx))
                }
                _ => None,
            };
        result
    } {
        return cached_ctx.price_strikes(strikes).ok();
    }

    let cf = HestonCharFn::new(
        spot, rate, div_yield, expiry, v0, kappa, theta, sigma_v, rho,
    );
    let ctx = Arc::new(CarrMadanContext::new(&cf, rate, expiry, spot, params).ok()?);
    let out = ctx.price_strikes(strikes).ok()?;
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((key, Arc::clone(&ctx)));
    }
    Some(out)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_heston_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
) -> f64 {
    heston_fft_prices_cached(
        spot,
        &[strike],
        expiry,
        rate,
        div_yield,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
    )
    .and_then(|v| v.first().map(|(_, p)| *p))
    .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_heston_fft_prices(
    spot: f64,
    strikes: Vec<f64>,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
) -> Vec<f64> {
    if strikes.is_empty() {
        return Vec::new();
    }
    heston_fft_prices_cached(
        spot, &strikes, expiry, rate, div_yield, v0, kappa, theta, sigma_v, rho,
    )
    .map(|pairs| pairs.into_iter().map(|(_, p)| p).collect())
    .unwrap_or_else(|| vec![f64::NAN; strikes.len()])
}

#[pyfunction]
pub fn py_vg_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    sigma: f64,
    theta_vg: f64,
    nu: f64,
) -> f64 {
    use openferric_core::engines::fft::CarrMadanParams;
    use openferric_core::models::VarianceGamma;
    let vg = VarianceGamma {
        sigma,
        theta: theta_vg,
        nu,
    };
    vg.european_calls_fft(
        spot,
        &[strike],
        rate,
        div_yield,
        expiry,
        CarrMadanParams::default(),
    )
    .ok()
    .and_then(|v| v.first().map(|(_, p)| *p))
    .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_cgmy_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    c: f64,
    g: f64,
    m: f64,
    y: f64,
) -> f64 {
    use openferric_core::engines::fft::CarrMadanParams;
    use openferric_core::models::Cgmy;
    let cgmy = Cgmy { c, g, m, y };
    cgmy.european_calls_fft(
        spot,
        &[strike],
        rate,
        div_yield,
        expiry,
        CarrMadanParams::default(),
    )
    .ok()
    .and_then(|v| v.first().map(|(_, p)| *p))
    .unwrap_or(f64::NAN)
}

#[pyfunction]
pub fn py_nig_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
) -> f64 {
    use openferric_core::engines::fft::CarrMadanParams;
    use openferric_core::models::Nig;
    let nig = Nig { alpha, beta, delta };
    nig.european_calls_fft(
        spot,
        &[strike],
        rate,
        div_yield,
        expiry,
        CarrMadanParams::default(),
    )
    .ok()
    .and_then(|v| v.first().map(|(_, p)| *p))
    .unwrap_or(f64::NAN)
}
