#[inline]
pub(crate) fn heston_intrinsic(spot: f64, strike: f64, is_call: bool) -> f64 {
    if is_call {
        (spot - strike).max(0.0)
    } else {
        (strike - spot).max(0.0)
    }
}

#[inline]
pub(crate) fn option_price_from_call(
    call_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    maturity: f64,
    is_call: bool,
) -> f64 {
    if is_call {
        call_price
    } else {
        call_price - spot * (-div_yield * maturity).exp() + strike * (-rate * maturity).exp()
    }
}
