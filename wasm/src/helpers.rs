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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heston_intrinsic_call_itm() {
        assert!((heston_intrinsic(110.0, 100.0, true) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn heston_intrinsic_call_otm() {
        assert!((heston_intrinsic(90.0, 100.0, true)).abs() < 1e-10);
    }

    #[test]
    fn heston_intrinsic_put_itm() {
        assert!((heston_intrinsic(90.0, 100.0, false) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn heston_intrinsic_put_otm() {
        assert!((heston_intrinsic(110.0, 100.0, false)).abs() < 1e-10);
    }

    #[test]
    fn option_price_from_call_returns_call_for_is_call() {
        assert!(
            (option_price_from_call(10.0, 100.0, 100.0, 0.05, 0.0, 1.0, true) - 10.0).abs() < 1e-10
        );
    }

    #[test]
    fn option_price_from_call_parity() {
        let call = 10.0;
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.02;
        let t = 1.0;
        let put = option_price_from_call(call, s, k, r, q, t, false);
        // put = call - S*exp(-qT) + K*exp(-rT)
        let expected = call - s * (-q * t).exp() + k * (-r * t).exp();
        assert!((put - expected).abs() < 1e-10);
    }
}
