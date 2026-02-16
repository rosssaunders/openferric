/// Convexity adjustment between futures and forward rates:
///
/// `futures_rate ~= forward_rate + 0.5 * sigma^2 * T1 * T2`
pub fn futures_forward_convexity_adjustment(vol: f64, t1: f64, t2: f64) -> f64 {
    if !vol.is_finite() || !t1.is_finite() || !t2.is_finite() || vol < 0.0 || t1 < 0.0 || t2 < 0.0 {
        return f64::NAN;
    }
    0.5 * vol * vol * t1 * t2
}

/// Converts a forward rate into its convexity-adjusted futures rate.
pub fn futures_rate_from_forward(forward_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
    if !forward_rate.is_finite() {
        return f64::NAN;
    }
    forward_rate + futures_forward_convexity_adjustment(vol, t1, t2)
}

/// Converts a futures rate into an adjusted forward rate.
pub fn forward_rate_from_futures(futures_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
    if !futures_rate.is_finite() {
        return f64::NAN;
    }
    futures_rate - futures_forward_convexity_adjustment(vol, t1, t2)
}

/// CMS convexity adjustment approximation for a swap rate fixing in arrears.
///
/// Uses swap-rate volatility and an annuity convexity scalar:
///
/// `cms_adjustment ~= 0.5 * swap_rate * sigma_s^2 * annuity_convexity`
pub fn cms_convexity_adjustment(swap_rate: f64, swap_rate_vol: f64, annuity_convexity: f64) -> f64 {
    if !swap_rate.is_finite()
        || !swap_rate_vol.is_finite()
        || !annuity_convexity.is_finite()
        || swap_rate < 0.0
        || swap_rate_vol < 0.0
        || annuity_convexity < 0.0
    {
        return f64::NAN;
    }

    0.5 * swap_rate * swap_rate_vol * swap_rate_vol * annuity_convexity
}

/// Convexity-adjusted CMS rate fixing in arrears.
pub fn cms_rate_in_arrears(swap_rate: f64, swap_rate_vol: f64, annuity_convexity: f64) -> f64 {
    swap_rate + cms_convexity_adjustment(swap_rate, swap_rate_vol, annuity_convexity)
}

/// Timing adjustment amount when payment date differs from natural fixing date.
///
/// A common approximation is:
///
/// `timing_adj ~= 0.5 * sigma^2 * T_nat * (T_pay - T_nat)`
pub fn timing_adjustment_amount(rate_vol: f64, natural_date: f64, payment_date: f64) -> f64 {
    if !rate_vol.is_finite()
        || !natural_date.is_finite()
        || !payment_date.is_finite()
        || rate_vol < 0.0
        || natural_date < 0.0
        || payment_date < 0.0
    {
        return f64::NAN;
    }

    if (payment_date - natural_date).abs() <= 1.0e-14 {
        return 0.0;
    }
    0.5 * rate_vol * rate_vol * natural_date * (payment_date - natural_date)
}

/// Timing-adjusted rate.
pub fn timing_adjusted_rate(rate: f64, rate_vol: f64, natural_date: f64, payment_date: f64) -> f64 {
    if !rate.is_finite() {
        return f64::NAN;
    }
    rate + timing_adjustment_amount(rate_vol, natural_date, payment_date)
}

/// Quanto drift adjustment for rates:
///
/// `drift_adjustment = -rho * sigma_r * sigma_fx`
pub fn quanto_drift_adjustment(rho: f64, sigma_r: f64, sigma_fx: f64) -> f64 {
    if !rho.is_finite()
        || !sigma_r.is_finite()
        || !sigma_fx.is_finite()
        || rho < -1.0
        || rho > 1.0
        || sigma_r < 0.0
        || sigma_fx < 0.0
    {
        return f64::NAN;
    }
    -rho * sigma_r * sigma_fx
}

/// Adds the quanto correction to a baseline drift.
pub fn quanto_adjusted_drift(baseline_drift: f64, rho: f64, sigma_r: f64, sigma_fx: f64) -> f64 {
    if !baseline_drift.is_finite() {
        return f64::NAN;
    }
    baseline_drift + quanto_drift_adjustment(rho, sigma_r, sigma_fx)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn futures_convexity_adjustment_is_positive() {
        let adj = futures_forward_convexity_adjustment(0.01, 1.0, 1.25);
        assert!(adj > 0.0);
        assert_relative_eq!(adj, 0.000_062_5, epsilon = 1.0e-12);
    }

    #[test]
    fn timing_adjustment_is_zero_when_dates_match() {
        assert_relative_eq!(
            timing_adjustment_amount(0.2, 1.5, 1.5),
            0.0,
            epsilon = 1.0e-16
        );
    }

    #[test]
    fn quanto_adjustment_formula_sign_matches_convention() {
        let adj = quanto_drift_adjustment(0.4, 0.01, 0.12);
        assert_relative_eq!(adj, -0.00048, epsilon = 1.0e-12);
    }
}
