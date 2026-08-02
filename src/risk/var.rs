//! Value-at-Risk and Expected-Shortfall estimators for historical and parametric workflows.
//!
//! Implemented analytics include:
//! - historical VaR/ES from empirical P&L quantiles,
//! - delta-normal VaR with volatility scaling by `sqrt(horizon_days / 252)`,
//! - delta-gamma VaR via normal moment matching of
//!   `L ~= -Delta r - 0.5 Gamma r^2`,
//! - closed-form normal ES,
//! - Cornish-Fisher VaR (direct moments or moments estimated from P&L),
//! - integration helpers for price-series returns and VaR backtesting.
//!
//! The module uses a loss-positive convention (`loss = -pnl`) and returns non-negative
//! tail metrics.
//!
//! Numerical notes: empirical tail metrics are sample-size sensitive (especially high
//! confidence ES), Cornish-Fisher can be unstable for extreme skew/kurtosis, and all
//! confidence levels must lie in `(0, 1)`.
//!
//! References:
//! - McNeil, Frey, Embrechts, *Quantitative Risk Management* (2005/2015), VaR/ES theory.
//! - J.P. Morgan/Reuters, *RiskMetrics Technical Document* (1996), delta-normal practice.
//! - Cornish and Fisher (1937), quantile expansion.
use crate::math::{
    VarBacktestResult, backtest_var, log_returns, normal_inv_cdf, normal_pdf, simple_returns,
};

const TRADING_DAYS_PER_YEAR: f64 = 252.0;

/// Historical Value-at-Risk from a P&L sample.
///
/// Positive P&L values are profits and negative values are losses.
/// Returned VaR is a positive loss number.
///
/// # Examples
/// ```rust
/// use openferric::risk::var::historical_var;
///
/// let pnl = [-2.0, -1.0, 0.5, 1.0, -0.2];
/// let var_95 = historical_var(&pnl, 0.95);
/// assert!((var_95 - 1.8).abs() < 1.0e-14);
/// ```
pub fn historical_var(pnl: &[f64], confidence: f64) -> f64 {
    validate_inputs(pnl, confidence);
    let mut losses: Vec<f64> = pnl.iter().map(|x| -x).collect();
    empirical_quantile(&mut losses, confidence).max(0.0)
}

/// Historical Expected Shortfall (CVaR) from a P&L sample.
///
/// # Examples
/// ```rust
/// use openferric::risk::var::{historical_expected_shortfall, historical_var};
///
/// let pnl = [-3.0, -2.0, -1.0, 0.5, 1.0];
/// let var_95 = historical_var(&pnl, 0.95);
/// let es_95 = historical_expected_shortfall(&pnl, 0.95);
/// assert!((var_95 - 2.8).abs() < 1.0e-14);
/// assert_eq!(es_95, 3.0);
/// ```
pub fn historical_expected_shortfall(pnl: &[f64], confidence: f64) -> f64 {
    validate_inputs(pnl, confidence);

    let var = historical_var(pnl, confidence);
    let mut tail_sum = 0.0;
    let mut tail_count = 0usize;

    for &x in pnl {
        let loss = -x;
        if loss >= var - 1.0e-12 {
            tail_sum += loss;
            tail_count += 1;
        }
    }

    if tail_count == 0 {
        var
    } else {
        (tail_sum / tail_count as f64).max(0.0)
    }
}

/// Delta-normal parametric VaR.
///
/// # Examples
/// ```rust
/// use openferric::risk::var::delta_normal_var;
///
/// let var_99 = delta_normal_var(1.0, 0.20, 0.99, 1.0);
/// // The implementation uses Acklam's inverse-normal approximation.
/// assert!((var_99 - 0.029_309_228_274_932_753).abs() < 1.0e-10);
/// ```
pub fn delta_normal_var(
    delta: f64,
    annual_volatility: f64,
    confidence: f64,
    horizon_days: f64,
) -> f64 {
    assert!(delta.is_finite(), "delta must be finite");
    validate_params(confidence, annual_volatility, horizon_days);
    let z = normal_inv_cdf(confidence);
    let sigma_h = annual_volatility.abs() * (horizon_days / TRADING_DAYS_PER_YEAR).sqrt();
    delta.abs() * sigma_h * z
}

/// Delta-gamma-normal VaR via normal moment matching for the loss approximation.
pub fn delta_gamma_normal_var(
    delta: f64,
    gamma: f64,
    annual_volatility: f64,
    confidence: f64,
    horizon_days: f64,
) -> f64 {
    assert!(delta.is_finite(), "delta must be finite");
    assert!(gamma.is_finite(), "gamma must be finite");
    validate_params(confidence, annual_volatility, horizon_days);

    let z = normal_inv_cdf(confidence);
    let sigma = annual_volatility.abs() * (horizon_days / TRADING_DAYS_PER_YEAR).sqrt();

    // Loss approximation: L ≈ -Δr - 0.5Γr², with r ~ N(0, σ²).
    let mean_loss = -0.5 * gamma * sigma * sigma;
    let var_loss = delta * delta * sigma * sigma + 0.5 * gamma * gamma * sigma.powi(4);
    let std_loss = var_loss.max(0.0).sqrt();

    (mean_loss + z * std_loss).max(0.0)
}

/// Closed-form Expected Shortfall for a normal loss distribution.
///
/// # Examples
/// ```rust
/// use openferric::risk::var::normal_expected_shortfall;
///
/// let es = normal_expected_shortfall(0.0, 1.0, 0.99);
/// assert!((es - 2.665_214_220_345_806).abs() < 2.0e-8);
/// ```
pub fn normal_expected_shortfall(mean_loss: f64, std_dev_loss: f64, confidence: f64) -> f64 {
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0,1)"
    );
    assert!(
        std_dev_loss.is_finite() && std_dev_loss >= 0.0,
        "std_dev_loss must be finite and >= 0"
    );
    assert!(mean_loss.is_finite(), "mean_loss must be finite");
    let z = normal_inv_cdf(confidence);
    mean_loss + std_dev_loss * normal_pdf(z) / (1.0 - confidence)
}

/// Cornish-Fisher adjusted VaR for a loss distribution.
///
/// `excess_kurtosis` should be kurtosis - 3.
pub fn cornish_fisher_var(
    mean_loss: f64,
    std_dev_loss: f64,
    skewness: f64,
    excess_kurtosis: f64,
    confidence: f64,
) -> f64 {
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0,1)"
    );
    assert!(
        std_dev_loss.is_finite() && std_dev_loss >= 0.0,
        "std_dev_loss must be finite and >= 0"
    );
    assert!(mean_loss.is_finite(), "mean_loss must be finite");
    assert!(skewness.is_finite(), "skewness must be finite");
    assert!(
        excess_kurtosis.is_finite(),
        "excess_kurtosis must be finite"
    );

    let z = normal_inv_cdf(confidence);
    let z2 = z * z;
    let z3 = z2 * z;

    let z_cf = z + (z2 - 1.0) * skewness / 6.0 + (z3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z3 - 5.0 * z) * skewness * skewness / 36.0;

    mean_loss + std_dev_loss * z_cf
}

/// Cornish-Fisher VaR using moments estimated from a P&L sample.
pub fn cornish_fisher_var_from_pnl(pnl: &[f64], confidence: f64) -> f64 {
    validate_inputs(pnl, confidence);
    let losses: Vec<f64> = pnl.iter().map(|x| -x).collect();
    let (mean, std, skew, ex_kurt) = sample_moments(&losses);
    cornish_fisher_var(mean, std, skew, ex_kurt, confidence).max(0.0)
}

/// Historical VaR computed directly from a price series.
///
/// The series is converted to returns (`simple` or `log`) and treated as one-period P&L.
pub fn historical_var_from_prices(prices: &[f64], confidence: f64, use_log_returns: bool) -> f64 {
    let returns = if use_log_returns {
        log_returns(prices)
    } else {
        simple_returns(prices)
    };
    historical_var(&returns, confidence)
}

/// Historical Expected Shortfall computed directly from a price series.
pub fn historical_expected_shortfall_from_prices(
    prices: &[f64],
    confidence: f64,
    use_log_returns: bool,
) -> f64 {
    let returns = if use_log_returns {
        log_returns(prices)
    } else {
        simple_returns(prices)
    };
    historical_expected_shortfall(&returns, confidence)
}

/// Rolling historical-VaR forecast series from prices.
///
/// The output has length `returns.len() - window`, where each point is a one-step-ahead
/// forecast from trailing `window` returns.
pub fn rolling_historical_var_from_prices(
    prices: &[f64],
    window: usize,
    confidence: f64,
    use_log_returns: bool,
) -> Vec<f64> {
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0,1)"
    );
    let returns = if use_log_returns {
        log_returns(prices)
    } else {
        simple_returns(prices)
    };
    assert!(window >= 2, "window must be >= 2");
    assert!(
        window < returns.len(),
        "window must be < number of returns for out-of-sample backtest"
    );

    // Reuse one loss buffer and select the two quantile ranks per window
    // (O(n*w)) instead of allocating and fully sorting every window
    // (O(n*w*log w)). Matches historical_var's interpolated quantile exactly.
    let mut losses = vec![0.0; window];
    let mut forecasts = Vec::with_capacity(returns.len() - window);
    for i in window..returns.len() {
        for (loss, r) in losses.iter_mut().zip(&returns[(i - window)..i]) {
            *loss = -r;
        }
        forecasts.push(empirical_quantile_select(&mut losses, confidence).max(0.0));
    }
    forecasts
}

/// Backtests rolling historical VaR forecasts generated from prices.
pub fn backtest_historical_var_from_prices(
    prices: &[f64],
    window: usize,
    confidence: f64,
    use_log_returns: bool,
) -> VarBacktestResult {
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0,1)"
    );
    let returns = if use_log_returns {
        log_returns(prices)
    } else {
        simple_returns(prices)
    };
    assert!(window >= 2, "window must be >= 2");
    assert!(
        window < returns.len(),
        "window must be < number of returns for out-of-sample backtest"
    );

    let forecasts = rolling_historical_var_from_prices(prices, window, confidence, use_log_returns);
    let losses = returns[window..].iter().map(|r| -r).collect::<Vec<_>>();
    backtest_var(&losses, &forecasts, confidence)
}

fn validate_inputs(pnl: &[f64], confidence: f64) {
    assert!(!pnl.is_empty(), "pnl must not be empty");
    assert!(
        pnl.iter().all(|value| value.is_finite()),
        "pnl must contain only finite values"
    );
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0,1)"
    );
}

fn validate_params(confidence: f64, annual_volatility: f64, horizon_days: f64) {
    assert!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0,1)"
    );
    assert!(
        annual_volatility.is_finite() && annual_volatility >= 0.0,
        "annual_volatility must be finite and >= 0"
    );
    assert!(
        horizon_days.is_finite() && horizon_days > 0.0,
        "horizon_days must be finite and > 0"
    );
}

/// Interpolated empirical quantile via selection instead of a full sort;
/// identical to `empirical_quantile` for any input.
fn empirical_quantile_select(sample: &mut [f64], p: f64) -> f64 {
    let n = sample.len();
    if n == 1 {
        return sample[0];
    }

    let rank = p * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let (_, lo_val, above) = sample.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let lo_val = *lo_val;
    if lo == hi {
        return lo_val;
    }

    // sorted[lo + 1] is the minimum of the partition above the lo-th element.
    let hi_val = above
        .iter()
        .copied()
        .min_by(|a, b| a.total_cmp(b))
        .expect("hi rank exists when lo < hi");
    let w = rank - lo as f64;
    lo_val + w * (hi_val - lo_val)
}

fn empirical_quantile(sample: &mut [f64], p: f64) -> f64 {
    sample.sort_by(|a, b| a.total_cmp(b));
    if sample.len() == 1 {
        return sample[0];
    }

    let rank = p * (sample.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sample[lo]
    } else {
        let w = rank - lo as f64;
        sample[lo] + w * (sample[hi] - sample[lo])
    }
}

fn sample_moments(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &x in values {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    if m2 <= 1.0e-16 {
        return (mean, 0.0, 0.0, 0.0);
    }

    let std = m2.sqrt();
    let skew = m3 / m2.powf(1.5);
    let excess_kurtosis = m4 / (m2 * m2) - 3.0;
    (mean, std, skew, excess_kurtosis)
}

#[cfg(test)]
mod tests {
    #[test]
    fn quantile_select_matches_full_sort_quantile() {
        // Deterministic LCG sample with ties and negatives.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut sample: Vec<f64> = (0..257)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((state >> 33) % 1000) as f64 - 500.0) / 100.0
            })
            .collect();
        for p in [0.0, 0.01, 0.25, 0.5, 0.95, 0.99, 0.999] {
            let mut a = sample.clone();
            let mut b = sample.clone();
            let q_sort = super::empirical_quantile(&mut a, p);
            let q_sel = super::empirical_quantile_select(&mut b, p);
            assert_eq!(q_sort, q_sel, "p={p}");
        }
        sample.truncate(1);
        let mut a = sample.clone();
        let mut b = sample;
        assert_eq!(
            super::empirical_quantile(&mut a, 0.5),
            super::empirical_quantile_select(&mut b, 0.5)
        );
    }

    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn historical_var_matches_exact_interpolated_order_statistics() {
        // Sorted losses are [-4,-3,-2,-1,0,1,2,3,4,5]. The estimator uses
        // rank p*(n-1), hence q(0.90)=4+0.1*(5-4) and q(0.95)=4+0.55*(5-4).
        let pnl = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(historical_var(&pnl, 0.90), 4.1, epsilon = 1e-15);
        assert_relative_eq!(historical_var(&pnl, 0.95), 4.55, epsilon = 1e-15);
    }

    #[test]
    fn delta_normal_var_matches_reference_value() {
        let var = delta_normal_var(1.0, 0.20, 0.99, 1.0);
        // SciPy `special.ndtri(0.99)` propagated through the RiskMetrics scaling.
        assert_relative_eq!(var, 0.029_309_228_274_932_753, epsilon = 1.0e-10);
    }

    #[test]
    fn normal_expected_shortfall_matches_reference_value() {
        let es = normal_expected_shortfall(0.0, 1.0, 0.99);
        assert_relative_eq!(es, 2.665_214_220_345_806, epsilon = 2.0e-8);
    }

    #[test]
    fn cornish_fisher_reduces_to_gaussian_for_zero_higher_moments() {
        let cf_var = cornish_fisher_var(0.0, 1.0, 0.0, 0.0, 0.99);
        assert_relative_eq!(cf_var, 2.326_347_874_040_840_8, epsilon = 5.0e-9);
    }

    #[test]
    fn price_series_var_wrapper_matches_returns_var() {
        let prices = vec![100.0, 101.0, 99.0, 100.5, 98.5, 99.2, 101.3, 100.8];
        let returns = simple_returns(&prices);

        let via_returns = historical_var(&returns, 0.95);
        let via_prices = historical_var_from_prices(&prices, 0.95, false);
        assert_eq!(via_prices, via_returns);
    }

    #[test]
    fn backtest_wrapper_matches_direct_backtest_exactly() {
        let mut prices = vec![100.0];
        for i in 1..260 {
            let drift = if i % 37 == 0 { -0.03 } else { 0.001 };
            prices.push(prices[i - 1] * (1.0 + drift));
        }

        let window = 60;
        let confidence = 0.99;
        let forecasts = rolling_historical_var_from_prices(&prices, window, confidence, true);
        let bt = backtest_historical_var_from_prices(&prices, window, confidence, true);
        let returns = log_returns(&prices);
        let losses = returns[window..].iter().map(|r| -r).collect::<Vec<_>>();
        let expected = backtest_var(&losses, &forecasts, confidence);

        assert_eq!(forecasts.len(), prices.len() - 1 - window);
        assert_eq!(bt, expected);
    }

    #[test]
    fn historical_tail_metrics_cover_singleton_interpolation_and_empty_tail() {
        assert_eq!(historical_var(&[-7.5], 0.975), 7.5);

        // Losses sort to [-1, 2, 3, 4]. At confidence 0.5, VaR is 2.5 and
        // expected shortfall is the exact average of the two losses above it.
        let pnl = [-4.0, -3.0, -2.0, 1.0];
        assert_eq!(historical_var(&pnl, 0.5), 2.5);
        assert_eq!(historical_expected_shortfall(&pnl, 0.5), 3.5);

        // With an all-profit sample the non-negative VaR floor leaves no
        // observations in the loss tail, so ES returns that zero floor.
        let profits = [1.0, 2.0, 3.0];
        assert_eq!(historical_var(&profits, 0.99), 0.0);
        assert_eq!(historical_expected_shortfall(&profits, 0.99), 0.0);
    }

    #[test]
    fn delta_gamma_var_reduces_to_delta_normal_and_has_exact_gamma_sign_shift() {
        let delta_only = delta_gamma_normal_var(1.75, 0.0, 0.30, 0.99, 10.0);
        assert_relative_eq!(
            delta_only,
            delta_normal_var(1.75, 0.30, 0.99, 10.0),
            epsilon = 16.0 * f64::EPSILON
        );
        assert_relative_eq!(
            delta_gamma_normal_var(-1.75, 0.0, 0.30, 0.99, 10.0),
            delta_only,
            epsilon = 16.0 * f64::EPSILON
        );

        let gamma = 2.0;
        let annual_volatility = 0.30;
        let horizon_days = 10.0;
        let positive_gamma =
            delta_gamma_normal_var(0.0, gamma, annual_volatility, 0.99, horizon_days);
        let negative_gamma =
            delta_gamma_normal_var(0.0, -gamma, annual_volatility, 0.99, horizon_days);
        let sigma_squared = annual_volatility * annual_volatility * horizon_days / 252.0;
        assert_relative_eq!(
            negative_gamma - positive_gamma,
            gamma * sigma_squared,
            epsilon = 8.0 * f64::EPSILON
        );

        assert_eq!(delta_gamma_normal_var(0.0, 0.0, 0.20, 0.99, 1.0), 0.0);
    }

    #[test]
    fn zero_dispersion_tail_formulas_and_sample_moment_paths_are_exact() {
        assert_eq!(normal_expected_shortfall(3.25, 0.0, 0.975), 3.25);
        assert_eq!(cornish_fisher_var(3.25, 0.0, 2.0, 5.0, 0.975), 3.25);

        let constant_losses = [-2.0; 5];
        assert_eq!(cornish_fisher_var_from_pnl(&constant_losses, 0.99), 2.0);
        assert_eq!(cornish_fisher_var_from_pnl(&[2.0; 5], 0.99), 0.0);
        assert_eq!(sample_moments(&[]), (0.0, 0.0, 0.0, 0.0));

        // Population moments of [-2,-1,0,1,2] are mean=0, variance=2,
        // skew=0 and excess kurtosis=-1.3.
        let symmetric = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let moments = sample_moments(&symmetric);
        assert_eq!(moments.0, 0.0);
        assert_relative_eq!(moments.1, 2.0_f64.sqrt(), epsilon = f64::EPSILON);
        assert_eq!(moments.2, 0.0);
        assert_relative_eq!(moments.3, -1.3, epsilon = 2.0 * f64::EPSILON);
        assert_relative_eq!(
            cornish_fisher_var_from_pnl(&symmetric, 0.95),
            cornish_fisher_var(0.0, 2.0_f64.sqrt(), 0.0, -1.3, 0.95),
            epsilon = 16.0 * f64::EPSILON
        );
    }

    #[test]
    fn price_series_wrappers_match_explicit_simple_and_log_return_samples() {
        let prices = [100.0, 102.0, 99.0, 103.0, 101.0, 104.0, 100.0];
        let simple = simple_returns(&prices);
        let log = log_returns(&prices);

        assert_eq!(
            historical_var_from_prices(&prices, 0.80, false),
            historical_var(&simple, 0.80)
        );
        assert_eq!(
            historical_var_from_prices(&prices, 0.80, true),
            historical_var(&log, 0.80)
        );
        assert_eq!(
            historical_expected_shortfall_from_prices(&prices, 0.80, false),
            historical_expected_shortfall(&simple, 0.80)
        );
        assert_eq!(
            historical_expected_shortfall_from_prices(&prices, 0.80, true),
            historical_expected_shortfall(&log, 0.80)
        );
    }

    #[test]
    fn rolling_forecasts_match_each_explicit_trailing_window_for_both_return_types() {
        let prices = [
            100.0, 102.0, 101.0, 104.0, 100.0, 105.0, 103.0, 107.0, 106.0,
        ];
        let window = 3;
        let confidence = 0.75;

        for use_log_returns in [false, true] {
            let returns = if use_log_returns {
                log_returns(&prices)
            } else {
                simple_returns(&prices)
            };
            let forecasts =
                rolling_historical_var_from_prices(&prices, window, confidence, use_log_returns);
            let expected = (window..returns.len())
                .map(|i| historical_var(&returns[(i - window)..i], confidence))
                .collect::<Vec<_>>();
            assert_eq!(forecasts, expected);

            let losses = returns[window..].iter().map(|r| -r).collect::<Vec<_>>();
            assert_eq!(
                backtest_historical_var_from_prices(&prices, window, confidence, use_log_returns,),
                backtest_var(&losses, &expected, confidence)
            );
        }
    }

    #[test]
    fn var_entry_points_reject_non_finite_and_shape_invalid_inputs() {
        fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
            std::panic::catch_unwind(f).is_err()
        }

        assert!(panics(|| {
            historical_var(&[], 0.95);
        }));
        assert!(panics(|| {
            historical_var(&[0.0, f64::NAN], 0.95);
        }));
        assert!(panics(|| {
            historical_expected_shortfall(&[0.0, f64::INFINITY], 0.95);
        }));
        for confidence in [0.0, 1.0, f64::NAN] {
            assert!(panics(|| {
                historical_var(&[0.0], confidence);
            }));
        }

        for (delta, vol, confidence, horizon) in [
            (f64::NAN, 0.2, 0.99, 1.0),
            (1.0, f64::INFINITY, 0.99, 1.0),
            (1.0, -0.2, 0.99, 1.0),
            (1.0, 0.2, 1.0, 1.0),
            (1.0, 0.2, 0.99, 0.0),
            (1.0, 0.2, 0.99, f64::NAN),
        ] {
            assert!(panics(|| {
                delta_normal_var(delta, vol, confidence, horizon);
            }));
        }
        assert!(panics(|| {
            delta_gamma_normal_var(1.0, f64::NAN, 0.2, 0.99, 1.0);
        }));

        for (mean, std_dev, confidence) in [
            (f64::NAN, 1.0, 0.95),
            (0.0, f64::INFINITY, 0.95),
            (0.0, -1.0, 0.95),
            (0.0, 1.0, f64::NAN),
        ] {
            assert!(panics(|| {
                normal_expected_shortfall(mean, std_dev, confidence);
            }));
        }
        for (mean, std_dev, skew, kurtosis, confidence) in [
            (f64::INFINITY, 1.0, 0.0, 0.0, 0.95),
            (0.0, -1.0, 0.0, 0.0, 0.95),
            (0.0, 1.0, f64::NAN, 0.0, 0.95),
            (0.0, 1.0, 0.0, f64::INFINITY, 0.95),
            (0.0, 1.0, 0.0, 0.0, 1.0),
        ] {
            assert!(panics(|| {
                cornish_fisher_var(mean, std_dev, skew, kurtosis, confidence);
            }));
        }

        let prices = [100.0, 101.0, 102.0, 103.0];
        for window in [1, prices.len() - 1] {
            assert!(panics(|| {
                rolling_historical_var_from_prices(&prices, window, 0.95, false);
            }));
            assert!(panics(|| {
                backtest_historical_var_from_prices(&prices, window, 0.95, true);
            }));
        }
        assert!(panics(|| {
            rolling_historical_var_from_prices(&prices, 2, 0.0, false);
        }));
        assert!(panics(|| {
            backtest_historical_var_from_prices(&prices, 2, 1.0, false);
        }));
    }
}
