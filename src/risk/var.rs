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
/// assert!(var_95 >= 0.0);
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
/// assert!(es_95 >= var_95);
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
/// assert!(var_99 > 0.0);
/// ```
pub fn delta_normal_var(
    delta: f64,
    annual_volatility: f64,
    confidence: f64,
    horizon_days: f64,
) -> f64 {
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
/// assert!(es > 2.0);
/// ```
pub fn normal_expected_shortfall(mean_loss: f64, std_dev_loss: f64, confidence: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&confidence),
        "confidence must be in (0,1)"
    );
    assert!(
        std_dev_loss.is_finite() && std_dev_loss >= 0.0,
        "std_dev_loss must be finite and >= 0"
    );
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
        (0.0..1.0).contains(&confidence),
        "confidence must be in (0,1)"
    );
    assert!(
        std_dev_loss.is_finite() && std_dev_loss >= 0.0,
        "std_dev_loss must be finite and >= 0"
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

    let mut forecasts = Vec::with_capacity(returns.len() - window);
    for i in window..returns.len() {
        forecasts.push(historical_var(&returns[(i - window)..i], confidence));
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
        (0.0..1.0).contains(&confidence),
        "confidence must be in (0,1)"
    );
}

fn validate_params(confidence: f64, annual_volatility: f64, horizon_days: f64) {
    assert!(
        (0.0..1.0).contains(&confidence),
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
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, StandardNormal};

    use super::*;

    #[test]
    fn historical_var_matches_standard_normal_quantiles() {
        let mut rng = StdRng::seed_from_u64(42);
        let pnl: Vec<f64> = (0..1000).map(|_| StandardNormal.sample(&mut rng)).collect();

        let var_95 = historical_var(&pnl, 0.95);
        let var_99 = historical_var(&pnl, 0.99);

        assert!((var_95 - 1.645).abs() < 0.2);
        assert!((var_99 - 2.326).abs() < 0.2);
    }

    #[test]
    fn delta_normal_var_matches_reference_value() {
        let var = delta_normal_var(1.0, 0.20, 0.99, 1.0);
        assert_relative_eq!(var, 0.0293, epsilon = 3.0e-4);
    }

    #[test]
    fn normal_expected_shortfall_matches_reference_value() {
        let es = normal_expected_shortfall(0.0, 1.0, 0.99);
        assert_relative_eq!(es, 2.665, epsilon = 5.0e-3);
    }

    #[test]
    fn cornish_fisher_reduces_to_gaussian_for_zero_higher_moments() {
        let cf_var = cornish_fisher_var(0.0, 1.0, 0.0, 0.0, 0.99);
        assert_relative_eq!(cf_var, 2.326, epsilon = 3.0e-3);
    }

    #[test]
    fn price_series_var_wrapper_matches_returns_var() {
        let prices = vec![100.0, 101.0, 99.0, 100.5, 98.5, 99.2, 101.3, 100.8];
        let returns = simple_returns(&prices);

        let via_returns = historical_var(&returns, 0.95);
        let via_prices = historical_var_from_prices(&prices, 0.95, false);
        assert_relative_eq!(via_prices, via_returns, epsilon = 1.0e-14);
    }

    #[test]
    fn backtest_wrapper_runs_and_produces_consistent_dimensions() {
        let mut prices = vec![100.0];
        for i in 1..260 {
            let drift = if i % 37 == 0 { -0.03 } else { 0.001 };
            prices.push(prices[i - 1] * (1.0 + drift));
        }

        let window = 60;
        let confidence = 0.99;
        let forecasts = rolling_historical_var_from_prices(&prices, window, confidence, true);
        let bt = backtest_historical_var_from_prices(&prices, window, confidence, true);

        assert_eq!(forecasts.len(), prices.len() - 1 - window);
        assert!(bt.kupiec.p_value.is_finite());
        assert!(bt.christoffersen.p_value_independence.is_finite());
        assert!(bt.christoffersen.p_value_conditional_coverage.is_finite());
    }
}
