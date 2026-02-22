//! Time-series and historical-data utilities for market-risk analytics.
//!
//! This module includes:
//! - return transforms (simple and log),
//! - rolling moments (mean, volatility, skewness, excess kurtosis),
//! - EWMA volatility (RiskMetrics recursion),
//! - realized-volatility estimators from OHLC data,
//! - sample and Ledoit-Wolf-shrunk correlation estimation,
//! - parametric return-distribution fitting (normal, Student-t, skewed Student-t),
//! - autocorrelation/partial-autocorrelation,
//! - VaR backtesting (Kupiec POF and Christoffersen independence/conditional-coverage).
//!
//! References:
//! - J.P. Morgan/Reuters, *RiskMetrics Technical Document* (1996), EWMA volatility.
//! - Parkinson (1980), high-low realized-volatility estimator.
//! - Garman and Klass (1980), OHLC volatility estimator.
//! - Yang and Zhang (2000), drift-independent volatility estimator.
//! - Ledoit and Wolf (2004), well-conditioned covariance/correlation shrinkage.
//! - Kupiec (1995), unconditional coverage test.
//! - Christoffersen (1998), independence and conditional-coverage tests.

use std::f64::consts::{LN_2, PI};

use nalgebra::{DMatrix, SymmetricEigen};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;

const MIN_STD: f64 = 1.0e-12;

/// Result of a normal-distribution fit to returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalFit {
    /// Estimated location (mean).
    pub mean: f64,
    /// Estimated scale (standard deviation).
    pub std_dev: f64,
    /// Maximized log-likelihood.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
}

/// Result of a Student-t fit to returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTFit {
    /// Estimated location parameter.
    pub location: f64,
    /// Estimated scale parameter.
    pub scale: f64,
    /// Estimated degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Maximized log-likelihood.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
}

/// Result of a skewed-Student-t fit to returns.
///
/// The skew specification follows Hansen's standardized skewed Student-t parameterization
/// with skewness parameter `lambda in (-1, 1)` and degrees of freedom `nu > 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewTFit {
    /// Estimated location parameter.
    pub location: f64,
    /// Estimated scale parameter.
    pub scale: f64,
    /// Estimated degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Estimated skewness parameter in `(-1, 1)`.
    pub skew_lambda: f64,
    /// Maximized log-likelihood.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
}

/// Bundle of fitted return distributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReturnDistributionFits {
    pub normal: NormalFit,
    pub student_t: StudentTFit,
    pub skew_t: SkewTFit,
}

impl ReturnDistributionFits {
    /// Returns the model name with the smallest AIC.
    pub fn best_model_by_aic(&self) -> &'static str {
        let mut best = ("normal", self.normal.aic);
        if self.student_t.aic < best.1 {
            best = ("student_t", self.student_t.aic);
        }
        if self.skew_t.aic < best.1 {
            best = ("skew_t", self.skew_t.aic);
        }
        best.0
    }
}

/// Result of Ledoit-Wolf shrunk-correlation estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct LedoitWolfCorrelation {
    /// Shrunk covariance matrix.
    pub covariance: Vec<Vec<f64>>,
    /// Correlation matrix implied by `covariance`.
    pub correlation: Vec<Vec<f64>>,
    /// Shrinkage intensity in `[0, 1]`.
    pub shrinkage: f64,
}

/// Kupiec proportion-of-failures (POF) backtest output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KupiecBacktestResult {
    /// Number of observed VaR breaches.
    pub exceptions: usize,
    /// Expected number of breaches under model calibration.
    pub expected_exceptions: f64,
    /// Likelihood-ratio statistic (chi-square with 1 df).
    pub lr_statistic: f64,
    /// p-value for the test.
    pub p_value: f64,
}

/// Christoffersen independence/conditional-coverage backtest output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChristoffersenBacktestResult {
    /// Number of `0 -> 0` transitions in the hit sequence.
    pub n00: usize,
    /// Number of `0 -> 1` transitions in the hit sequence.
    pub n01: usize,
    /// Number of `1 -> 0` transitions in the hit sequence.
    pub n10: usize,
    /// Number of `1 -> 1` transitions in the hit sequence.
    pub n11: usize,
    /// Independence likelihood-ratio statistic (chi-square with 1 df).
    pub lr_independence: f64,
    /// Conditional-coverage likelihood-ratio statistic (chi-square with 2 df).
    pub lr_conditional_coverage: f64,
    /// p-value for the independence test.
    pub p_value_independence: f64,
    /// p-value for the conditional-coverage test.
    pub p_value_conditional_coverage: f64,
}

/// Combined VaR backtesting summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarBacktestResult {
    pub kupiec: KupiecBacktestResult,
    pub christoffersen: ChristoffersenBacktestResult,
    /// Fraction of observations that breached VaR.
    pub exception_rate: f64,
}

/// Computes simple returns from a price series.
///
/// `r_t = P_t / P_{t-1} - 1`
///
/// # Panics
/// Panics if fewer than 2 prices are supplied, or if any price is non-finite or <= 0.
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    validate_prices(prices);
    prices
        .windows(2)
        .map(|w| w[1] / w[0] - 1.0)
        .collect::<Vec<_>>()
}

/// Computes log returns from a price series.
///
/// `r_t = ln(P_t / P_{t-1})`
///
/// # Panics
/// Panics if fewer than 2 prices are supplied, or if any price is non-finite or <= 0.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    validate_prices(prices);
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect::<Vec<_>>()
}

/// Rolling mean over fixed-size windows.
///
/// Returns a vector of length `series.len() - window + 1`, where each value is the mean of
/// `series[i..i+window]`.
///
/// # Panics
/// Panics if `window == 0` or `window > series.len()` or if non-finite inputs are present.
pub fn rolling_mean(series: &[f64], window: usize) -> Vec<f64> {
    validate_series_and_window(series, window);
    series
        .windows(window)
        .map(sample_mean)
        .collect::<Vec<f64>>()
}

/// Rolling sample standard deviation over fixed-size windows.
///
/// Uses denominator `window - 1` (unbiased sample variance).
///
/// # Panics
/// Panics if `window < 2`, `window > series.len()`, or non-finite inputs are present.
pub fn rolling_std_dev(series: &[f64], window: usize) -> Vec<f64> {
    validate_series_and_window(series, window);
    assert!(window >= 2, "window must be >= 2 for standard deviation");
    series
        .windows(window)
        .map(sample_std_dev)
        .collect::<Vec<f64>>()
}

/// Rolling skewness over fixed-size windows.
///
/// Uses moment ratio `m3 / m2^(3/2)` on each window.
///
/// # Panics
/// Panics if `window < 2`, `window > series.len()`, or non-finite inputs are present.
pub fn rolling_skewness(series: &[f64], window: usize) -> Vec<f64> {
    validate_series_and_window(series, window);
    assert!(window >= 2, "window must be >= 2 for skewness");
    series
        .windows(window)
        .map(|w| {
            let (_, m2, m3, _) = central_moments(w);
            if m2 <= MIN_STD * MIN_STD {
                0.0
            } else {
                m3 / m2.powf(1.5)
            }
        })
        .collect::<Vec<f64>>()
}

/// Rolling excess kurtosis over fixed-size windows.
///
/// Uses moment ratio `m4 / m2^2 - 3` on each window.
///
/// # Panics
/// Panics if `window < 2`, `window > series.len()`, or non-finite inputs are present.
pub fn rolling_excess_kurtosis(series: &[f64], window: usize) -> Vec<f64> {
    validate_series_and_window(series, window);
    assert!(window >= 2, "window must be >= 2 for kurtosis");
    series
        .windows(window)
        .map(|w| {
            let (_, m2, _, m4) = central_moments(w);
            if m2 <= MIN_STD * MIN_STD {
                0.0
            } else {
                m4 / (m2 * m2) - 3.0
            }
        })
        .collect::<Vec<f64>>()
}

/// RiskMetrics-style EWMA volatility series.
///
/// Recursion:
/// `sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2`.
///
/// The initial variance is the sample variance of the return series.
/// Output has the same length as `returns` and is in the same periodicity as inputs.
///
/// # Panics
/// Panics if `returns` is empty, has non-finite values, or if `lambda` is not in `[0, 1)`.
pub fn ewma_volatility(returns: &[f64], lambda: f64) -> Vec<f64> {
    assert!(!returns.is_empty(), "returns must not be empty");
    assert!(
        returns.iter().all(|x| x.is_finite()),
        "returns must be finite"
    );
    assert!(
        lambda.is_finite() && (0.0..1.0).contains(&lambda),
        "lambda must be finite and in [0,1)"
    );

    let mut v = sample_variance(returns).max(MIN_STD * MIN_STD);
    let mut out = Vec::with_capacity(returns.len());
    for &r in returns {
        v = lambda * v + (1.0 - lambda) * r * r;
        out.push(v.max(0.0).sqrt());
    }
    out
}

/// Annualized close-to-close realized volatility from closing prices.
///
/// Computes sample standard deviation of log close-to-close returns and scales by
/// `sqrt(periods_per_year)`.
///
/// # Panics
/// Panics if invalid inputs are supplied.
pub fn realized_vol_close_to_close(closes: &[f64], periods_per_year: f64) -> f64 {
    validate_prices(closes);
    validate_periods_per_year(periods_per_year);
    let r = log_returns(closes);
    sample_std_dev(&r) * periods_per_year.sqrt()
}

/// Annualized Parkinson realized volatility from high/low series.
///
/// Formula:
/// `sigma^2 = (1 / (4 n ln 2)) * sum_t ln(H_t / L_t)^2`.
///
/// # Errors
/// Returns an error for mismatched lengths, too-short series, or invalid price values.
pub fn realized_vol_parkinson(
    highs: &[f64],
    lows: &[f64],
    periods_per_year: f64,
) -> Result<f64, String> {
    validate_periods_per_year(periods_per_year);
    validate_ohlc_pair(highs, lows, "highs", "lows")?;
    let n = highs.len() as f64;
    let mut sum = 0.0;
    for i in 0..highs.len() {
        if highs[i] < lows[i] {
            return Err(format!("high must be >= low at index {i}"));
        }
        let x = (highs[i] / lows[i]).ln();
        sum += x * x;
    }

    let var = sum / (4.0 * n * LN_2);
    Ok((var * periods_per_year).max(0.0).sqrt())
}

/// Annualized Garman-Klass realized volatility from OHLC series.
///
/// Formula:
/// `sigma^2 = (1/n) * sum_t [0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2]`.
///
/// # Errors
/// Returns an error for mismatched lengths, too-short series, or invalid price values.
pub fn realized_vol_garman_klass(
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    periods_per_year: f64,
) -> Result<f64, String> {
    validate_periods_per_year(periods_per_year);
    validate_ohlc(opens, highs, lows, closes)?;

    let n = opens.len() as f64;
    let c = 2.0 * LN_2 - 1.0;

    let mut sum = 0.0;
    for i in 0..opens.len() {
        let hl = (highs[i] / lows[i]).ln();
        let co = (closes[i] / opens[i]).ln();
        let term = 0.5 * hl * hl - c * co * co;
        sum += term;
    }

    let var = (sum / n).max(0.0);
    Ok((var * periods_per_year).sqrt())
}

/// Annualized Yang-Zhang realized volatility from OHLC series.
///
/// This estimator combines overnight, open-close, and Rogers-Satchell components and is
/// robust to drift.
///
/// # Errors
/// Returns an error for mismatched lengths, too-short series, or invalid price values.
pub fn realized_vol_yang_zhang(
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    periods_per_year: f64,
) -> Result<f64, String> {
    validate_periods_per_year(periods_per_year);
    validate_ohlc(opens, highs, lows, closes)?;
    if opens.len() < 3 {
        return Err("yang-zhang volatility requires at least 3 observations".to_string());
    }

    let n = opens.len();

    let mut overnight = Vec::with_capacity(n - 1);
    for t in 1..n {
        overnight.push((opens[t] / closes[t - 1]).ln());
    }

    let mut open_close = Vec::with_capacity(n);
    let mut rs_sum = 0.0;
    for t in 0..n {
        let oc = (closes[t] / opens[t]).ln();
        open_close.push(oc);

        let u = (highs[t] / opens[t]).ln();
        let d = (lows[t] / opens[t]).ln();
        rs_sum += u * (u - oc) + d * (d - oc);
    }

    let sigma_o2 = sample_variance(&overnight);
    let sigma_c2 = sample_variance(&open_close);
    let sigma_rs2 = rs_sum / n as f64;

    let n_f = n as f64;
    let k = 0.34 / (1.34 + (n_f + 1.0) / (n_f - 1.0));

    let var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2;
    Ok((var.max(0.0) * periods_per_year).sqrt())
}

/// Estimates a sample correlation matrix from aligned return series.
///
/// Input layout is `returns[asset][time]`.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or if series are too short.
pub fn sample_correlation_matrix(returns: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let cov = sample_covariance_matrix(returns)?;
    Ok(covariance_to_correlation(&cov))
}

/// Ledoit-Wolf shrinkage estimator for covariance/correlation.
///
/// Shrinks the sample covariance toward a scaled identity target and returns the implied
/// correlation matrix.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or if series are too short.
pub fn ledoit_wolf_correlation_matrix(
    returns: &[Vec<f64>],
) -> Result<LedoitWolfCorrelation, String> {
    validate_panel(returns)?;

    let n_assets = returns.len();
    let n_obs = returns[0].len();

    let means = returns
        .iter()
        .map(|row| sample_mean(row))
        .collect::<Vec<_>>();
    let mut centered = vec![vec![0.0; n_obs]; n_assets];
    for i in 0..n_assets {
        for t in 0..n_obs {
            centered[i][t] = returns[i][t] - means[i];
        }
    }

    // Population covariance is used in the Ledoit-Wolf shrinkage intensity estimator.
    let s_pop = covariance_from_centered(&centered, n_obs as f64);
    let s_sample = covariance_from_centered(&centered, (n_obs - 1) as f64);

    let mu = trace(&s_pop) / n_assets as f64;
    let f = scaled_identity(n_assets, mu);

    let mut pi_hat = 0.0;
    let mut t = 0usize;
    while t < n_obs {
        let mut outer = vec![vec![0.0; n_assets]; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                outer[i][j] = centered[i][t] * centered[j][t];
            }
        }
        pi_hat += frobenius_norm_sq_diff(&outer, &s_pop);
        t += 1;
    }
    pi_hat /= n_obs as f64;

    let delta_hat = frobenius_norm_sq_diff(&s_pop, &f);
    let shrinkage = if delta_hat <= MIN_STD {
        1.0
    } else {
        (pi_hat / delta_hat).clamp(0.0, 1.0)
    };

    let mut cov = vec![vec![0.0; n_assets]; n_assets];
    for i in 0..n_assets {
        for j in 0..n_assets {
            cov[i][j] = (1.0 - shrinkage) * s_sample[i][j] + shrinkage * f[i][j];
        }
    }

    let corr = covariance_to_correlation(&cov);

    Ok(LedoitWolfCorrelation {
        covariance: cov,
        correlation: corr,
        shrinkage,
    })
}

/// Fits normal, Student-t, and skewed-Student-t distributions to returns.
///
/// # Panics
/// Panics if the input series is empty or contains non-finite values.
pub fn fit_return_distributions(returns: &[f64]) -> ReturnDistributionFits {
    ReturnDistributionFits {
        normal: fit_normal_distribution(returns),
        student_t: fit_student_t_distribution(returns),
        skew_t: fit_skew_t_distribution(returns),
    }
}

/// Fits a normal distribution to returns.
///
/// Uses MLE for `(mu, sigma)`.
///
/// # Panics
/// Panics if the input series is empty or contains non-finite values.
pub fn fit_normal_distribution(returns: &[f64]) -> NormalFit {
    validate_nonempty_finite(returns, "returns");
    let n = returns.len() as f64;
    let mean = sample_mean(returns);

    let mut var = 0.0;
    for &x in returns {
        let d = x - mean;
        var += d * d;
    }
    var = (var / n).max(MIN_STD * MIN_STD);
    let std = var.sqrt();

    let log_likelihood = -0.5 * n * ((2.0 * PI * var).ln() + 1.0);
    let k = 2.0;
    let aic = 2.0 * k - 2.0 * log_likelihood;
    let bic = k * n.ln() - 2.0 * log_likelihood;

    NormalFit {
        mean,
        std_dev: std,
        log_likelihood,
        aic,
        bic,
    }
}

/// Fits a Student-t distribution to returns.
///
/// Uses a one-dimensional grid-search over degrees of freedom (`nu > 2`) with location fixed
/// to sample mean and scale mapped from sample standard deviation.
///
/// # Panics
/// Panics if the input series is empty or contains non-finite values.
pub fn fit_student_t_distribution(returns: &[f64]) -> StudentTFit {
    validate_nonempty_finite(returns, "returns");
    let n = returns.len() as f64;
    let mean = sample_mean(returns);
    let std = sample_std_dev(returns).max(MIN_STD);

    let mut best_nu: f64 = 8.0;
    let mut best_scale = std * ((best_nu - 2.0) / best_nu).sqrt();
    let mut best_ll = f64::NEG_INFINITY;

    for step in 0..500 {
        let nu = 2.05 + step as f64 * (197.95 / 499.0);
        let scale = (std * ((nu - 2.0) / nu).sqrt()).max(MIN_STD);
        let ll = student_t_log_likelihood(returns, mean, scale, nu);
        if ll > best_ll {
            best_ll = ll;
            best_nu = nu;
            best_scale = scale;
        }
    }

    let k = 3.0;
    let aic = 2.0 * k - 2.0 * best_ll;
    let bic = k * n.ln() - 2.0 * best_ll;

    StudentTFit {
        location: mean,
        scale: best_scale,
        degrees_of_freedom: best_nu,
        log_likelihood: best_ll,
        aic,
        bic,
    }
}

/// Fits a Hansen-style skewed Student-t distribution to returns.
///
/// Uses a grid-search over `nu > 2` and `lambda in (-1, 1)`.
///
/// # Panics
/// Panics if the input series is empty or contains non-finite values.
pub fn fit_skew_t_distribution(returns: &[f64]) -> SkewTFit {
    validate_nonempty_finite(returns, "returns");
    let n = returns.len() as f64;
    let mean = sample_mean(returns);
    let scale = sample_std_dev(returns).max(MIN_STD);

    let mut best_nu = 8.0;
    let mut best_lambda = 0.0;
    let mut best_ll = f64::NEG_INFINITY;

    for nu_step in 0..140 {
        let nu = 2.05 + nu_step as f64 * (77.95 / 139.0);
        for l_step in 0..61 {
            let lambda = -0.9 + l_step as f64 * (1.8 / 60.0);
            let ll = skew_t_log_likelihood(returns, mean, scale, nu, lambda);
            if ll > best_ll {
                best_ll = ll;
                best_nu = nu;
                best_lambda = lambda;
            }
        }
    }

    let k = 4.0;
    let aic = 2.0 * k - 2.0 * best_ll;
    let bic = k * n.ln() - 2.0 * best_ll;

    SkewTFit {
        location: mean,
        scale,
        degrees_of_freedom: best_nu,
        skew_lambda: best_lambda,
        log_likelihood: best_ll,
        aic,
        bic,
    }
}

/// Autocorrelation function up to `max_lag`.
///
/// Returns a vector of length `max_lag + 1`, with lag-0 equal to 1.
///
/// # Panics
/// Panics if the series is too short, contains non-finite values, or if `max_lag >= len`.
pub fn autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    validate_nonempty_finite(series, "series");
    assert!(
        series.len() >= 2,
        "series must contain at least two observations"
    );
    assert!(max_lag < series.len(), "max_lag must be < series length");

    let n = series.len();
    let mean = sample_mean(series);

    let mut denom = 0.0;
    for &x in series {
        let d = x - mean;
        denom += d * d;
    }
    if denom <= MIN_STD * MIN_STD {
        let mut out = vec![0.0; max_lag + 1];
        out[0] = 1.0;
        return out;
    }

    let mut acf = vec![0.0; max_lag + 1];
    acf[0] = 1.0;
    for lag in 1..=max_lag {
        let mut num = 0.0;
        for t in lag..n {
            num += (series[t] - mean) * (series[t - lag] - mean);
        }
        acf[lag] = num / denom;
    }
    acf
}

/// Partial autocorrelation function up to `max_lag` using Levinson-Durbin recursion.
///
/// Returns a vector of length `max_lag + 1`, with lag-0 equal to 1.
///
/// # Panics
/// Panics if the series is too short, contains non-finite values, or if `max_lag >= len`.
pub fn partial_autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    let acf = autocorrelation(series, max_lag);
    if max_lag == 0 {
        return vec![1.0];
    }

    let mut pacf = vec![0.0; max_lag + 1];
    pacf[0] = 1.0;

    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
    let mut v = vec![0.0; max_lag + 1];
    v[0] = acf[0].max(MIN_STD);

    for k in 1..=max_lag {
        let mut sum = 0.0;
        for j in 1..k {
            sum += phi[k - 1][j] * acf[k - j];
        }

        let mut kk = (acf[k] - sum) / v[k - 1].max(MIN_STD);
        if !kk.is_finite() {
            kk = 0.0;
        }
        kk = kk.clamp(-1.0, 1.0);

        phi[k][k] = kk;
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - kk * phi[k - 1][k - j];
        }

        v[k] = (v[k - 1] * (1.0 - kk * kk)).max(MIN_STD);
        pacf[k] = kk;
    }

    pacf
}

/// Builds a breach-indicator sequence from realized losses and VaR forecasts.
///
/// Convention: a breach is `loss > var`.
///
/// # Panics
/// Panics when dimensions mismatch or non-finite values appear.
pub fn var_breach_indicators(losses: &[f64], var_forecasts: &[f64]) -> Vec<bool> {
    validate_loss_var_series(losses, var_forecasts);
    losses
        .iter()
        .zip(var_forecasts.iter())
        .map(|(l, v)| l > v)
        .collect()
}

/// Kupiec proportion-of-failures VaR backtest.
///
/// Expected breach probability is `1 - confidence`.
///
/// # Panics
/// Panics when input validation fails.
pub fn kupiec_test(losses: &[f64], var_forecasts: &[f64], confidence: f64) -> KupiecBacktestResult {
    validate_loss_var_series(losses, var_forecasts);
    validate_confidence(confidence);

    let hits = var_breach_indicators(losses, var_forecasts);
    let n = hits.len();
    let x = hits.iter().filter(|&&h| h).count();

    let p = (1.0 - confidence).clamp(1.0e-12, 1.0 - 1.0e-12);
    let pi = (x as f64 / n as f64).clamp(1.0e-12, 1.0 - 1.0e-12);

    let ln_l0 = (n - x) as f64 * (1.0 - p).ln() + x as f64 * p.ln();
    let ln_l1 = (n - x) as f64 * (1.0 - pi).ln() + x as f64 * pi.ln();
    let lr = (2.0 * (ln_l1 - ln_l0)).max(0.0);

    let chi = ChiSquared::new(1.0).expect("valid chi-square dof");
    let p_value = 1.0 - chi.cdf(lr);

    KupiecBacktestResult {
        exceptions: x,
        expected_exceptions: n as f64 * p,
        lr_statistic: lr,
        p_value,
    }
}

/// Christoffersen independence and conditional-coverage VaR backtest.
///
/// # Panics
/// Panics when input validation fails.
pub fn christoffersen_test(
    losses: &[f64],
    var_forecasts: &[f64],
    confidence: f64,
) -> ChristoffersenBacktestResult {
    validate_loss_var_series(losses, var_forecasts);
    validate_confidence(confidence);
    assert!(
        losses.len() >= 2,
        "christoffersen test requires at least two observations"
    );

    let hits = var_breach_indicators(losses, var_forecasts);

    let mut n00 = 0usize;
    let mut n01 = 0usize;
    let mut n10 = 0usize;
    let mut n11 = 0usize;

    for t in 1..hits.len() {
        match (hits[t - 1], hits[t]) {
            (false, false) => n00 += 1,
            (false, true) => n01 += 1,
            (true, false) => n10 += 1,
            (true, true) => n11 += 1,
        }
    }

    let p01 = safe_prob(n01, n00 + n01);
    let p11 = safe_prob(n11, n10 + n11);
    let p1 = safe_prob(n01 + n11, n00 + n01 + n10 + n11);

    let ln_l0 = (n00 + n10) as f64 * (1.0 - p1).ln() + (n01 + n11) as f64 * p1.ln();
    let ln_l1 = n00 as f64 * (1.0 - p01).ln()
        + n01 as f64 * p01.ln()
        + n10 as f64 * (1.0 - p11).ln()
        + n11 as f64 * p11.ln();

    let lr_ind = (2.0 * (ln_l1 - ln_l0)).max(0.0);

    let kupiec = kupiec_test(losses, var_forecasts, confidence);
    let lr_cc = kupiec.lr_statistic + lr_ind;

    let chi1 = ChiSquared::new(1.0).expect("valid chi-square dof");
    let chi2 = ChiSquared::new(2.0).expect("valid chi-square dof");

    ChristoffersenBacktestResult {
        n00,
        n01,
        n10,
        n11,
        lr_independence: lr_ind,
        lr_conditional_coverage: lr_cc,
        p_value_independence: 1.0 - chi1.cdf(lr_ind),
        p_value_conditional_coverage: 1.0 - chi2.cdf(lr_cc),
    }
}

/// Runs Kupiec and Christoffersen VaR backtests together.
///
/// # Panics
/// Panics when input validation fails.
pub fn backtest_var(losses: &[f64], var_forecasts: &[f64], confidence: f64) -> VarBacktestResult {
    let kupiec = kupiec_test(losses, var_forecasts, confidence);
    let christoffersen = christoffersen_test(losses, var_forecasts, confidence);
    VarBacktestResult {
        kupiec,
        christoffersen,
        exception_rate: kupiec.exceptions as f64 / losses.len() as f64,
    }
}

/// Condition number of a correlation matrix from eigenvalues.
///
/// Returns `None` for malformed matrices or non-positive spectra.
pub fn correlation_condition_number(corr: &[Vec<f64>]) -> Option<f64> {
    if corr.is_empty() || corr.iter().any(|row| row.len() != corr.len()) {
        return None;
    }

    let n = corr.len();
    let mut data = Vec::with_capacity(n * n);
    for row in corr {
        for &x in row {
            if !x.is_finite() {
                return None;
            }
            data.push(x);
        }
    }

    let m = DMatrix::from_row_slice(n, n, &data);
    let eig = SymmetricEigen::new(m).eigenvalues;

    let mut min_ev = f64::INFINITY;
    let mut max_ev = f64::NEG_INFINITY;
    for &ev in eig.iter() {
        min_ev = min_ev.min(ev);
        max_ev = max_ev.max(ev);
    }
    if min_ev <= 0.0 || !min_ev.is_finite() || !max_ev.is_finite() {
        None
    } else {
        Some(max_ev / min_ev)
    }
}

fn validate_prices(prices: &[f64]) {
    assert!(prices.len() >= 2, "prices must contain at least two values");
    assert!(
        prices.iter().all(|x| x.is_finite() && *x > 0.0),
        "prices must be finite and strictly positive"
    );
}

fn validate_nonempty_finite(values: &[f64], name: &str) {
    assert!(!values.is_empty(), "{name} must not be empty");
    assert!(
        values.iter().all(|x| x.is_finite()),
        "{name} must contain only finite values"
    );
}

fn validate_periods_per_year(periods_per_year: f64) {
    assert!(
        periods_per_year.is_finite() && periods_per_year > 0.0,
        "periods_per_year must be finite and > 0"
    );
}

fn validate_series_and_window(series: &[f64], window: usize) {
    validate_nonempty_finite(series, "series");
    assert!(window > 0, "window must be > 0");
    assert!(window <= series.len(), "window must be <= series length");
}

fn validate_ohlc_pair(a: &[f64], b: &[f64], aname: &str, bname: &str) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{aname} and {bname} must have same length"));
    }
    if a.len() < 2 {
        return Err(format!("{aname}/{bname} series must have length >= 2"));
    }
    if a.iter()
        .chain(b.iter())
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(format!("{aname} and {bname} must be finite and > 0"));
    }
    Ok(())
}

fn validate_ohlc(opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]) -> Result<(), String> {
    let n = opens.len();
    if n < 2 {
        return Err("OHLC series must have length >= 2".to_string());
    }
    if highs.len() != n || lows.len() != n || closes.len() != n {
        return Err("OHLC series lengths must match".to_string());
    }
    if opens
        .iter()
        .chain(highs.iter())
        .chain(lows.iter())
        .chain(closes.iter())
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("OHLC values must be finite and > 0".to_string());
    }
    for i in 0..n {
        if highs[i] < lows[i] {
            return Err(format!("high must be >= low at index {i}"));
        }
        if highs[i] < opens[i] || highs[i] < closes[i] {
            return Err(format!("high must be >= open and close at index {i}"));
        }
        if lows[i] > opens[i] || lows[i] > closes[i] {
            return Err(format!("low must be <= open and close at index {i}"));
        }
    }
    Ok(())
}

fn validate_panel(returns: &[Vec<f64>]) -> Result<(), String> {
    if returns.is_empty() {
        return Err("returns panel must not be empty".to_string());
    }
    let n_obs = returns[0].len();
    if n_obs < 2 {
        return Err("each return series must contain at least two observations".to_string());
    }
    for (i, row) in returns.iter().enumerate() {
        if row.len() != n_obs {
            return Err(format!("row {i} length mismatch"));
        }
        if row.iter().any(|x| !x.is_finite()) {
            return Err(format!("row {i} contains non-finite values"));
        }
    }
    Ok(())
}

fn validate_loss_var_series(losses: &[f64], var_forecasts: &[f64]) {
    assert!(
        !losses.is_empty() && losses.len() == var_forecasts.len(),
        "losses and var_forecasts must be non-empty and have same length"
    );
    assert!(
        losses.iter().all(|x| x.is_finite()),
        "losses must be finite"
    );
    assert!(
        var_forecasts.iter().all(|x| x.is_finite() && *x >= 0.0),
        "var_forecasts must be finite and >= 0"
    );
}

fn validate_confidence(confidence: f64) {
    assert!(
        confidence.is_finite() && (0.0..1.0).contains(&confidence),
        "confidence must be in (0,1)"
    );
}

fn sample_mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64]) -> f64 {
    assert!(values.len() >= 2, "at least 2 observations are required");
    let mean = sample_mean(values);
    let mut sum = 0.0;
    for &x in values {
        let d = x - mean;
        sum += d * d;
    }
    sum / (values.len() as f64 - 1.0)
}

fn sample_std_dev(values: &[f64]) -> f64 {
    sample_variance(values).max(0.0).sqrt()
}

fn central_moments(values: &[f64]) -> (f64, f64, f64, f64) {
    let mean = sample_mean(values);
    let n = values.len() as f64;

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

    (mean, m2 / n, m3 / n, m4 / n)
}

fn covariance_from_centered(centered: &[Vec<f64>], denom: f64) -> Vec<Vec<f64>> {
    let n_assets = centered.len();
    let n_obs = centered[0].len();
    let mut cov = vec![vec![0.0; n_assets]; n_assets];

    for i in 0..n_assets {
        for j in i..n_assets {
            let mut sum = 0.0;
            let mut t = 0usize;
            while t < n_obs {
                sum += centered[i][t] * centered[j][t];
                t += 1;
            }
            let v = sum / denom;
            cov[i][j] = v;
            cov[j][i] = v;
        }
    }
    cov
}

fn sample_covariance_matrix(returns: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    validate_panel(returns)?;

    let n_assets = returns.len();
    let n_obs = returns[0].len();
    let means = returns
        .iter()
        .map(|row| sample_mean(row))
        .collect::<Vec<_>>();

    let mut centered = vec![vec![0.0; n_obs]; n_assets];
    for i in 0..n_assets {
        for t in 0..n_obs {
            centered[i][t] = returns[i][t] - means[i];
        }
    }

    Ok(covariance_from_centered(&centered, (n_obs - 1) as f64))
}

fn covariance_to_correlation(cov: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = cov.len();
    let mut corr = vec![vec![0.0; n]; n];
    let mut stds = vec![0.0; n];

    for i in 0..n {
        stds[i] = cov[i][i].max(0.0).sqrt();
    }

    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            let denom = (stds[i] * stds[j]).max(MIN_STD);
            let rho = (cov[i][j] / denom).clamp(-1.0, 1.0);
            corr[i][j] = rho;
            corr[j][i] = rho;
        }
    }

    corr
}

fn trace(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .sum::<f64>()
}

fn scaled_identity(n: usize, scale: f64) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; n];
    for (i, row) in out.iter_mut().enumerate().take(n) {
        row[i] = scale;
    }
    out
}

fn frobenius_norm_sq_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let d = a[i][j] - b[i][j];
            s += d * d;
        }
    }
    s
}

fn student_t_log_likelihood(values: &[f64], mu: f64, scale: f64, nu: f64) -> f64 {
    values
        .iter()
        .map(|&x| student_t_log_pdf(x, mu, scale, nu))
        .sum::<f64>()
}

fn student_t_log_pdf(x: f64, mu: f64, scale: f64, nu: f64) -> f64 {
    let z = (x - mu) / scale;
    let a = ln_gamma((nu + 1.0) * 0.5) - ln_gamma(nu * 0.5);
    let b = -0.5 * (nu * PI).ln() - scale.ln();
    let c = -0.5 * (nu + 1.0) * (1.0 + (z * z) / nu).ln();
    a + b + c
}

fn skew_t_log_likelihood(values: &[f64], mu: f64, scale: f64, nu: f64, lambda: f64) -> f64 {
    values
        .iter()
        .map(|&x| skew_t_log_pdf(x, mu, scale, nu, lambda))
        .sum::<f64>()
}

fn skew_t_log_pdf(x: f64, mu: f64, scale: f64, nu: f64, lambda: f64) -> f64 {
    let y = (x - mu) / scale;

    let c = (ln_gamma((nu + 1.0) * 0.5) - ln_gamma(nu * 0.5)).exp() / ((PI * (nu - 2.0)).sqrt());
    let a = 4.0 * lambda * c * ((nu - 2.0) / (nu - 1.0));
    let b_sq = 1.0 + 3.0 * lambda * lambda - a * a;
    if b_sq <= MIN_STD {
        return -1.0e12;
    }
    let b = b_sq.sqrt();

    let threshold = -a / b;
    let denom = if y < threshold {
        1.0 - lambda
    } else {
        1.0 + lambda
    };

    if denom <= MIN_STD {
        return -1.0e12;
    }

    let z = (b * y + a) / denom;
    let core = 1.0 + z * z / (nu - 2.0);
    if core <= 0.0 {
        return -1.0e12;
    }

    (b * c).ln() - scale.ln() - 0.5 * (nu + 1.0) * core.ln()
}

fn safe_prob(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.5
    } else {
        (num as f64 / den as f64).clamp(1.0e-12, 1.0 - 1.0e-12)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, StandardNormal, StudentT};

    use super::*;

    fn ar1_series(phi: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = vec![0.0; n];
        for t in 1..n {
            let eps: f64 = StandardNormal.sample(&mut rng);
            x[t] = phi * x[t - 1] + eps;
        }
        x
    }

    #[test]
    fn return_transforms_match_known_values() {
        let prices = vec![100.0, 102.0, 101.0, 103.0];
        let simple = simple_returns(&prices);
        let log = log_returns(&prices);

        assert_relative_eq!(simple[0], 0.02, epsilon = 1.0e-12);
        assert_relative_eq!(simple[1], -0.009_803_921_568_627_45, epsilon = 1.0e-14);
        assert_relative_eq!(simple[2], 0.019_801_980_198_019_82, epsilon = 1.0e-14);

        assert_relative_eq!(log[0], (1.02_f64).ln(), epsilon = 1.0e-12);
        assert_relative_eq!(log[1], (101.0_f64 / 102.0).ln(), epsilon = 1.0e-12);
        assert_relative_eq!(log[2], (103.0_f64 / 101.0).ln(), epsilon = 1.0e-12);
    }

    #[test]
    fn rolling_statistics_are_correct() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let means = rolling_mean(&s, 3);
        assert_eq!(means.len(), 3);
        assert_relative_eq!(means[0], 2.0, epsilon = 1.0e-12);
        assert_relative_eq!(means[1], 3.0, epsilon = 1.0e-12);
        assert_relative_eq!(means[2], 4.0, epsilon = 1.0e-12);

        let std = rolling_std_dev(&s, 3);
        let expected = 1.0_f64;
        assert_relative_eq!(std[0], expected, epsilon = 1.0e-12);
        assert_relative_eq!(std[1], expected, epsilon = 1.0e-12);
        assert_relative_eq!(std[2], expected, epsilon = 1.0e-12);

        let skew = rolling_skewness(&s, 3);
        assert!(skew.iter().all(|x| x.abs() < 1.0e-12));

        let kurt = rolling_excess_kurtosis(&s, 3);
        assert!(kurt.iter().all(|x| *x < 0.0));
    }

    #[test]
    fn ewma_matches_manual_recursion() {
        let r = vec![0.01, -0.02, 0.015, -0.005, 0.03];
        let lambda = 0.94;
        let ew = ewma_volatility(&r, lambda);

        let mut v = sample_variance(&r);
        for i in 0..r.len() {
            v = lambda * v + (1.0 - lambda) * r[i] * r[i];
            assert_relative_eq!(ew[i], v.sqrt(), epsilon = 1.0e-14);
        }
    }

    #[test]
    fn realized_vol_estimators_are_finite_and_consistent() {
        let opens = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let highs = vec![101.0, 103.0, 103.0, 103.0, 104.0, 105.0, 106.0, 106.0];
        let lows = vec![99.5, 100.5, 101.0, 100.8, 102.0, 103.0, 102.8, 104.0];
        let closes = vec![100.8, 102.5, 101.8, 102.2, 103.6, 104.2, 105.3, 105.8];

        let c2c = realized_vol_close_to_close(&closes, 252.0);
        let p = realized_vol_parkinson(&highs, &lows, 252.0).unwrap();
        let gk = realized_vol_garman_klass(&opens, &highs, &lows, &closes, 252.0).unwrap();
        let yz = realized_vol_yang_zhang(&opens, &highs, &lows, &closes, 252.0).unwrap();

        assert!(c2c.is_finite() && c2c > 0.0);
        assert!(p.is_finite() && p > 0.0);
        assert!(gk.is_finite() && gk > 0.0);
        assert!(yz.is_finite() && yz > 0.0);

        // Yang-Zhang should remain same order of magnitude as close-to-close for this sample.
        assert!((yz / c2c) > 0.1 && (yz / c2c) < 3.0);
    }

    #[test]
    fn ledoit_wolf_improves_condition_number() {
        // Highly collinear synthetic returns.
        let mut rng = StdRng::seed_from_u64(7);
        let n = 600;
        let mut f = vec![0.0; n];
        for x in f.iter_mut().take(n) {
            *x = StandardNormal.sample(&mut rng);
        }

        let mut a1 = vec![0.0; n];
        let mut a2 = vec![0.0; n];
        let mut a3 = vec![0.0; n];
        for t in 0..n {
            let e1: f64 = StandardNormal.sample(&mut rng);
            let e2: f64 = StandardNormal.sample(&mut rng);
            let e3: f64 = StandardNormal.sample(&mut rng);
            a1[t] = f[t] + 0.02 * e1;
            a2[t] = 0.98 * f[t] + 0.03 * e2;
            a3[t] = 1.02 * f[t] + 0.02 * e3;
        }

        let panel = vec![a1, a2, a3];
        let sample = sample_correlation_matrix(&panel).unwrap();
        let lw = ledoit_wolf_correlation_matrix(&panel).unwrap();

        let cond_sample = correlation_condition_number(&sample).unwrap();
        let cond_lw = correlation_condition_number(&lw.correlation).unwrap();

        assert!(lw.shrinkage >= 0.0 && lw.shrinkage <= 1.0);
        assert!(cond_lw <= cond_sample + 1.0e-10);
    }

    #[test]
    fn distribution_fits_detect_heavy_tails() {
        let mut rng = StdRng::seed_from_u64(123);
        let t_dist = StudentT::new(5.0).unwrap();
        let mut r = Vec::with_capacity(2000);
        for _ in 0..2000 {
            let x: f64 = t_dist.sample(&mut rng) * 0.01;
            r.push(x);
        }

        let fits = fit_return_distributions(&r);

        assert!(fits.student_t.degrees_of_freedom < 30.0);
        assert!(fits.student_t.log_likelihood > fits.normal.log_likelihood - 1.0e-8);
        assert!(fits.skew_t.log_likelihood > fits.normal.log_likelihood - 1.0e-8);
    }

    #[test]
    fn distribution_fits_capture_skewness_signal() {
        // Build positively skewed heavy-tailed sample via mixture.
        let mut rng = StdRng::seed_from_u64(999);
        let t_dist = StudentT::new(6.0).unwrap();
        let mut r = Vec::with_capacity(2500);
        for i in 0..2500 {
            let base: f64 = t_dist.sample(&mut rng) * 0.008;
            let bump = if i % 10 == 0 { 0.015 } else { 0.0 };
            r.push(base + bump);
        }

        let skew_t = fit_skew_t_distribution(&r);
        assert!(skew_t.skew_lambda > -0.2);
        assert!(skew_t.log_likelihood.is_finite());
    }

    #[test]
    fn acf_and_pacf_match_ar1_behavior() {
        let phi = 0.7;
        let s = ar1_series(phi, 8000, 321);
        let acf = autocorrelation(&s, 5);
        let pacf = partial_autocorrelation(&s, 5);

        assert!((acf[1] - phi).abs() < 0.05);
        assert!((pacf[1] - phi).abs() < 0.05);
        assert!(pacf[2].abs() < 0.10);
        assert!(pacf[3].abs() < 0.10);
    }

    #[test]
    fn kupiec_and_christoffersen_backtests_behave_as_expected() {
        let n = 500;
        let confidence = 0.99;

        // 1% target hit rate: 5 deterministic breaches.
        let mut losses = vec![0.0; n];
        let vars = vec![1.0; n];
        for i in [20usize, 120, 220, 320, 420] {
            losses[i] = 1.5;
        }

        let kup = kupiec_test(&losses, &vars, confidence);
        let chr = christoffersen_test(&losses, &vars, confidence);

        assert_eq!(kup.exceptions, 5);
        assert!(kup.p_value > 0.10);
        assert!(chr.p_value_independence > 0.10);

        let combined = backtest_var(&losses, &vars, confidence);
        assert_relative_eq!(combined.exception_rate, 0.01, epsilon = 1.0e-12);
    }
}
