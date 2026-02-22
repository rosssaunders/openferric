//! Interpolation routines for rates and curve construction.
//!
//! This module provides production-oriented interpolators used in fixed-income
//! analytics, including shape-preserving local schemes and parametric curve
//! families.
//!
//! References:
//! - Hagan and West (2006), *Interpolation Methods for Curve Construction*.
//! - Fritsch and Carlson (1980), monotone piecewise cubic interpolation.
//! - Nelson and Siegel (1987), parsimonious yield-curve model.
//! - Svensson (1994), extended Nelson-Siegel specification.
//! - EIOPA, Smith-Wilson technical documentation for Solvency II.

use nalgebra::{DMatrix, DVector};

/// Extrapolation behavior outside the calibrated node range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMode {
    /// Keep the endpoint value constant.
    Flat,
    /// Extend using endpoint tangent/slope.
    Linear,
    /// Return an error outside node range.
    Error,
}

/// Errors returned by interpolators.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationError {
    InvalidInput(&'static str),
    ExtrapolationDisabled,
    SingularSystem,
    NonConvergence,
}

/// Common interpolation interface.
pub trait Interpolator {
    /// Returns interpolated value `y(x)`.
    fn value(&self, x: f64) -> Result<f64, InterpolationError>;

    /// Returns first derivative `dy/dx`.
    fn derivative(&self, x: f64) -> Result<f64, InterpolationError>;

    /// Returns Jacobian `dy(x) / d(input_i)` with respect to calibration inputs.
    fn jacobian(&self, x: f64) -> Result<Vec<f64>, InterpolationError>;

    /// Returns interpolation abscissas.
    fn x(&self) -> &[f64];

    /// Returns interpolation ordinates or calibration inputs.
    fn y(&self) -> &[f64];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryLocation {
    Left,
    Inside(usize),
    Right,
}

fn validate_xy(x: &[f64], y: &[f64], min_len: usize) -> Result<(), InterpolationError> {
    if x.len() != y.len() {
        return Err(InterpolationError::InvalidInput(
            "x and y must have same length",
        ));
    }
    if x.len() < min_len {
        return Err(InterpolationError::InvalidInput(
            "not enough interpolation nodes",
        ));
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(InterpolationError::InvalidInput(
            "x must be strictly increasing",
        ));
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(InterpolationError::InvalidInput("x and y must be finite"));
    }
    Ok(())
}

fn validate_positive_y(y: &[f64]) -> Result<(), InterpolationError> {
    if y.iter().any(|v| *v <= 0.0) {
        return Err(InterpolationError::InvalidInput(
            "y must be strictly positive",
        ));
    }
    Ok(())
}

fn query_location(x: &[f64], xq: f64) -> QueryLocation {
    if xq < x[0] {
        return QueryLocation::Left;
    }
    if xq > x[x.len() - 1] {
        return QueryLocation::Right;
    }
    let idx = x.partition_point(|v| *v <= xq);
    if idx == 0 {
        QueryLocation::Inside(0)
    } else if idx >= x.len() {
        QueryLocation::Inside(x.len() - 2)
    } else {
        QueryLocation::Inside(idx - 1)
    }
}

#[inline]
fn linear_weights(x0: f64, x1: f64, xq: f64) -> (f64, f64) {
    let w = if (x1 - x0).abs() <= f64::EPSILON {
        0.0
    } else {
        (xq - x0) / (x1 - x0)
    };
    (1.0 - w, w)
}

fn pchip_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 2 {
        let m = (y[1] - y[0]) / (x[1] - x[0]);
        return vec![m, m];
    }

    let mut h = vec![0.0; n - 1];
    let mut delta = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    let mut d = vec![0.0; n];

    for k in 1..(n - 1) {
        if delta[k - 1] * delta[k] <= 0.0 {
            d[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
        }
    }

    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
    if d[0].signum() != delta[0].signum() {
        d[0] = 0.0;
    } else if delta[0].signum() != delta[1].signum() && d[0].abs() > 3.0 * delta[0].abs() {
        d[0] = 3.0 * delta[0];
    }

    let m = n - 1;
    d[m] = ((2.0 * h[m - 1] + h[m - 2]) * delta[m - 1] - h[m - 1] * delta[m - 2])
        / (h[m - 1] + h[m - 2]);
    if d[m].signum() != delta[m - 1].signum() {
        d[m] = 0.0;
    } else if delta[m - 1].signum() != delta[m - 2].signum()
        && d[m].abs() > 3.0 * delta[m - 1].abs()
    {
        d[m] = 3.0 * delta[m - 1];
    }

    d
}

fn hagan_west_like_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 2 {
        let m = (y[1] - y[0]) / (x[1] - x[0]);
        return vec![m, m];
    }

    let mut h = vec![0.0; n - 1];
    let mut s = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        h[i] = x[i + 1] - x[i];
        s[i] = (y[i + 1] - y[i]) / h[i];
    }

    let mut d = vec![0.0; n];
    d[0] = s[0];
    d[n - 1] = s[n - 2];

    for i in 1..(n - 1) {
        let avg = (h[i] * s[i - 1] + h[i - 1] * s[i]) / (h[i - 1] + h[i]);
        if s[i - 1] * s[i] <= 0.0 {
            d[i] = 0.0;
        } else {
            let bound = 3.0 * s[i - 1].abs().min(s[i].abs());
            d[i] = avg.signum() * avg.abs().min(bound);
        }
    }

    d
}

#[inline]
fn hermite_eval(x0: f64, x1: f64, y0: f64, y1: f64, m0: f64, m1: f64, xq: f64) -> f64 {
    let h = x1 - x0;
    let s = (xq - x0) / h;
    let s2 = s * s;
    let s3 = s2 * s;

    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;

    h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
}

#[inline]
fn hermite_eval_derivative(x0: f64, x1: f64, y0: f64, y1: f64, m0: f64, m1: f64, xq: f64) -> f64 {
    let h = x1 - x0;
    let s = (xq - x0) / h;
    let s2 = s * s;

    ((6.0 * s2 - 6.0 * s) * y0 + (-6.0 * s2 + 6.0 * s) * y1) / h
        + (3.0 * s2 - 4.0 * s + 1.0) * m0
        + (3.0 * s2 - 2.0 * s) * m1
}

fn finite_difference_jacobian<F>(
    base: &[f64],
    mut evaluator: F,
) -> Result<Vec<f64>, InterpolationError>
where
    F: FnMut(&[f64]) -> Result<f64, InterpolationError>,
{
    let mut out = vec![0.0; base.len()];
    for i in 0..base.len() {
        let h = 1.0e-6 * base[i].abs().max(1.0);
        let mut up = base.to_vec();
        up[i] += h;
        let up_v = evaluator(&up)?;

        let mut dn = base.to_vec();
        dn[i] -= h;
        let dn_v = evaluator(&dn)?;

        out[i] = (up_v - dn_v) / (2.0 * h);
    }
    Ok(out)
}

/// Piecewise-linear interpolation in `y`.
#[derive(Debug, Clone)]
pub struct LinearInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl LinearInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        Ok(Self {
            x,
            y,
            extrapolation,
        })
    }
}

impl Interpolator for LinearInterpolator {
    fn value(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(self.y[0]),
                ExtrapolationMode::Linear => {
                    let (w0, w1) = linear_weights(self.x[0], self.x[1], xq);
                    Ok(w0 * self.y[0] + w1 * self.y[1])
                }
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(self.y[n - 1]),
                    ExtrapolationMode::Linear => {
                        let (w0, w1) = linear_weights(self.x[n - 2], self.x[n - 1], xq);
                        Ok(w0 * self.y[n - 2] + w1 * self.y[n - 1])
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => {
                let (w0, w1) = linear_weights(self.x[i], self.x[i + 1], xq);
                Ok(w0 * self.y[i] + w1 * self.y[i + 1])
            }
        }
    }

    fn derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(0.0),
                ExtrapolationMode::Linear => Ok((self.y[1] - self.y[0]) / (self.x[1] - self.x[0])),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(0.0),
                    ExtrapolationMode::Linear => {
                        Ok((self.y[n - 1] - self.y[n - 2]) / (self.x[n - 1] - self.x[n - 2]))
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => {
                Ok((self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i]))
            }
        }
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        let mut j = vec![0.0; self.y.len()];
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => {
                    j[0] = 1.0;
                    Ok(j)
                }
                ExtrapolationMode::Linear => {
                    let (w0, w1) = linear_weights(self.x[0], self.x[1], xq);
                    j[0] = w0;
                    j[1] = w1;
                    Ok(j)
                }
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.y.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => {
                        j[n - 1] = 1.0;
                        Ok(j)
                    }
                    ExtrapolationMode::Linear => {
                        let (w0, w1) = linear_weights(self.x[n - 2], self.x[n - 1], xq);
                        j[n - 2] = w0;
                        j[n - 1] = w1;
                        Ok(j)
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => {
                let (w0, w1) = linear_weights(self.x[i], self.x[i + 1], xq);
                j[i] = w0;
                j[i + 1] = w1;
                Ok(j)
            }
        }
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

/// Log-linear interpolation in strictly positive `y`.
#[derive(Debug, Clone)]
pub struct LogLinearInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    log_y: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl LogLinearInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        validate_positive_y(&y)?;
        let log_y = y.iter().map(|v| v.ln()).collect();
        Ok(Self {
            x,
            y,
            log_y,
            extrapolation,
        })
    }
}

impl Interpolator for LogLinearInterpolator {
    fn value(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(self.y[0]),
                ExtrapolationMode::Linear => {
                    let (w0, w1) = linear_weights(self.x[0], self.x[1], xq);
                    Ok((w0 * self.log_y[0] + w1 * self.log_y[1]).exp())
                }
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(self.y[n - 1]),
                    ExtrapolationMode::Linear => {
                        let (w0, w1) = linear_weights(self.x[n - 2], self.x[n - 1], xq);
                        Ok((w0 * self.log_y[n - 2] + w1 * self.log_y[n - 1]).exp())
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => {
                let (w0, w1) = linear_weights(self.x[i], self.x[i + 1], xq);
                Ok((w0 * self.log_y[i] + w1 * self.log_y[i + 1]).exp())
            }
        }
    }

    fn derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        let yq = self.value(xq)?;
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(0.0),
                ExtrapolationMode::Linear => {
                    Ok(yq * (self.log_y[1] - self.log_y[0]) / (self.x[1] - self.x[0]))
                }
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(0.0),
                    ExtrapolationMode::Linear => Ok(yq * (self.log_y[n - 1] - self.log_y[n - 2])
                        / (self.x[n - 1] - self.x[n - 2])),
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => {
                Ok(yq * (self.log_y[i + 1] - self.log_y[i]) / (self.x[i + 1] - self.x[i]))
            }
        }
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        let mut j = vec![0.0; self.y.len()];
        let yq = self.value(xq)?;

        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => {
                    j[0] = 1.0;
                    Ok(j)
                }
                ExtrapolationMode::Linear => {
                    let (w0, w1) = linear_weights(self.x[0], self.x[1], xq);
                    j[0] = yq * w0 / self.y[0];
                    j[1] = yq * w1 / self.y[1];
                    Ok(j)
                }
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.y.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => {
                        j[n - 1] = 1.0;
                        Ok(j)
                    }
                    ExtrapolationMode::Linear => {
                        let (w0, w1) = linear_weights(self.x[n - 2], self.x[n - 1], xq);
                        j[n - 2] = yq * w0 / self.y[n - 2];
                        j[n - 1] = yq * w1 / self.y[n - 1];
                        Ok(j)
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => {
                let (w0, w1) = linear_weights(self.x[i], self.x[i + 1], xq);
                j[i] = yq * w0 / self.y[i];
                j[i + 1] = yq * w1 / self.y[i + 1];
                Ok(j)
            }
        }
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

/// Hagan-West inspired monotone-convex interpolation via filtered cubic Hermite segments.
#[derive(Debug, Clone)]
pub struct MonotoneConvexInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    slopes: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl MonotoneConvexInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        let slopes = hagan_west_like_slopes(&x, &y);
        Ok(Self {
            x,
            y,
            slopes,
            extrapolation,
        })
    }
}

impl Interpolator for MonotoneConvexInterpolator {
    fn value(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(self.y[0]),
                ExtrapolationMode::Linear => Ok(self.y[0] + self.slopes[0] * (xq - self.x[0])),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(self.y[n - 1]),
                    ExtrapolationMode::Linear => {
                        Ok(self.y[n - 1] + self.slopes[n - 1] * (xq - self.x[n - 1]))
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval(
                self.x[i],
                self.x[i + 1],
                self.y[i],
                self.y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }

    fn derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(0.0),
                ExtrapolationMode::Linear => Ok(self.slopes[0]),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(0.0),
                    ExtrapolationMode::Linear => Ok(self.slopes[n - 1]),
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval_derivative(
                self.x[i],
                self.x[i + 1],
                self.y[i],
                self.y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            MonotoneConvexInterpolator::new(self.x.clone(), yb.to_vec(), self.extrapolation)?
                .value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

/// Cardinal-tension cubic spline.
#[derive(Debug, Clone)]
pub struct TensionSplineInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    slopes: Vec<f64>,
    tension: f64,
    extrapolation: ExtrapolationMode,
}

impl TensionSplineInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        tension: f64,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        if !(0.0..=1.0).contains(&tension) {
            return Err(InterpolationError::InvalidInput(
                "tension must be in [0, 1]",
            ));
        }

        let n = x.len();
        let mut slopes = vec![0.0; n];
        if n == 2 {
            let m = (1.0 - tension) * (y[1] - y[0]) / (x[1] - x[0]);
            slopes[0] = m;
            slopes[1] = m;
        } else {
            let sec0 = (y[1] - y[0]) / (x[1] - x[0]);
            slopes[0] = (1.0 - tension) * sec0;
            for i in 1..(n - 1) {
                let h_prev = x[i] - x[i - 1];
                let h_next = x[i + 1] - x[i];
                let sec_prev = (y[i] - y[i - 1]) / h_prev;
                let sec_next = (y[i + 1] - y[i]) / h_next;
                let centered = (h_next * sec_prev + h_prev * sec_next) / (h_prev + h_next);
                slopes[i] = (1.0 - tension) * centered;
            }
            let secn = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
            slopes[n - 1] = (1.0 - tension) * secn;
        }

        Ok(Self {
            x,
            y,
            slopes,
            tension,
            extrapolation,
        })
    }
}

impl Interpolator for TensionSplineInterpolator {
    fn value(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(self.y[0]),
                ExtrapolationMode::Linear => Ok(self.y[0] + self.slopes[0] * (xq - self.x[0])),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(self.y[n - 1]),
                    ExtrapolationMode::Linear => {
                        Ok(self.y[n - 1] + self.slopes[n - 1] * (xq - self.x[n - 1]))
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval(
                self.x[i],
                self.x[i + 1],
                self.y[i],
                self.y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }

    fn derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(0.0),
                ExtrapolationMode::Linear => Ok(self.slopes[0]),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(0.0),
                    ExtrapolationMode::Linear => Ok(self.slopes[n - 1]),
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval_derivative(
                self.x[i],
                self.x[i + 1],
                self.y[i],
                self.y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            TensionSplineInterpolator::new(
                self.x.clone(),
                yb.to_vec(),
                self.tension,
                self.extrapolation,
            )?
            .value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

/// Shape-preserving cubic Hermite interpolation (PCHIP style).
#[derive(Debug, Clone)]
pub struct HermiteMonotoneInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    slopes: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl HermiteMonotoneInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        let slopes = pchip_slopes(&x, &y);
        Ok(Self {
            x,
            y,
            slopes,
            extrapolation,
        })
    }
}

impl Interpolator for HermiteMonotoneInterpolator {
    fn value(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(self.y[0]),
                ExtrapolationMode::Linear => Ok(self.y[0] + self.slopes[0] * (xq - self.x[0])),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(self.y[n - 1]),
                    ExtrapolationMode::Linear => {
                        Ok(self.y[n - 1] + self.slopes[n - 1] * (xq - self.x[n - 1]))
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval(
                self.x[i],
                self.x[i + 1],
                self.y[i],
                self.y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }

    fn derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(0.0),
                ExtrapolationMode::Linear => Ok(self.slopes[0]),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(0.0),
                    ExtrapolationMode::Linear => Ok(self.slopes[n - 1]),
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval_derivative(
                self.x[i],
                self.x[i + 1],
                self.y[i],
                self.y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            HermiteMonotoneInterpolator::new(self.x.clone(), yb.to_vec(), self.extrapolation)?
                .value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

/// Log-cubic monotone interpolator.
#[derive(Debug, Clone)]
pub struct LogCubicMonotoneInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    log_y: Vec<f64>,
    slopes: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl LogCubicMonotoneInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        validate_positive_y(&y)?;
        let log_y: Vec<f64> = y.iter().map(|v| v.ln()).collect();
        let slopes = pchip_slopes(&x, &log_y);
        Ok(Self {
            x,
            y,
            log_y,
            slopes,
            extrapolation,
        })
    }

    fn log_value(&self, xq: f64) -> Result<f64, InterpolationError> {
        match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => Ok(self.log_y[0]),
                ExtrapolationMode::Linear => Ok(self.log_y[0] + self.slopes[0] * (xq - self.x[0])),
                ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => Ok(self.log_y[n - 1]),
                    ExtrapolationMode::Linear => {
                        Ok(self.log_y[n - 1] + self.slopes[n - 1] * (xq - self.x[n - 1]))
                    }
                    ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
                }
            }
            QueryLocation::Inside(i) => Ok(hermite_eval(
                self.x[i],
                self.x[i + 1],
                self.log_y[i],
                self.log_y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            )),
        }
    }
}

impl Interpolator for LogCubicMonotoneInterpolator {
    fn value(&self, xq: f64) -> Result<f64, InterpolationError> {
        Ok(self.log_value(xq)?.exp())
    }

    fn derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        let u = self.log_value(xq)?;
        let du = match query_location(&self.x, xq) {
            QueryLocation::Left => match self.extrapolation {
                ExtrapolationMode::Flat => 0.0,
                ExtrapolationMode::Linear => self.slopes[0],
                ExtrapolationMode::Error => return Err(InterpolationError::ExtrapolationDisabled),
            },
            QueryLocation::Right => {
                let n = self.x.len();
                match self.extrapolation {
                    ExtrapolationMode::Flat => 0.0,
                    ExtrapolationMode::Linear => self.slopes[n - 1],
                    ExtrapolationMode::Error => {
                        return Err(InterpolationError::ExtrapolationDisabled);
                    }
                }
            }
            QueryLocation::Inside(i) => hermite_eval_derivative(
                self.x[i],
                self.x[i + 1],
                self.log_y[i],
                self.log_y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
                xq,
            ),
        };

        Ok(u.exp() * du)
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            let mut yb_pos = yb.to_vec();
            for v in &mut yb_pos {
                *v = v.max(1.0e-12);
            }
            LogCubicMonotoneInterpolator::new(self.x.clone(), yb_pos, self.extrapolation)?.value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

fn ns_basis(t: f64, tau: f64) -> (f64, f64, f64, f64) {
    let x = t / tau;
    if x.abs() < 1.0e-8 {
        let f1 = 1.0 - 0.5 * x + x * x / 6.0;
        let f2 = 0.5 * x - x * x / 3.0;
        let df1_dx = -0.5 + x / 3.0;
        let df2_dx = 0.5 - 2.0 * x / 3.0;
        return (f1, f2, df1_dx / tau, df2_dx / tau);
    }

    let e = (-x).exp();
    let f1 = (1.0 - e) / x;
    let f2 = f1 - e;
    let df1_dx = (e * (x + 1.0) - 1.0) / (x * x);
    let df2_dx = df1_dx + e;
    (f1, f2, df1_dx / tau, df2_dx / tau)
}

fn fit_nelson_siegel(x: &[f64], y: &[f64]) -> Result<[f64; 4], InterpolationError> {
    validate_xy(x, y, 3)?;

    let taus = logspace(0.05, 40.0, 80);
    let mut best = None::<([f64; 4], f64)>;

    for tau in taus {
        let mut a = DMatrix::zeros(x.len(), 3);
        for (i, t) in x.iter().enumerate() {
            let (f1, f2, _, _) = ns_basis(*t, tau);
            a[(i, 0)] = 1.0;
            a[(i, 1)] = f1;
            a[(i, 2)] = f2;
        }

        let yv = DVector::from_column_slice(y);
        let Some(beta) = solve_least_squares(&a, &yv) else {
            continue;
        };

        let residual = (&a * &beta - yv.clone()).norm_squared() / x.len() as f64;
        let candidate = ([beta[0], beta[1], beta[2], tau], residual);
        if best.as_ref().is_none_or(|(_, err)| residual < *err) {
            best = Some(candidate);
        }
    }

    best.map(|(p, _)| p)
        .ok_or(InterpolationError::NonConvergence)
}

/// Nelson-Siegel parametric interpolator with least-squares calibration.
#[derive(Debug, Clone)]
pub struct NelsonSiegelInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    beta0: f64,
    beta1: f64,
    beta2: f64,
    tau: f64,
    extrapolation: ExtrapolationMode,
}

impl NelsonSiegelInterpolator {
    pub fn fit(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let [beta0, beta1, beta2, tau] = fit_nelson_siegel(&x, &y)?;
        Ok(Self {
            x,
            y,
            beta0,
            beta1,
            beta2,
            tau,
            extrapolation,
        })
    }

    pub fn from_params(
        x: Vec<f64>,
        y: Vec<f64>,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        tau: f64,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 1)?;
        if tau <= 0.0 {
            return Err(InterpolationError::InvalidInput("tau must be > 0"));
        }
        Ok(Self {
            x,
            y,
            beta0,
            beta1,
            beta2,
            tau,
            extrapolation,
        })
    }

    fn model_value(&self, t: f64) -> f64 {
        let (f1, f2, _, _) = ns_basis(t.max(1.0e-12), self.tau);
        self.beta0 + self.beta1 * f1 + self.beta2 * f2
    }

    fn model_derivative(&self, t: f64) -> f64 {
        let (_, _, df1, df2) = ns_basis(t.max(1.0e-12), self.tau);
        self.beta1 * df1 + self.beta2 * df2
    }

    fn apply_mode_value(&self, xq: f64) -> Result<f64, InterpolationError> {
        if self.x.is_empty() {
            return Ok(self.model_value(xq.max(1.0e-12)));
        }
        let x0 = self.x[0];
        let x1 = self.x[self.x.len() - 1];
        if (x0..=x1).contains(&xq) {
            return Ok(self.model_value(xq));
        }
        match self.extrapolation {
            ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            ExtrapolationMode::Flat => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_value(xb))
            }
            ExtrapolationMode::Linear => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_value(xb) + self.model_derivative(xb) * (xq - xb))
            }
        }
    }

    fn apply_mode_derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        if self.x.is_empty() {
            return Ok(self.model_derivative(xq.max(1.0e-12)));
        }
        let x0 = self.x[0];
        let x1 = self.x[self.x.len() - 1];
        if (x0..=x1).contains(&xq) {
            return Ok(self.model_derivative(xq));
        }
        match self.extrapolation {
            ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            ExtrapolationMode::Flat => Ok(0.0),
            ExtrapolationMode::Linear => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_derivative(xb))
            }
        }
    }
}

impl Interpolator for NelsonSiegelInterpolator {
    fn value(&self, x: f64) -> Result<f64, InterpolationError> {
        self.apply_mode_value(x)
    }

    fn derivative(&self, x: f64) -> Result<f64, InterpolationError> {
        self.apply_mode_derivative(x)
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            NelsonSiegelInterpolator::fit(self.x.clone(), yb.to_vec(), self.extrapolation)?
                .value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

fn nss_basis(t: f64, tau1: f64, tau2: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (f1, f2, df1, df2) = ns_basis(t, tau1);
    let (_, f3, _, df3) = ns_basis(t, tau2);
    (f1, f2, f3, df1, df2, df3)
}

fn fit_nss(x: &[f64], y: &[f64]) -> Result<[f64; 6], InterpolationError> {
    validate_xy(x, y, 4)?;

    let taus1 = logspace(0.05, 10.0, 28);
    let taus2 = logspace(0.2, 50.0, 32);
    let mut best = None::<([f64; 6], f64)>;

    for tau1 in &taus1 {
        for tau2 in &taus2 {
            if tau2 <= tau1 {
                continue;
            }

            let mut a = DMatrix::zeros(x.len(), 4);
            for (i, t) in x.iter().enumerate() {
                let (f1, f2, f3, _, _, _) = nss_basis(*t, *tau1, *tau2);
                a[(i, 0)] = 1.0;
                a[(i, 1)] = f1;
                a[(i, 2)] = f2;
                a[(i, 3)] = f3;
            }

            let yv = DVector::from_column_slice(y);
            let Some(beta) = solve_least_squares(&a, &yv) else {
                continue;
            };

            let residual = (&a * &beta - yv.clone()).norm_squared() / x.len() as f64;
            let candidate = ([beta[0], beta[1], beta[2], beta[3], *tau1, *tau2], residual);
            if best.as_ref().is_none_or(|(_, err)| residual < *err) {
                best = Some(candidate);
            }
        }
    }

    best.map(|(p, _)| p)
        .ok_or(InterpolationError::NonConvergence)
}

/// Nelson-Siegel-Svensson parametric interpolator.
#[derive(Debug, Clone)]
pub struct NelsonSiegelSvenssonInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    beta0: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    tau1: f64,
    tau2: f64,
    extrapolation: ExtrapolationMode,
}

impl NelsonSiegelSvenssonInterpolator {
    pub fn fit(
        x: Vec<f64>,
        y: Vec<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let [beta0, beta1, beta2, beta3, tau1, tau2] = fit_nss(&x, &y)?;
        Ok(Self {
            x,
            y,
            beta0,
            beta1,
            beta2,
            beta3,
            tau1,
            tau2,
            extrapolation,
        })
    }

    fn model_value(&self, t: f64) -> f64 {
        let (f1, f2, f3, _, _, _) = nss_basis(t.max(1.0e-12), self.tau1, self.tau2);
        self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3
    }

    fn model_derivative(&self, t: f64) -> f64 {
        let (_, _, _, df1, df2, df3) = nss_basis(t.max(1.0e-12), self.tau1, self.tau2);
        self.beta1 * df1 + self.beta2 * df2 + self.beta3 * df3
    }

    fn apply_mode_value(&self, xq: f64) -> Result<f64, InterpolationError> {
        let x0 = self.x[0];
        let x1 = self.x[self.x.len() - 1];
        if (x0..=x1).contains(&xq) {
            return Ok(self.model_value(xq));
        }
        match self.extrapolation {
            ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            ExtrapolationMode::Flat => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_value(xb))
            }
            ExtrapolationMode::Linear => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_value(xb) + self.model_derivative(xb) * (xq - xb))
            }
        }
    }

    fn apply_mode_derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        let x0 = self.x[0];
        let x1 = self.x[self.x.len() - 1];
        if (x0..=x1).contains(&xq) {
            return Ok(self.model_derivative(xq));
        }
        match self.extrapolation {
            ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            ExtrapolationMode::Flat => Ok(0.0),
            ExtrapolationMode::Linear => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_derivative(xb))
            }
        }
    }
}

impl Interpolator for NelsonSiegelSvenssonInterpolator {
    fn value(&self, x: f64) -> Result<f64, InterpolationError> {
        self.apply_mode_value(x)
    }

    fn derivative(&self, x: f64) -> Result<f64, InterpolationError> {
        self.apply_mode_derivative(x)
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            NelsonSiegelSvenssonInterpolator::fit(self.x.clone(), yb.to_vec(), self.extrapolation)?
                .value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

fn smith_wilson_kernel(t: f64, u: f64, ufr: f64, alpha: f64) -> f64 {
    let m = t.min(u);
    let big_m = t.max(u);
    let pref = (-ufr * (t + u)).exp();
    let g = alpha * m - (-alpha * big_m).exp() * (alpha * m).sinh();
    pref * g
}

fn smith_wilson_kernel_dt(t: f64, u: f64, ufr: f64, alpha: f64) -> f64 {
    let m = t.min(u);
    let big_m = t.max(u);
    let pref = (-ufr * (t + u)).exp();

    let g = alpha * m - (-alpha * big_m).exp() * (alpha * m).sinh();
    let dg_dt = if t <= u {
        alpha - alpha * (-alpha * u).exp() * (alpha * t).cosh()
    } else {
        alpha * (-alpha * t).exp() * (alpha * u).sinh()
    };

    pref * (dg_dt - ufr * g)
}

/// Smith-Wilson discount-factor interpolator (Solvency II style).
#[derive(Debug, Clone)]
pub struct SmithWilsonInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    ufr: f64,
    alpha: f64,
    zeta: DVector<f64>,
    extrapolation: ExtrapolationMode,
}

impl SmithWilsonInterpolator {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        ufr: f64,
        alpha: f64,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        validate_xy(&x, &y, 2)?;
        validate_positive_y(&y)?;
        if !ufr.is_finite() || !alpha.is_finite() || alpha <= 0.0 {
            return Err(InterpolationError::InvalidInput(
                "ufr must be finite and alpha > 0",
            ));
        }

        let n = x.len();
        let mut w = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                w[(i, j)] = smith_wilson_kernel(x[i], x[j], ufr, alpha);
            }
        }

        let m = DVector::from_column_slice(&y);
        let mu: Vec<f64> = x.iter().map(|t| (-ufr * *t).exp()).collect();
        let muv = DVector::from_column_slice(&mu);
        let rhs = m - muv;

        let Some(zeta) = w.lu().solve(&rhs) else {
            return Err(InterpolationError::SingularSystem);
        };

        Ok(Self {
            x,
            y,
            ufr,
            alpha,
            zeta,
            extrapolation,
        })
    }

    fn model_value(&self, t: f64) -> f64 {
        let mu = (-self.ufr * t).exp();
        let mut corr = 0.0;
        for (i, u) in self.x.iter().enumerate() {
            corr += self.zeta[i] * smith_wilson_kernel(t, *u, self.ufr, self.alpha);
        }
        (mu + corr).max(1.0e-14)
    }

    fn model_derivative(&self, t: f64) -> f64 {
        let mu_dt = -self.ufr * (-self.ufr * t).exp();
        let mut corr = 0.0;
        for (i, u) in self.x.iter().enumerate() {
            corr += self.zeta[i] * smith_wilson_kernel_dt(t, *u, self.ufr, self.alpha);
        }
        mu_dt + corr
    }

    fn apply_mode_value(&self, xq: f64) -> Result<f64, InterpolationError> {
        let x0 = self.x[0];
        let x1 = self.x[self.x.len() - 1];
        if (x0..=x1).contains(&xq) {
            return Ok(self.model_value(xq));
        }

        match self.extrapolation {
            ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            ExtrapolationMode::Flat => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_value(xb))
            }
            ExtrapolationMode::Linear => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok((self.model_value(xb) + self.model_derivative(xb) * (xq - xb)).max(1.0e-14))
            }
        }
    }

    fn apply_mode_derivative(&self, xq: f64) -> Result<f64, InterpolationError> {
        let x0 = self.x[0];
        let x1 = self.x[self.x.len() - 1];
        if (x0..=x1).contains(&xq) {
            return Ok(self.model_derivative(xq));
        }

        match self.extrapolation {
            ExtrapolationMode::Error => Err(InterpolationError::ExtrapolationDisabled),
            ExtrapolationMode::Flat => Ok(0.0),
            ExtrapolationMode::Linear => {
                let xb = if xq < x0 { x0 } else { x1 };
                Ok(self.model_derivative(xb))
            }
        }
    }
}

impl Interpolator for SmithWilsonInterpolator {
    fn value(&self, x: f64) -> Result<f64, InterpolationError> {
        self.apply_mode_value(x)
    }

    fn derivative(&self, x: f64) -> Result<f64, InterpolationError> {
        self.apply_mode_derivative(x)
    }

    fn jacobian(&self, xq: f64) -> Result<Vec<f64>, InterpolationError> {
        finite_difference_jacobian(&self.y, |yb| {
            let mut yb_pos = yb.to_vec();
            for v in &mut yb_pos {
                *v = v.max(1.0e-12);
            }
            SmithWilsonInterpolator::new(
                self.x.clone(),
                yb_pos,
                self.ufr,
                self.alpha,
                self.extrapolation,
            )?
            .value(xq)
        })
    }

    fn x(&self) -> &[f64] {
        &self.x
    }

    fn y(&self) -> &[f64] {
        &self.y
    }
}

/// Type-erased container for any supported interpolator.
#[derive(Debug, Clone)]
pub enum AnyInterpolator {
    Linear(LinearInterpolator),
    LogLinear(LogLinearInterpolator),
    MonotoneConvex(MonotoneConvexInterpolator),
    TensionSpline(TensionSplineInterpolator),
    HermiteMonotone(HermiteMonotoneInterpolator),
    LogCubicMonotone(LogCubicMonotoneInterpolator),
    NelsonSiegel(NelsonSiegelInterpolator),
    NelsonSiegelSvensson(NelsonSiegelSvenssonInterpolator),
    SmithWilson(SmithWilsonInterpolator),
}

impl Interpolator for AnyInterpolator {
    fn value(&self, x: f64) -> Result<f64, InterpolationError> {
        match self {
            AnyInterpolator::Linear(v) => v.value(x),
            AnyInterpolator::LogLinear(v) => v.value(x),
            AnyInterpolator::MonotoneConvex(v) => v.value(x),
            AnyInterpolator::TensionSpline(v) => v.value(x),
            AnyInterpolator::HermiteMonotone(v) => v.value(x),
            AnyInterpolator::LogCubicMonotone(v) => v.value(x),
            AnyInterpolator::NelsonSiegel(v) => v.value(x),
            AnyInterpolator::NelsonSiegelSvensson(v) => v.value(x),
            AnyInterpolator::SmithWilson(v) => v.value(x),
        }
    }

    fn derivative(&self, x: f64) -> Result<f64, InterpolationError> {
        match self {
            AnyInterpolator::Linear(v) => v.derivative(x),
            AnyInterpolator::LogLinear(v) => v.derivative(x),
            AnyInterpolator::MonotoneConvex(v) => v.derivative(x),
            AnyInterpolator::TensionSpline(v) => v.derivative(x),
            AnyInterpolator::HermiteMonotone(v) => v.derivative(x),
            AnyInterpolator::LogCubicMonotone(v) => v.derivative(x),
            AnyInterpolator::NelsonSiegel(v) => v.derivative(x),
            AnyInterpolator::NelsonSiegelSvensson(v) => v.derivative(x),
            AnyInterpolator::SmithWilson(v) => v.derivative(x),
        }
    }

    fn jacobian(&self, x: f64) -> Result<Vec<f64>, InterpolationError> {
        match self {
            AnyInterpolator::Linear(v) => v.jacobian(x),
            AnyInterpolator::LogLinear(v) => v.jacobian(x),
            AnyInterpolator::MonotoneConvex(v) => v.jacobian(x),
            AnyInterpolator::TensionSpline(v) => v.jacobian(x),
            AnyInterpolator::HermiteMonotone(v) => v.jacobian(x),
            AnyInterpolator::LogCubicMonotone(v) => v.jacobian(x),
            AnyInterpolator::NelsonSiegel(v) => v.jacobian(x),
            AnyInterpolator::NelsonSiegelSvensson(v) => v.jacobian(x),
            AnyInterpolator::SmithWilson(v) => v.jacobian(x),
        }
    }

    fn x(&self) -> &[f64] {
        match self {
            AnyInterpolator::Linear(v) => v.x(),
            AnyInterpolator::LogLinear(v) => v.x(),
            AnyInterpolator::MonotoneConvex(v) => v.x(),
            AnyInterpolator::TensionSpline(v) => v.x(),
            AnyInterpolator::HermiteMonotone(v) => v.x(),
            AnyInterpolator::LogCubicMonotone(v) => v.x(),
            AnyInterpolator::NelsonSiegel(v) => v.x(),
            AnyInterpolator::NelsonSiegelSvensson(v) => v.x(),
            AnyInterpolator::SmithWilson(v) => v.x(),
        }
    }

    fn y(&self) -> &[f64] {
        match self {
            AnyInterpolator::Linear(v) => v.y(),
            AnyInterpolator::LogLinear(v) => v.y(),
            AnyInterpolator::MonotoneConvex(v) => v.y(),
            AnyInterpolator::TensionSpline(v) => v.y(),
            AnyInterpolator::HermiteMonotone(v) => v.y(),
            AnyInterpolator::LogCubicMonotone(v) => v.y(),
            AnyInterpolator::NelsonSiegel(v) => v.y(),
            AnyInterpolator::NelsonSiegelSvensson(v) => v.y(),
            AnyInterpolator::SmithWilson(v) => v.y(),
        }
    }
}

fn logspace(low: f64, high: f64, n: usize) -> Vec<f64> {
    let l0 = low.ln();
    let l1 = high.ln();
    (0..n)
        .map(|i| {
            let u = if n <= 1 {
                0.0
            } else {
                i as f64 / (n as f64 - 1.0)
            };
            (l0 + u * (l1 - l0)).exp()
        })
        .collect()
}

fn solve_least_squares(a: &DMatrix<f64>, y: &DVector<f64>) -> Option<DVector<f64>> {
    let at = a.transpose();
    let ata = &at * a;
    let aty = at * y;
    ata.lu().solve(&aty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_jacobian_matches_weights() {
        let itp = LinearInterpolator::new(
            vec![1.0, 2.0, 4.0],
            vec![0.01, 0.02, 0.04],
            ExtrapolationMode::Linear,
        )
        .unwrap();

        let y = itp.value(1.5).unwrap();
        assert_relative_eq!(y, 0.015, epsilon = 1e-12);

        let j = itp.jacobian(1.5).unwrap();
        assert_relative_eq!(j[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(j[1], 0.5, epsilon = 1e-12);
        assert_relative_eq!(j[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn log_linear_positive_and_smooth() {
        let itp = LogLinearInterpolator::new(
            vec![1.0, 2.0, 4.0],
            vec![0.99, 0.95, 0.87],
            ExtrapolationMode::Linear,
        )
        .unwrap();

        assert!(itp.value(3.0).unwrap() > 0.0);
        assert!(itp.derivative(3.0).unwrap() < 0.0);
    }

    #[test]
    fn monotone_convex_stays_within_neighbor_bounds_on_monotone_data() {
        let x = vec![1.0, 2.0, 3.0, 5.0, 7.0];
        let y = vec![0.01, 0.012, 0.013, 0.016, 0.018];
        let itp = MonotoneConvexInterpolator::new(x.clone(), y.clone(), ExtrapolationMode::Linear)
            .unwrap();

        for w in x.windows(2) {
            let mid = 0.5 * (w[0] + w[1]);
            let v = itp.value(mid).unwrap();
            let i = x.partition_point(|z| *z < mid).saturating_sub(1);
            assert!(v >= y[i].min(y[i + 1]) - 1.0e-12);
            assert!(v <= y[i].max(y[i + 1]) + 1.0e-12);
        }
    }

    #[test]
    fn hermite_monotone_preserves_shape_for_monotone_series() {
        let itp = HermiteMonotoneInterpolator::new(
            vec![0.5, 1.0, 2.0, 5.0],
            vec![0.02, 0.021, 0.025, 0.029],
            ExtrapolationMode::Linear,
        )
        .unwrap();

        let mut prev = itp.value(0.5).unwrap();
        for i in 1..40 {
            let t = 0.5 + (4.5 * i as f64) / 40.0;
            let v = itp.value(t).unwrap();
            assert!(v >= prev - 1.0e-12);
            prev = v;
        }
    }

    #[test]
    fn tension_spline_interpolates_nodes() {
        let x = vec![0.5, 1.0, 2.0, 3.0];
        let y = vec![0.015, 0.017, 0.019, 0.021];
        let itp =
            TensionSplineInterpolator::new(x.clone(), y.clone(), 0.3, ExtrapolationMode::Flat)
                .unwrap();

        for (xi, yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(itp.value(*xi).unwrap(), *yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn log_cubic_stays_positive() {
        let itp = LogCubicMonotoneInterpolator::new(
            vec![1.0, 2.0, 5.0, 10.0],
            vec![0.99, 0.96, 0.87, 0.71],
            ExtrapolationMode::Linear,
        )
        .unwrap();

        for t in [0.5, 1.5, 3.0, 7.0, 12.0] {
            assert!(itp.value(t).unwrap() > 0.0);
        }
    }

    #[test]
    fn nelson_siegel_fit_retrieves_curve_within_one_bp() {
        let x = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
        let beta0 = 0.022;
        let beta1 = -0.014;
        let beta2 = 0.018;
        let tau = 1.8;

        let y: Vec<f64> = x
            .iter()
            .map(|t| {
                let (f1, f2, _, _) = ns_basis(*t, tau);
                beta0 + beta1 * f1 + beta2 * f2
            })
            .collect();

        let fit =
            NelsonSiegelInterpolator::fit(x.clone(), y.clone(), ExtrapolationMode::Linear).unwrap();
        for (ti, yi) in x.iter().zip(y.iter()) {
            let err_bp = (fit.value(*ti).unwrap() - yi) * 1.0e4;
            assert!(err_bp.abs() < 1.0);
        }
    }

    #[test]
    fn nss_fit_retrieves_curve_within_two_bp() {
        let x = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
        let beta0 = 0.021;
        let beta1 = -0.011;
        let beta2 = 0.015;
        let beta3 = 0.006;
        let tau1 = 1.6;
        let tau2 = 8.0;

        let y: Vec<f64> = x
            .iter()
            .map(|t| {
                let (f1, f2, f3, _, _, _) = nss_basis(*t, tau1, tau2);
                beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3
            })
            .collect();

        let fit =
            NelsonSiegelSvenssonInterpolator::fit(x.clone(), y.clone(), ExtrapolationMode::Linear)
                .unwrap();

        for (ti, yi) in x.iter().zip(y.iter()) {
            let err_bp = (fit.value(*ti).unwrap() - yi) * 1.0e4;
            assert!(err_bp.abs() < 2.0);
        }
    }

    #[test]
    fn smith_wilson_matches_pillars() {
        let x = vec![1.0, 2.0, 5.0, 10.0, 20.0];
        let y = vec![0.99, 0.965, 0.90, 0.80, 0.64];
        let sw = SmithWilsonInterpolator::new(
            x.clone(),
            y.clone(),
            0.032,
            0.12,
            ExtrapolationMode::Linear,
        )
        .unwrap();

        for (t, df) in x.iter().zip(y.iter()) {
            assert_relative_eq!(sw.value(*t).unwrap(), *df, epsilon = 1e-10);
        }
    }

    #[test]
    fn extrapolation_error_mode_rejects_outside() {
        let itp =
            LinearInterpolator::new(vec![1.0, 2.0], vec![0.01, 0.02], ExtrapolationMode::Error)
                .unwrap();

        assert!(matches!(
            itp.value(0.5),
            Err(InterpolationError::ExtrapolationDisabled)
        ));
        assert!(matches!(
            itp.value(3.0),
            Err(InterpolationError::ExtrapolationDisabled)
        ));
    }
}
