use std::f64::consts::PI;

pub mod fast_norm;
pub mod gamma;

pub use fast_norm::{
    beasley_springer_moro_inv_cdf, fast_norm_cdf, fast_norm_inv_cdf, fast_norm_pdf, hart_norm_cdf,
};

#[derive(Debug, Clone, PartialEq)]
pub enum MathError {
    NonConvergence,
    ZeroDerivative,
    InvalidInput(&'static str),
}

#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    fast_norm_pdf(x)
}

#[inline(always)]
pub fn branch_free_normal_cdf(x: f64) -> f64 {
    hart_norm_cdf(x)
}

#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    branch_free_normal_cdf(x)
}

/// Inverse of the standard normal CDF.
#[inline]
pub fn normal_inv_cdf(p: f64) -> f64 {
    beasley_springer_moro_inv_cdf(p)
}

/// Bivariate standard normal CDF `P[X <= x, Y <= y]` with correlation `rho`.
///
/// Uses Plackett's identity with one-dimensional Gauss-Legendre integration:
/// `Phi2(x,y,rho) = Phi(x)Phi(y) + ∫_0^rho phi2(x,y,r) dr`.
pub fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if x.is_nan() || y.is_nan() || rho.is_nan() {
        return f64::NAN;
    }

    let rho = rho.clamp(-1.0, 1.0);

    if x <= -10.0 || y <= -10.0 {
        return 0.0;
    }
    if x >= 10.0 {
        return normal_cdf(y);
    }
    if y >= 10.0 {
        return normal_cdf(x);
    }

    if rho.abs() < 1.0e-12 {
        return normal_cdf(x) * normal_cdf(y);
    }
    if rho >= 1.0 - 1.0e-12 {
        return normal_cdf(x.min(y));
    }
    if rho <= -1.0 + 1.0e-12 {
        // For rho = -1: P(X <= x, Y <= y) = max(P(X <= x) - P(X < -y), 0).
        return (normal_cdf(x) - normal_cdf(-y)).max(0.0);
    }

    let base = normal_cdf(x) * normal_cdf(y);
    let integral = gauss_legendre_integrate(
        |r| {
            let one_minus_r2 = 1.0 - r * r;
            let exponent = -(x * x - 2.0 * r * x * y + y * y) / (2.0 * one_minus_r2);
            exponent.exp() / (2.0 * PI * one_minus_r2.sqrt())
        },
        0.0,
        rho,
        96,
    )
    .unwrap_or(0.0);

    (base + integral).clamp(0.0, 1.0)
}

#[inline]
pub fn newton_raphson<F, G>(
    f: F,
    df: G,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, MathError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(MathError::InvalidInput("tol must be positive"));
    }
    if max_iter == 0 {
        return Err(MathError::InvalidInput("max_iter must be > 0"));
    }

    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() <= tol {
            return Ok(x);
        }
        let dfx = df(x);
        if dfx.abs() <= 1e-14 {
            return Err(MathError::ZeroDerivative);
        }
        let x_next = x - fx / dfx;
        if (x_next - x).abs() <= tol {
            return Ok(x_next);
        }
        x = x_next;
    }

    Err(MathError::NonConvergence)
}

#[derive(Debug, Clone)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    y2: Vec<f64>,
}

impl CubicSpline {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, MathError> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(MathError::InvalidInput(
                "x and y must have same length >= 2",
            ));
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(MathError::InvalidInput("x must be strictly increasing"));
        }

        let n = x.len();
        let mut y2 = vec![0.0_f64; n];
        let mut u = vec![0.0_f64; n - 1];

        // Natural boundary conditions.
        y2[0] = 0.0;
        u[0] = 0.0;

        for i in 1..(n - 1) {
            let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            let ddydx =
                (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            u[i] = (6.0 * ddydx / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
        }

        y2[n - 1] = 0.0;
        for k in (0..(n - 1)).rev() {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }

        Ok(Self { x, y, y2 })
    }

    #[inline]
    pub fn interpolate(&self, xq: f64) -> f64 {
        let n = self.x.len();

        if xq <= self.x[0] {
            return self.y[0];
        }
        if xq >= self.x[n - 1] {
            return self.y[n - 1];
        }

        let mut klo = 0usize;
        let mut khi = n - 1;
        while khi - klo > 1 {
            let k = (khi + klo) >> 1;
            if self.x[k] > xq {
                khi = k;
            } else {
                klo = k;
            }
        }

        let h = self.x[khi] - self.x[klo];
        let a = (self.x[khi] - xq) / h;
        let b = (xq - self.x[klo]) / h;

        a * self.y[klo]
            + b * self.y[khi]
            + ((a * a * a - a) * self.y2[klo] + (b * b * b - b) * self.y2[khi]) * (h * h) / 6.0
    }
}

#[inline]
fn legendre_polynomial_and_derivative(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut p_nm2 = 1.0;
    let mut p_nm1 = x;
    for k in 2..=n {
        let kf = k as f64;
        let p_n = ((2.0 * kf - 1.0) * x * p_nm1 - (kf - 1.0) * p_nm2) / kf;
        p_nm2 = p_nm1;
        p_nm1 = p_n;
    }

    let p_n = p_nm1;
    let p_n_minus_1 = p_nm2;
    let dp_n = (n as f64) * (x * p_n - p_n_minus_1) / (x * x - 1.0);
    (p_n, dp_n)
}

#[inline]
pub fn gauss_legendre_nodes_weights(n: usize) -> Result<(Vec<f64>, Vec<f64>), MathError> {
    if n == 0 {
        return Err(MathError::InvalidInput("n must be > 0"));
    }

    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    let m = n.div_ceil(2);

    for i in 0..m {
        let i1 = i as f64 + 1.0;
        let nn = n as f64;
        let mut z = (PI * (i1 - 0.25) / (nn + 0.5)).cos();

        for _ in 0..80 {
            let (p, dp) = legendre_polynomial_and_derivative(n, z);
            let dz = -p / dp;
            z += dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }

        let (_, dp) = legendre_polynomial_and_derivative(n, z);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);

        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    Ok((nodes, weights))
}

#[inline]
pub fn gauss_legendre_integrate<F>(f: F, a: f64, b: f64, n: usize) -> Result<f64, MathError>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gauss_legendre_nodes_weights(n)?;
    let c1 = 0.5 * (b - a);
    let c2 = 0.5 * (b + a);

    let sum = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * f(c1 * x + c2))
        .sum::<f64>();

    Ok(c1 * sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn normal_pdf_and_cdf_sanity() {
        assert_relative_eq!(normal_pdf(0.0), 0.398_942_280_401_432_7, epsilon = 1e-12);
        assert_relative_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-9);
        assert_relative_eq!(normal_cdf(1.0), 0.841_344_746, epsilon = 2e-5);
        assert_relative_eq!(normal_cdf(-1.0), 1.0 - normal_cdf(1.0), epsilon = 1e-12);
    }

    #[test]
    fn normal_inv_cdf_inverts_cdf() {
        let ps = [1e-6, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0 - 1e-6];
        for &p in &ps {
            let x = normal_inv_cdf(p);
            let p_back = normal_cdf(x);
            assert_relative_eq!(p_back, p, epsilon = 2e-7);
        }
    }

    #[test]
    fn bivariate_cdf_matches_closed_form_at_origin() {
        // Phi2(0, 0; rho) = 1/4 + asin(rho)/(2*pi).
        let rho = 0.5_f64;
        let expected = 0.25 + rho.asin() / (2.0 * PI);
        assert_relative_eq!(
            bivariate_normal_cdf(0.0, 0.0, rho),
            expected,
            epsilon = 2e-5
        );

        let rho_neg = -0.5_f64;
        let expected_neg = 0.25 + rho_neg.asin() / (2.0 * PI);
        assert_relative_eq!(
            bivariate_normal_cdf(0.0, 0.0, rho_neg),
            expected_neg,
            epsilon = 2e-5
        );
    }

    #[test]
    fn bivariate_cdf_has_basic_symmetry() {
        let x = 0.35;
        let y = -0.70;
        let rho = 0.42;

        let xy = bivariate_normal_cdf(x, y, rho);
        let yx = bivariate_normal_cdf(y, x, rho);
        assert_relative_eq!(xy, yx, epsilon = 2e-6);
    }

    #[test]
    fn newton_raphson_finds_root() {
        let root = newton_raphson(|x| x * x - 2.0, |x| 2.0 * x, 1.0, 1e-12, 50).unwrap();
        assert_relative_eq!(root, 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn cubic_spline_interpolates_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline::new(x.clone(), y.clone()).unwrap();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(spline.interpolate(xi), yi, epsilon = 1e-12);
        }

        let mid = spline.interpolate(1.5);
        assert_relative_eq!(mid, 2.2, epsilon = 0.2);
    }

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        let int_x4 = gauss_legendre_integrate(|x| x.powi(4), 0.0, 1.0, 8).unwrap();
        assert_relative_eq!(int_x4, 0.2, epsilon = 1e-12);

        let int_x5_sym = gauss_legendre_integrate(|x| x.powi(5), -1.0, 1.0, 8).unwrap();
        assert_relative_eq!(int_x5_sym, 0.0, epsilon = 1e-12);
    }
}
