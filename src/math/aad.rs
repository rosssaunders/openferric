//! Module `math::aad`.
//!
//! Adjoint algorithmic differentiation (AAD) primitives:
//! - forward-mode dual numbers (`Dual`, `Dual2`)
//! - reverse-mode tape with arena-style contiguous storage (`AadTape`)
//! - checkpoint/rewind for tape memory reuse in long Monte Carlo loops

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::core::{Greeks, OptionType};
use crate::math::{normal_cdf, normal_pdf};

/// Forward-mode first-order dual number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub value: f64,
    pub derivative: f64,
}

impl Dual {
    #[inline]
    pub fn variable(value: f64) -> Self {
        Self {
            value,
            derivative: 1.0,
        }
    }

    #[inline]
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: 0.0,
        }
    }

    #[inline]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            derivative: value * self.derivative,
        }
    }

    #[inline]
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        Self {
            value,
            derivative: 0.5 * self.derivative / value,
        }
    }

    #[inline]
    pub fn normal_cdf(self) -> Self {
        let value = normal_cdf(self.value);
        Self {
            value,
            derivative: normal_pdf(self.value) * self.derivative,
        }
    }

    #[inline]
    pub fn positive_part(self) -> Self {
        if self.value > 0.0 {
            self
        } else {
            Self::constant(0.0)
        }
    }
}

impl Add for Dual {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl Sub for Dual {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + rhs.derivative * self.value,
        }
    }
}

impl Div for Dual {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let inv = 1.0 / rhs.value;
        let value = self.value * inv;
        Self {
            value,
            derivative: (self.derivative - value * rhs.derivative) * inv,
        }
    }
}

impl Neg for Dual {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value + rhs,
            derivative: self.derivative,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value - rhs,
            derivative: self.derivative,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value * rhs,
            derivative: self.derivative * rhs,
        }
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        let inv = 1.0 / rhs;
        Self {
            value: self.value * inv,
            derivative: self.derivative * inv,
        }
    }
}

/// Forward-mode second-order dual number for single-variable Hessian slices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual2 {
    pub value: f64,
    pub first: f64,
    pub second: f64,
}

impl Dual2 {
    #[inline]
    pub fn variable(value: f64) -> Self {
        Self {
            value,
            first: 1.0,
            second: 0.0,
        }
    }

    #[inline]
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            first: 0.0,
            second: 0.0,
        }
    }

    #[inline]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            first: value * self.first,
            second: value * (self.first * self.first + self.second),
        }
    }

    #[inline]
    pub fn ln(self) -> Self {
        let inv = 1.0 / self.value;
        let inv_sq = inv * inv;
        Self {
            value: self.value.ln(),
            first: self.first * inv,
            second: self.second * inv - self.first * self.first * inv_sq,
        }
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let inv_two_sqrt = 0.5 / value;
        Self {
            value,
            first: self.first * inv_two_sqrt,
            second: self.second * inv_two_sqrt
                - self.first * self.first / (4.0 * self.value * value),
        }
    }

    #[inline]
    pub fn normal_cdf(self) -> Self {
        let pdf = normal_pdf(self.value);
        Self {
            value: normal_cdf(self.value),
            first: pdf * self.first,
            second: pdf * (self.second - self.value * self.first * self.first),
        }
    }

    #[inline]
    pub fn positive_part(self) -> Self {
        if self.value > 0.0 {
            self
        } else {
            Self::constant(0.0)
        }
    }
}

impl Add for Dual2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            first: self.first + rhs.first,
            second: self.second + rhs.second,
        }
    }
}

impl Sub for Dual2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            first: self.first - rhs.first,
            second: self.second - rhs.second,
        }
    }
}

impl Mul for Dual2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            first: self.first * rhs.value + rhs.first * self.value,
            second: self.second * rhs.value
                + 2.0 * self.first * rhs.first
                + self.value * rhs.second,
        }
    }
}

impl Div for Dual2 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let inv = 1.0 / rhs.value;
        let inv_sq = inv * inv;
        let inv_cu = inv_sq * inv;
        Self {
            value: self.value * inv,
            first: self.first * inv - self.value * rhs.first * inv_sq,
            second: self.second * inv - 2.0 * self.first * rhs.first * inv_sq
                + self.value * (2.0 * rhs.first * rhs.first - rhs.value * rhs.second) * inv_cu,
        }
    }
}

impl Neg for Dual2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            first: -self.first,
            second: -self.second,
        }
    }
}

impl Add<f64> for Dual2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value + rhs,
            first: self.first,
            second: self.second,
        }
    }
}

impl Sub<f64> for Dual2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value - rhs,
            first: self.first,
            second: self.second,
        }
    }
}

impl Mul<f64> for Dual2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value * rhs,
            first: self.first * rhs,
            second: self.second * rhs,
        }
    }
}

impl Div<f64> for Dual2 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        let inv = 1.0 / rhs;
        Self {
            value: self.value * inv,
            first: self.first * inv,
            second: self.second * inv,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NodeOp {
    Leaf,
    Unary {
        parent: usize,
        weight: f64,
    },
    Binary {
        left: usize,
        right: usize,
        left_weight: f64,
        right_weight: f64,
    },
}

/// Variable handle on a reverse-mode tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var {
    idx: usize,
}

/// Checkpoint marker for tape rewind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeCheckpoint {
    len: usize,
}

/// Reverse-mode tape with arena-style contiguous vectors.
#[derive(Debug, Clone, Default)]
pub struct AadTape {
    values: Vec<f64>,
    ops: Vec<NodeOp>,
    adjoints: Vec<f64>,
}

impl AadTape {
    #[inline]
    pub fn with_capacity(nodes: usize) -> Self {
        Self {
            values: Vec::with_capacity(nodes),
            ops: Vec::with_capacity(nodes),
            adjoints: Vec::with_capacity(nodes),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.values.clear();
        self.ops.clear();
        self.adjoints.clear();
    }

    #[inline]
    pub fn checkpoint(&self) -> TapeCheckpoint {
        TapeCheckpoint {
            len: self.values.len(),
        }
    }

    #[inline]
    pub fn rewind(&mut self, checkpoint: TapeCheckpoint) {
        self.values.truncate(checkpoint.len);
        self.ops.truncate(checkpoint.len);
        self.adjoints.truncate(checkpoint.len);
    }

    #[inline]
    fn push_node(&mut self, value: f64, op: NodeOp) -> Var {
        self.values.push(value);
        self.ops.push(op);
        self.adjoints.push(0.0);
        Var {
            idx: self.values.len() - 1,
        }
    }

    #[inline]
    pub fn variable(&mut self, value: f64) -> Var {
        self.push_node(value, NodeOp::Leaf)
    }

    #[inline]
    pub fn constant(&mut self, value: f64) -> Var {
        self.push_node(value, NodeOp::Leaf)
    }

    #[inline]
    pub fn value(&self, var: Var) -> f64 {
        self.values[var.idx]
    }

    #[inline]
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.push_node(
            self.values[a.idx] + self.values[b.idx],
            NodeOp::Binary {
                left: a.idx,
                right: b.idx,
                left_weight: 1.0,
                right_weight: 1.0,
            },
        )
    }

    #[inline]
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.push_node(
            self.values[a.idx] - self.values[b.idx],
            NodeOp::Binary {
                left: a.idx,
                right: b.idx,
                left_weight: 1.0,
                right_weight: -1.0,
            },
        )
    }

    #[inline]
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let av = self.values[a.idx];
        let bv = self.values[b.idx];
        self.push_node(
            av * bv,
            NodeOp::Binary {
                left: a.idx,
                right: b.idx,
                left_weight: bv,
                right_weight: av,
            },
        )
    }

    #[inline]
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let av = self.values[a.idx];
        let bv = self.values[b.idx];
        let inv = 1.0 / bv;
        self.push_node(
            av * inv,
            NodeOp::Binary {
                left: a.idx,
                right: b.idx,
                left_weight: inv,
                right_weight: -av * inv * inv,
            },
        )
    }

    #[inline]
    pub fn neg(&mut self, a: Var) -> Var {
        self.push_node(
            -self.values[a.idx],
            NodeOp::Unary {
                parent: a.idx,
                weight: -1.0,
            },
        )
    }

    #[inline]
    pub fn exp(&mut self, a: Var) -> Var {
        let value = self.values[a.idx].exp();
        self.push_node(
            value,
            NodeOp::Unary {
                parent: a.idx,
                weight: value,
            },
        )
    }

    #[inline]
    pub fn ln(&mut self, a: Var) -> Var {
        let av = self.values[a.idx];
        self.push_node(
            av.ln(),
            NodeOp::Unary {
                parent: a.idx,
                weight: 1.0 / av,
            },
        )
    }

    #[inline]
    pub fn sqrt(&mut self, a: Var) -> Var {
        let av = self.values[a.idx];
        let value = av.sqrt();
        self.push_node(
            value,
            NodeOp::Unary {
                parent: a.idx,
                weight: 0.5 / value,
            },
        )
    }

    #[inline]
    pub fn normal_cdf(&mut self, a: Var) -> Var {
        let av = self.values[a.idx];
        self.push_node(
            normal_cdf(av),
            NodeOp::Unary {
                parent: a.idx,
                weight: normal_pdf(av),
            },
        )
    }

    #[inline]
    pub fn positive_part(&mut self, a: Var) -> Var {
        let av = self.values[a.idx];
        let value = av.max(0.0);
        self.push_node(
            value,
            NodeOp::Unary {
                parent: a.idx,
                weight: if av > 0.0 { 1.0 } else { 0.0 },
            },
        )
    }

    /// Reverse sweep, writing node adjoints in-place.
    #[inline]
    pub fn reverse(&mut self, output: Var) {
        if self.adjoints.is_empty() {
            return;
        }
        self.adjoints.fill(0.0);
        self.adjoints[output.idx] = 1.0;

        for i in (0..=output.idx).rev() {
            let bar = self.adjoints[i];
            if bar == 0.0 {
                continue;
            }
            match self.ops[i] {
                NodeOp::Leaf => {}
                NodeOp::Unary { parent, weight } => {
                    self.adjoints[parent] += bar * weight;
                }
                NodeOp::Binary {
                    left,
                    right,
                    left_weight,
                    right_weight,
                } => {
                    self.adjoints[left] += bar * left_weight;
                    self.adjoints[right] += bar * right_weight;
                }
            }
        }
    }

    #[inline]
    pub fn adjoint(&self, var: Var) -> f64 {
        self.adjoints[var.idx]
    }

    #[inline]
    pub fn gradient(&mut self, output: Var, inputs: &[Var], out: &mut [f64]) {
        assert_eq!(inputs.len(), out.len(), "inputs/out length mismatch");
        self.reverse(output);
        for (dst, src) in out.iter_mut().zip(inputs.iter()) {
            *dst = self.adjoints[src.idx];
        }
    }

    #[inline]
    pub fn gradient_vec(&mut self, output: Var, inputs: &[Var]) -> Vec<f64> {
        let mut grads = vec![0.0; inputs.len()];
        self.gradient(output, inputs, &mut grads);
        grads
    }
}

/// Black-Scholes price + Greeks from mixed AAD:
/// reverse-mode for first-order sensitivities and forward second-order for gamma.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn black_scholes_price_greeks_aad(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, Greeks) {
    #[inline]
    fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
        match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        }
    }

    if expiry <= 0.0 {
        return (
            intrinsic(option_type, spot, strike),
            Greeks {
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
            },
        );
    }
    if vol <= 0.0 || !vol.is_finite() || spot <= 0.0 || strike <= 0.0 {
        return (
            intrinsic(option_type, spot, strike),
            Greeks {
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
            },
        );
    }

    let mut tape = AadTape::with_capacity(48);
    let s = tape.variable(spot);
    let r = tape.variable(rate);
    let q = tape.variable(dividend_yield);
    let sigma = tape.variable(vol);
    let t = tape.variable(expiry);
    let k = tape.constant(strike);

    let sqrt_t = tape.sqrt(t);
    let sig_sqrt_t = tape.mul(sigma, sqrt_t);
    let s_over_k = tape.div(s, k);
    let ln_sk = tape.ln(s_over_k);

    let sigma_sq = tape.mul(sigma, sigma);
    let half_c = tape.constant(0.5);
    let half_sigma_sq = tape.mul(half_c, sigma_sq);
    let r_minus_q = tape.sub(r, q);
    let drift_term = tape.add(r_minus_q, half_sigma_sq);
    let drift = tape.mul(drift_term, t);
    let num_d1 = tape.add(ln_sk, drift);
    let d1 = tape.div(num_d1, sig_sqrt_t);
    let d2 = tape.sub(d1, sig_sqrt_t);

    let rt = tape.mul(r, t);
    let qt = tape.mul(q, t);
    let minus_rt = tape.neg(rt);
    let minus_qt = tape.neg(qt);
    let df_r = tape.exp(minus_rt);
    let df_q = tape.exp(minus_qt);
    let nd1 = tape.normal_cdf(d1);
    let nd2 = tape.normal_cdf(d2);

    let s_df_q = tape.mul(s, df_q);
    let k_df_r = tape.mul(k, df_r);
    let lhs = tape.mul(s_df_q, nd1);
    let rhs = tape.mul(k_df_r, nd2);
    let call = tape.sub(lhs, rhs);
    let call_minus_sdfq = tape.sub(call, s_df_q);
    let put = tape.add(call_minus_sdfq, k_df_r);
    let price_var = match option_type {
        OptionType::Call => call,
        OptionType::Put => put,
    };

    let price = tape.value(price_var);
    let grads = tape.gradient_vec(price_var, &[s, r, sigma, t]);

    let s2 = Dual2::variable(spot);
    let sig_sqrt_t_2 = vol * expiry.sqrt();
    let d1_2 = ((s2 / Dual2::constant(strike)).ln()
        + (rate - dividend_yield + 0.5 * vol * vol) * expiry)
        / sig_sqrt_t_2;
    let d2_2 = d1_2 - Dual2::constant(sig_sqrt_t_2);
    let df_r_2 = (-rate * expiry).exp();
    let df_q_2 = (-dividend_yield * expiry).exp();
    let call_2 =
        (s2 * df_q_2) * d1_2.normal_cdf() - Dual2::constant(strike * df_r_2) * d2_2.normal_cdf();
    let price_2 = match option_type {
        OptionType::Call => call_2,
        OptionType::Put => call_2 - s2 * df_q_2 + Dual2::constant(strike * df_r_2),
    };

    (
        price,
        Greeks {
            delta: grads[0],
            gamma: price_2.second,
            vega: grads[2],
            theta: -grads[3],
            rho: grads[1],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::analytic::black_scholes::{
        bs_delta, bs_gamma, bs_price, bs_rho, bs_theta, bs_vega,
    };

    #[test]
    fn dual_matches_closed_form_derivative() {
        let x = Dual::variable(1.25);
        let y = ((x * x + Dual::constant(0.3)).exp() / (x + 1.0)).ln();
        let f = |v: f64| ((v * v + 0.3).exp() / (v + 1.0)).ln();
        let h = 1e-7;
        let expected = (f(x.value + h) - f(x.value - h)) / (2.0 * h);
        assert!((y.derivative - expected).abs() < 1e-8);
    }

    #[test]
    fn reverse_mode_gradient_matches_manual() {
        let mut tape = AadTape::with_capacity(16);
        let x = tape.variable(1.1);
        let y = tape.variable(0.8);
        let xy = tape.mul(x, y);
        let z = tape.exp(xy);
        let ln_y = tape.ln(y);
        let out = tape.add(z, ln_y);
        let g = tape.gradient_vec(out, &[x, y]);

        let expected_dx = (1.1_f64 * 0.8).exp() * 0.8;
        let expected_dy = (1.1_f64 * 0.8).exp() * 1.1 + 1.0 / 0.8;

        assert!((g[0] - expected_dx).abs() < 1e-12);
        assert!((g[1] - expected_dy).abs() < 1e-12);
    }

    #[test]
    fn checkpoint_rewind_reuses_tape() {
        let mut tape = AadTape::with_capacity(8);
        let cp0 = tape.checkpoint();
        let x = tape.variable(2.0);
        let y = tape.exp(x);
        assert!(tape.value(y) > 7.0);
        tape.rewind(cp0);
        assert_eq!(tape.len(), 0);

        let a = tape.variable(4.0);
        let b = tape.sqrt(a);
        assert!((tape.value(b) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn black_scholes_aad_matches_analytic_to_1e10() {
        let cases = [
            (OptionType::Call, 100.0, 100.0, 0.05, 0.01, 0.20, 1.0),
            (OptionType::Put, 110.0, 95.0, 0.02, 0.015, 0.35, 2.0),
            (OptionType::Call, 87.5, 120.0, -0.01, 0.0, 0.5, 0.7),
        ];

        for (option_type, s, k, r, q, vol, t) in cases {
            let (price, greeks) = black_scholes_price_greeks_aad(option_type, s, k, r, q, vol, t);
            let p_ref = bs_price(option_type, s, k, r, q, vol, t);
            let d_ref = bs_delta(option_type, s, k, r, q, vol, t);
            let g_ref = bs_gamma(s, k, r, q, vol, t);
            let v_ref = bs_vega(s, k, r, q, vol, t);
            let th_ref = bs_theta(option_type, s, k, r, q, vol, t);
            let rh_ref = bs_rho(option_type, s, k, r, q, vol, t);

            assert!((price - p_ref).abs() < 1e-10, "price mismatch");
            assert!((greeks.delta - d_ref).abs() < 1e-10, "delta mismatch");
            assert!((greeks.gamma - g_ref).abs() < 1e-10, "gamma mismatch");
            assert!((greeks.vega - v_ref).abs() < 1e-10, "vega mismatch");
            assert!((greeks.theta - th_ref).abs() < 1e-10, "theta mismatch");
            assert!((greeks.rho - rh_ref).abs() < 1e-10, "rho mismatch");
        }
    }
}
