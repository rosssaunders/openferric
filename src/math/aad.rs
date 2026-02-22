//! Adjoint Algorithmic Differentiation (AAD) primitives.
//!
//! This module provides:
//! - forward-mode dual numbers for single-seed sensitivities,
//! - reverse-mode tape differentiation for gradients of scalar outputs,
//! - an arena-style tape allocator with checkpoint/rewind support.
//!
//! The implementation follows standard operator-overloading and tape
//! accumulation patterns used in quantitative finance risk engines.
//!
//! References:
//! - Savine (2018), *Modern Computational Finance*.
//! - Giles & Glasserman (2006), smoking adjoints for Monte Carlo.
//! - Capriotti (2011), fast Greeks by algorithmic differentiation.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::{normal_cdf, normal_pdf};

const DEFAULT_BLOCK_SIZE: usize = 4_096;

/// Forward-mode dual number with one derivative component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Function value.
    pub value: f64,
    /// Derivative with respect to the seeded variable.
    pub derivative: f64,
}

impl Dual {
    /// Creates a dual number from value and derivative.
    #[inline]
    pub const fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    /// Creates a constant dual number (`d/dx = 0`).
    #[inline]
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: 0.0,
        }
    }

    /// Creates an active variable (`d/dx = 1`).
    #[inline]
    pub const fn variable(value: f64) -> Self {
        Self {
            value,
            derivative: 1.0,
        }
    }

    /// Exponential.
    #[inline]
    pub fn exp(self) -> Self {
        let v = self.value.exp();
        Self {
            value: v,
            derivative: v * self.derivative,
        }
    }

    /// Natural logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }

    /// Square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        Self {
            value: v,
            derivative: 0.5 * self.derivative / v,
        }
    }

    /// Standard normal CDF.
    #[inline]
    pub fn normal_cdf(self) -> Self {
        Self {
            value: normal_cdf(self.value),
            derivative: normal_pdf(self.value) * self.derivative,
        }
    }

    /// Positive part `max(x, 0)` with pathwise derivative convention.
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

impl Mul for Dual {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
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

impl Div for Dual {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let inv = 1.0 / rhs.value;
        Self {
            value: self.value * inv,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) * inv * inv,
        }
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value / rhs,
            derivative: self.derivative / rhs,
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

/// Computes a value and first derivative in forward mode.
#[inline]
pub fn forward_sensitivity<F>(x: f64, f: F) -> (f64, f64)
where
    F: FnOnce(Dual) -> Dual,
{
    let out = f(Dual::variable(x));
    (out.value, out.derivative)
}

/// Node identifier on the reverse tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);

#[derive(Debug, Clone, Copy)]
enum Op {
    Input,
    Constant,
    Unary {
        arg: VarId,
        darg: f64,
    },
    Binary {
        lhs: VarId,
        rhs: VarId,
        dlhs: f64,
        drhs: f64,
    },
}

#[derive(Debug, Clone, Copy)]
struct Node {
    value: f64,
    adjoint: f64,
    op: Op,
}

impl Node {
    #[inline]
    fn new(value: f64, op: Op) -> Self {
        Self {
            value,
            adjoint: 0.0,
            op,
        }
    }
}

#[derive(Debug, Clone)]
struct TapeArena {
    block_size: usize,
    blocks: Vec<Vec<Node>>,
    len: usize,
}

impl TapeArena {
    fn with_capacity(capacity: usize, block_size: usize) -> Self {
        let num_blocks = if capacity == 0 {
            0
        } else {
            capacity.div_ceil(block_size)
        };
        let mut blocks = Vec::with_capacity(num_blocks.max(1));
        if num_blocks > 0 {
            for _ in 0..num_blocks {
                blocks.push(Vec::with_capacity(block_size));
            }
        }
        Self {
            block_size,
            blocks,
            len: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, node: Node) -> VarId {
        let block = self.len / self.block_size;
        if block == self.blocks.len() {
            self.blocks.push(Vec::with_capacity(self.block_size));
        }
        self.blocks[block].push(node);
        let id = VarId(self.len);
        self.len += 1;
        id
    }

    #[inline]
    fn get(&self, id: VarId) -> Node {
        let block = id.0 / self.block_size;
        let offset = id.0 % self.block_size;
        self.blocks[block][offset]
    }

    #[inline]
    fn get_mut(&mut self, id: VarId) -> &mut Node {
        let block = id.0 / self.block_size;
        let offset = id.0 % self.block_size;
        &mut self.blocks[block][offset]
    }

    fn for_each_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Node),
    {
        for block in &mut self.blocks {
            for node in block {
                f(node);
            }
        }
    }

    fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }

        let keep_blocks = if len == 0 {
            0
        } else {
            (len - 1) / self.block_size + 1
        };

        while self.blocks.len() > keep_blocks {
            self.blocks.pop();
        }

        if keep_blocks > 0 {
            let tail_len = len - (keep_blocks - 1) * self.block_size;
            self.blocks[keep_blocks - 1].truncate(tail_len);
        }

        self.len = len;

        if self.blocks.is_empty() {
            self.blocks.push(Vec::with_capacity(self.block_size));
        }
    }
}

/// Tape checkpoint used for memory-bounded computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeCheckpoint {
    node_len: usize,
    input_len: usize,
}

/// Reverse-mode tape with arena-backed node storage.
#[derive(Debug, Clone)]
pub struct Tape {
    arena: TapeArena,
    inputs: Vec<VarId>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Tape {
    /// Creates an empty tape.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a tape with reserved node capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: TapeArena::with_capacity(capacity, DEFAULT_BLOCK_SIZE),
            inputs: Vec::new(),
        }
    }

    /// Number of nodes currently held on tape.
    #[inline]
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    /// Returns true when no nodes are present.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates an active input variable and tracks it in the input set.
    pub fn input(&mut self, value: f64) -> VarId {
        let id = self.arena.push(Node::new(value, Op::Input));
        self.inputs.push(id);
        id
    }

    /// Creates a constant node.
    pub fn constant(&mut self, value: f64) -> VarId {
        self.arena.push(Node::new(value, Op::Constant))
    }

    /// Value lookup for a node.
    #[inline]
    pub fn value(&self, id: VarId) -> f64 {
        self.arena.get(id).value
    }

    /// Current list of input nodes.
    #[inline]
    pub fn inputs(&self) -> &[VarId] {
        &self.inputs
    }

    /// Creates a checkpoint that can later be rewound to.
    #[inline]
    pub fn checkpoint(&self) -> TapeCheckpoint {
        TapeCheckpoint {
            node_len: self.len(),
            input_len: self.inputs.len(),
        }
    }

    /// Rewinds the tape to a previous checkpoint.
    pub fn rewind(&mut self, checkpoint: TapeCheckpoint) {
        self.arena.truncate(checkpoint.node_len);
        self.inputs.truncate(checkpoint.input_len);
    }

    /// Clears all nodes and tracked inputs.
    pub fn clear(&mut self) {
        self.arena.truncate(0);
        self.inputs.clear();
    }

    #[inline]
    fn unary(&mut self, arg: VarId, value: f64, darg: f64) -> VarId {
        self.arena.push(Node::new(value, Op::Unary { arg, darg }))
    }

    #[inline]
    fn binary(&mut self, lhs: VarId, rhs: VarId, value: f64, dlhs: f64, drhs: f64) -> VarId {
        self.arena.push(Node::new(
            value,
            Op::Binary {
                lhs,
                rhs,
                dlhs,
                drhs,
            },
        ))
    }

    #[inline]
    pub fn add(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let lv = self.value(lhs);
        let rv = self.value(rhs);
        self.binary(lhs, rhs, lv + rv, 1.0, 1.0)
    }

    #[inline]
    pub fn add_const(&mut self, lhs: VarId, c: f64) -> VarId {
        self.unary(lhs, self.value(lhs) + c, 1.0)
    }

    #[inline]
    pub fn sub(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let lv = self.value(lhs);
        let rv = self.value(rhs);
        self.binary(lhs, rhs, lv - rv, 1.0, -1.0)
    }

    #[inline]
    pub fn sub_const(&mut self, lhs: VarId, c: f64) -> VarId {
        self.unary(lhs, self.value(lhs) - c, 1.0)
    }

    #[inline]
    pub fn const_sub(&mut self, c: f64, rhs: VarId) -> VarId {
        self.unary(rhs, c - self.value(rhs), -1.0)
    }

    #[inline]
    pub fn mul(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let lv = self.value(lhs);
        let rv = self.value(rhs);
        self.binary(lhs, rhs, lv * rv, rv, lv)
    }

    #[inline]
    pub fn mul_const(&mut self, lhs: VarId, c: f64) -> VarId {
        self.unary(lhs, self.value(lhs) * c, c)
    }

    #[inline]
    pub fn div(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let lv = self.value(lhs);
        let rv = self.value(rhs);
        let inv = 1.0 / rv;
        self.binary(lhs, rhs, lv * inv, inv, -lv * inv * inv)
    }

    #[inline]
    pub fn div_const(&mut self, lhs: VarId, c: f64) -> VarId {
        self.unary(lhs, self.value(lhs) / c, 1.0 / c)
    }

    #[inline]
    pub fn neg(&mut self, arg: VarId) -> VarId {
        self.unary(arg, -self.value(arg), -1.0)
    }

    #[inline]
    pub fn exp(&mut self, arg: VarId) -> VarId {
        let v = self.value(arg).exp();
        self.unary(arg, v, v)
    }

    #[inline]
    pub fn ln(&mut self, arg: VarId) -> VarId {
        let x = self.value(arg);
        self.unary(arg, x.ln(), 1.0 / x)
    }

    #[inline]
    pub fn sqrt(&mut self, arg: VarId) -> VarId {
        let x = self.value(arg);
        let v = x.sqrt();
        self.unary(arg, v, 0.5 / v)
    }

    #[inline]
    pub fn normal_cdf(&mut self, arg: VarId) -> VarId {
        let x = self.value(arg);
        self.unary(arg, normal_cdf(x), normal_pdf(x))
    }

    /// Positive part `max(x, 0)` with pathwise derivative convention.
    #[inline]
    pub fn positive_part(&mut self, arg: VarId) -> VarId {
        let x = self.value(arg);
        if x > 0.0 {
            self.unary(arg, x, 1.0)
        } else {
            self.unary(arg, 0.0, 0.0)
        }
    }

    /// Reverse accumulation of gradient for selected inputs.
    pub fn gradient(&mut self, target: VarId, wrt: &[VarId]) -> Vec<f64> {
        self.reset_adjoints();
        self.arena.get_mut(target).adjoint = 1.0;
        self.reverse_sweep();
        wrt.iter().map(|&id| self.arena.get(id).adjoint).collect()
    }

    /// Reverse accumulation for all tracked tape inputs.
    #[inline]
    pub fn gradient_wrt_inputs(&mut self, target: VarId) -> Vec<f64> {
        let wrt = self.inputs.clone();
        self.gradient(target, &wrt)
    }

    fn reset_adjoints(&mut self) {
        self.arena.for_each_mut(|node| {
            node.adjoint = 0.0;
        });
    }

    fn reverse_sweep(&mut self) {
        for idx in (0..self.len()).rev() {
            let id = VarId(idx);
            let node = self.arena.get(id);
            if node.adjoint == 0.0 {
                continue;
            }
            match node.op {
                Op::Input | Op::Constant => {}
                Op::Unary { arg, darg } => {
                    self.arena.get_mut(arg).adjoint += node.adjoint * darg;
                }
                Op::Binary {
                    lhs,
                    rhs,
                    dlhs,
                    drhs,
                } => {
                    self.arena.get_mut(lhs).adjoint += node.adjoint * dlhs;
                    self.arena.get_mut(rhs).adjoint += node.adjoint * drhs;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptionType;
    use crate::greeks::black_scholes_merton_greeks;
    use approx::assert_relative_eq;

    #[test]
    fn forward_dual_derivative_matches_closed_form() {
        let (value, deriv) = forward_sensitivity(2.3, |x| x * x + x.exp());
        assert_relative_eq!(value, 2.3 * 2.3 + 2.3_f64.exp(), epsilon = 1e-14);
        assert_relative_eq!(deriv, 2.0 * 2.3 + 2.3_f64.exp(), epsilon = 1e-14);
    }

    #[test]
    fn reverse_tape_gradient_matches_analytic() {
        let mut tape = Tape::with_capacity(16);
        let x = tape.input(1.25);
        let y = tape.input(0.75);

        // f(x,y) = x^2 * y + exp(y)
        let x2 = tape.mul(x, x);
        let t1 = tape.mul(x2, y);
        let t2 = tape.exp(y);
        let f = tape.add(t1, t2);

        let g = tape.gradient(f, &[x, y]);
        assert_relative_eq!(
            tape.value(f),
            1.25 * 1.25 * 0.75 + 0.75_f64.exp(),
            epsilon = 1e-14
        );
        assert_relative_eq!(g[0], 2.0 * 1.25 * 0.75, epsilon = 1e-13);
        assert_relative_eq!(g[1], 1.25 * 1.25 + 0.75_f64.exp(), epsilon = 1e-13);
    }

    #[test]
    fn checkpoint_rewind_bounds_tape_growth() {
        let mut tape = Tape::with_capacity(64);
        let x = tape.input(1.0);
        let checkpoint = tape.checkpoint();
        let base_len = tape.len();

        for _ in 0..100 {
            tape.rewind(checkpoint);
            let mut y = x;
            for _ in 0..8 {
                y = tape.add_const(y, 1.0);
                y = tape.exp(y);
            }
            let _ = tape.gradient(y, &[x]);
        }

        assert!(tape.len() >= base_len);
        assert!(tape.len() <= base_len + 20);
    }

    #[test]
    fn forward_mode_black_scholes_delta_matches_closed_form() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.03;
        let q = 0.01;
        let sigma = 0.2;
        let t = 2.0;

        let (_price, delta) = forward_sensitivity(s, |spot| {
            let strike = Dual::constant(k);
            let rate = Dual::constant(r);
            let div = Dual::constant(q);
            let vol = Dual::constant(sigma);
            let expiry = Dual::constant(t);

            let sqrt_t = expiry.sqrt();
            let sig_sqrt_t = vol * sqrt_t;
            let d1 = ((spot / strike).ln()
                + (rate - div + Dual::constant(0.5) * vol * vol) * expiry)
                / sig_sqrt_t;
            let d2 = d1 - sig_sqrt_t;
            let df_q = (-div * expiry).exp();
            let df_r = (-rate * expiry).exp();
            spot * df_q * d1.normal_cdf() - strike * df_r * d2.normal_cdf()
        });

        let analytic = black_scholes_merton_greeks(OptionType::Call, s, k, r, q, sigma, t);
        assert_relative_eq!(delta, analytic.delta, epsilon = 1e-10);
    }
}
