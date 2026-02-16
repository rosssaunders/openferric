//! Reusable pre-allocated buffers for pricing hot paths.

/// Shared scratch buffers for repeated pricing calls.
///
/// Buffers grow on demand and never shrink, amortizing allocations across runs.
#[derive(Debug, Clone, Default)]
pub struct PricingArena {
    pub path_buffer: Vec<f64>,
    pub payoff_buffer: Vec<f64>,
    pub tree_buffer: Vec<f64>,
}

impl PricingArena {
    /// Creates an arena sized for expected Monte Carlo paths and tree steps.
    pub fn with_capacity(max_paths: usize, max_steps: usize) -> Self {
        Self {
            path_buffer: Vec::with_capacity(max_steps.saturating_add(1)),
            payoff_buffer: Vec::with_capacity(max_paths),
            tree_buffer: Vec::with_capacity(max_steps.saturating_add(1)),
        }
    }

    #[inline]
    fn ensure_len(buffer: &mut Vec<f64>, n: usize) {
        if buffer.len() < n {
            buffer.resize(n, 0.0);
        }
    }

    /// Returns a mutable path slice with length `n`.
    #[inline]
    pub fn path_slice(&mut self, n: usize) -> &mut [f64] {
        Self::ensure_len(&mut self.path_buffer, n);
        &mut self.path_buffer[..n]
    }

    /// Returns a mutable payoff slice with length `n`.
    #[inline]
    pub fn payoff_slice(&mut self, n: usize) -> &mut [f64] {
        Self::ensure_len(&mut self.payoff_buffer, n);
        &mut self.payoff_buffer[..n]
    }

    /// Returns a mutable tree slice with length `n`.
    #[inline]
    pub fn tree_slice(&mut self, n: usize) -> &mut [f64] {
        Self::ensure_len(&mut self.tree_buffer, n);
        &mut self.tree_buffer[..n]
    }
}

#[cfg(test)]
mod tests {
    use super::PricingArena;

    #[test]
    fn arena_grows_but_does_not_shrink() {
        let mut arena = PricingArena::with_capacity(4, 4);

        assert_eq!(arena.path_slice(3).len(), 3);
        assert_eq!(arena.payoff_slice(2).len(), 2);
        assert_eq!(arena.tree_slice(5).len(), 5);

        let path_len = arena.path_buffer.len();
        let payoff_len = arena.payoff_buffer.len();
        let tree_len = arena.tree_buffer.len();

        let _ = arena.path_slice(1);
        let _ = arena.payoff_slice(1);
        let _ = arena.tree_slice(1);

        assert_eq!(arena.path_buffer.len(), path_len);
        assert_eq!(arena.payoff_buffer.len(), payoff_len);
        assert_eq!(arena.tree_buffer.len(), tree_len);
    }
}
