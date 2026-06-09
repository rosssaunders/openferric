//! Structured product DSL for OpenFerric.
//!
//! Provides a composable, text-based domain-specific language for defining and
//! pricing exotic structured products (autocallables, Phoenix notes, accumulators,
//! range accruals, etc.) without writing Rust code.
//!
//! # Architecture
//!
//! ```text
//! DSL text ──→ Lexer ──→ Parser ──→ AST ──→ Compiler ──→ IR (CompiledProduct) ──→ Evaluator
//!                                                                                     ↑
//!                                                          Multi-asset MC Engine ──────┘
//!                                                          (correlated GBM paths)
//! ```
//!
//! The language uses indentation-based blocks (no braces) and `then` for
//! conditionals, inspired by F#.
//!
//! # Numeric semantics
//!
//! - `==` and `!=` compare with an **absolute** `f64::EPSILON` tolerance:
//!   `a == b` is true iff `|a - b| < f64::EPSILON`. This is consistent across
//!   the scalar interpreter, the SIMD batch evaluators, and the JIT.
//! - `min`, `max`, `worst_of`, and `best_of` propagate NaN in all backends:
//!   a NaN operand (e.g. from division by zero) poisons the result rather
//!   than being silently dropped.
//! - Schedule observation dates beyond the product maturity are truncated at
//!   compile time (the LSP emits a warning); hand-built IR containing such
//!   dates is rejected when the execution plan is built.
//!
//! # Repeated evaluation
//!
//! For evaluating many paths against the same product, use
//! [`eval::ProductEvaluator`], which compiles the execution plan once and
//! reuses scratch buffers; [`eval::evaluate_product`] is a one-shot
//! convenience that rebuilds the plan on every call.
//!
//! # Quick Start
//!
//! ```rust
//! use openferric::dsl::{parse_and_compile, DslMonteCarloEngine, DslProduct, MultiAssetMarket};
//!
//! let source = "\
//! product \"Forward\"
//!     notional: 100
//!     maturity: 1.0
//!     underlyings
//!         SPX = asset(0)
//!     schedule annual from 1.0 to 1.0
//!         redeem notional
//! ";
//!
//! let product = parse_and_compile(source).unwrap();
//! let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
//! let engine = DslMonteCarloEngine::new(10_000, 100, 42);
//! let result = engine.price_multi_asset(&product, &market).unwrap();
//! assert!(result.price > 0.0);
//! ```

pub mod analysis;
pub mod ast;
pub mod compiler;
pub mod engine;
pub mod error;
pub mod eval;
pub mod ir;
#[cfg(feature = "jit")]
pub mod jit;
pub mod lexer;
pub mod market;
pub mod parser;

// Re-exports for ergonomic usage.
pub use compiler::compile;
pub use engine::{DslMonteCarloEngine, DslProduct};
pub use error::DslError;
pub use eval::ProductEvaluator;
pub use ir::CompiledProduct;
pub use market::{AssetMarketData, MultiAssetMarket};

/// Parse and compile a DSL source string into a `CompiledProduct`.
///
/// This is the main entry point for the DSL pipeline.
pub fn parse_and_compile(source: &str) -> Result<CompiledProduct, DslError> {
    let tokens = lexer::tokenize(source)?;
    let ast = parser::parse(tokens)?;
    compiler::compile(&ast)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::market::AssetMarketData;

    #[test]
    fn end_to_end_simple_forward() {
        let source = "\
product \"Forward\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        redeem notional
";

        let product = parse_and_compile(source).unwrap();
        assert_eq!(product.name, "Forward");
        assert_eq!(product.notional, 100.0);
        assert_eq!(product.maturity, 1.0);

        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
        let engine = DslMonteCarloEngine::new(10_000, 100, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // Redeem notional at T=1.0, discounted: 100 * exp(-0.05) ~ 95.12
        let expected = 100.0 * (-0.05f64).exp();
        assert!(
            (result.price - expected).abs() < 1.0,
            "expected ~{expected}, got {}",
            result.price
        );
    }

    #[test]
    fn end_to_end_autocallable() {
        let source = "\
product \"WoF Autocall 18m\"
    notional: 1_000_000
    maturity: 1.5

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)
        NKY  = asset(2)

    state
        ki_hit: bool = false

    schedule quarterly from 0.25 to 1.5
        let wof = worst_of(performances())

        if wof <= 0.60 then
            set ki_hit = true

        if wof >= 1.0 and not is_final then
            pay notional * 0.08 * observation_date
            redeem notional

        if is_final then
            pay notional * 0.08 * 1.5
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
";

        let product = parse_and_compile(source).unwrap();
        assert_eq!(product.name, "WoF Autocall 18m");
        assert_eq!(product.num_underlyings, 3);
        assert_eq!(product.state_vars.len(), 1);
        assert_eq!(product.schedules[0].dates.len(), 6);

        let market = MultiAssetMarket {
            assets: vec![
                AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.20,
                    dividend_yield: 0.02,
                },
                AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.22,
                    dividend_yield: 0.03,
                },
                AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.25,
                    dividend_yield: 0.01,
                },
            ],
            correlation: vec![
                vec![1.0, 0.7, 0.5],
                vec![0.7, 1.0, 0.6],
                vec![0.5, 0.6, 1.0],
            ],
            rate: 0.03,
        };

        let engine = DslMonteCarloEngine::new(50_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        assert!(
            result.price > 500_000.0 && result.price < 1_200_000.0,
            "autocallable price {} out of expected range",
            result.price
        );
        assert!(result.stderr.is_some());
    }

    #[test]
    fn end_to_end_phoenix_with_memory() {
        let source = "\
product \"Phoenix Memory\"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false
        missed_coupons: float = 0.0

    schedule quarterly from 0.25 to 1.0
        let wof = worst_of(performances())

        if wof <= 0.60 then
            set ki_hit = true

        if wof >= 0.70 then
            pay notional * (0.02 + missed_coupons)
            set missed_coupons = 0.0
        else
            set missed_coupons = missed_coupons + 0.02

        if wof >= 1.0 and not is_final then
            redeem notional

        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
";

        let product = parse_and_compile(source).unwrap();
        assert_eq!(product.name, "Phoenix Memory");
        assert_eq!(product.state_vars.len(), 2);

        let market = MultiAssetMarket::single(100.0, 0.20, 0.03, 0.02);
        let engine = DslMonteCarloEngine::new(50_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        assert!(
            result.price > 800_000.0 && result.price < 1_100_000.0,
            "phoenix price {} out of expected range",
            result.price
        );
    }

    #[test]
    fn end_to_end_underlying_name_and_maturity() {
        // `SPX` resolves to the asset's spot; `maturity` to the product
        // maturity constant — in every backend, since both compile to plain
        // bytecode (PRICE / PUSH_CONST).
        let source = "\
product \"UnderlyingRef\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        pay notional * 0.08 * maturity
        redeem SPX
";

        let product = parse_and_compile(source).unwrap();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
        let engine = DslMonteCarloEngine::new(50_000, 100, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // redeem S(T): discounted forward = S0 = 100 (q=0); coupon = 8 discounted.
        let expected = 100.0 + 8.0 * (-0.05f64).exp();
        let rel_err = ((result.price - expected) / expected).abs();
        assert!(
            rel_err < 0.02,
            "expected ~{expected}, got {} (rel_err {rel_err})",
            result.price
        );
    }

    #[test]
    fn parse_error_produces_clear_message() {
        // Missing product name string.
        let result = parse_and_compile("product\n    notional: 100\n");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("expected"), "error: {err}");
    }

    #[test]
    fn compile_error_produces_clear_message() {
        let result = parse_and_compile("product \"Test\"\n    maturity: 1.0\n");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("notional"), "error: {err}");
    }
}
