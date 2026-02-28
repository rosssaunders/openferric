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

pub mod ast;
pub mod compiler;
pub mod engine;
pub mod error;
pub mod eval;
pub mod ir;
pub mod lexer;
pub mod market;
pub mod parser;

// Re-exports for ergonomic usage.
pub use compiler::compile;
pub use engine::{DslMonteCarloEngine, DslProduct};
pub use error::DslError;
pub use ir::CompiledProduct;
pub use market::{AssetData, MultiAssetMarket};

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
    use crate::dsl::market::AssetData;

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
                AssetData { spot: 100.0, vol: 0.20, dividend_yield: 0.02 },
                AssetData { spot: 100.0, vol: 0.22, dividend_yield: 0.03 },
                AssetData { spot: 100.0, vol: 0.25, dividend_yield: 0.01 },
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
