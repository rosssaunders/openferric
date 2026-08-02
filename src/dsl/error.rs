//! DSL error types with span-based diagnostics.

use serde::{Deserialize, Serialize};

use crate::core::PricingError;
use std::fmt;

/// Source span for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// Errors produced by the DSL pipeline (lex, parse, compile, evaluate).
#[derive(Debug, Clone, PartialEq)]
pub enum DslError {
    /// Lexer error: unexpected character or malformed token.
    LexError { message: String, span: Span },
    /// Parser error: unexpected token or missing construct.
    ParseError { message: String, span: Span },
    /// Compiler error: type mismatch, undefined variable, etc.
    CompileError { message: String, span: Option<Span> },
    /// Runtime evaluation error.
    EvalError(String),
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LexError { message, span } => {
                write!(f, "lex error at {}-{}: {message}", span.start, span.end)
            }
            Self::ParseError { message, span } => {
                write!(f, "parse error at {}-{}: {message}", span.start, span.end)
            }
            Self::CompileError {
                message,
                span: Some(span),
            } => write!(f, "compile error at {}-{}: {message}", span.start, span.end),
            Self::CompileError {
                message,
                span: None,
            } => write!(f, "compile error: {message}"),
            Self::EvalError(msg) => write!(f, "eval error: {msg}"),
        }
    }
}

impl std::error::Error for DslError {}

impl From<DslError> for PricingError {
    fn from(e: DslError) -> Self {
        PricingError::InvalidInput(e.to_string())
    }
}

/// Annotates a DSL source string with line/column information for a given span.
pub fn annotate_source(source: &str, span: Span) -> String {
    // Spans are byte offsets, but diagnostics are presented to people as
    // character columns. Clamp externally supplied/deserialized spans and
    // move an offset inside a multi-byte character back to its boundary so
    // formatting an error can never panic.
    let mut offset = span.start.min(source.len());
    while !source.is_char_boundary(offset) {
        offset -= 1;
    }
    let before = &source[..offset];
    let line = before.chars().filter(|&c| c == '\n').count() + 1;
    let col = before
        .rsplit('\n')
        .next()
        .map_or(1, |line| line.chars().count() + 1);
    format!("line {line}, col {col}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_constructor_and_serde_preserve_byte_range() {
        let span = Span::new(7, 19);

        assert_eq!(span.start, 7);
        assert_eq!(span.end, 19);
        assert_eq!(format!("{span:?}"), "Span { start: 7, end: 19 }");

        let json = serde_json::to_string(&span).expect("Span should serialize");
        assert_eq!(json, r#"{"start":7,"end":19}"#);
        assert_eq!(
            serde_json::from_str::<Span>(&json).expect("Span should deserialize"),
            span
        );
    }

    #[test]
    fn dsl_error_display_locks_every_category_and_span_form() {
        let cases = [
            (
                DslError::LexError {
                    message: "unexpected `@`".into(),
                    span: Span::new(4, 5),
                },
                "lex error at 4-5: unexpected `@`",
            ),
            (
                DslError::ParseError {
                    message: "expected maturity".into(),
                    span: Span::new(10, 18),
                },
                "parse error at 10-18: expected maturity",
            ),
            (
                DslError::CompileError {
                    message: "undefined variable `coupon`".into(),
                    span: Some(Span::new(31, 37)),
                },
                "compile error at 31-37: undefined variable `coupon`",
            ),
            (
                DslError::CompileError {
                    message: "missing notional".into(),
                    span: None,
                },
                "compile error: missing notional",
            ),
            (
                DslError::EvalError("asset path is empty".into()),
                "eval error: asset path is empty",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn dsl_error_implements_std_error_without_a_source() {
        fn as_std_error(error: &DslError) -> &(dyn std::error::Error + 'static) {
            error
        }

        let error = DslError::EvalError("path evaluation failed".into());
        assert_eq!(
            as_std_error(&error).to_string(),
            "eval error: path evaluation failed"
        );
        assert!(as_std_error(&error).source().is_none());
    }

    #[test]
    fn every_dsl_error_category_converts_to_invalid_input_with_display_context() {
        let cases = [
            (
                DslError::LexError {
                    message: "bad character".into(),
                    span: Span::new(1, 2),
                },
                "lex error at 1-2: bad character",
            ),
            (
                DslError::ParseError {
                    message: "bad grammar".into(),
                    span: Span::new(3, 8),
                },
                "parse error at 3-8: bad grammar",
            ),
            (
                DslError::CompileError {
                    message: "bad type".into(),
                    span: Some(Span::new(13, 17)),
                },
                "compile error at 13-17: bad type",
            ),
            (
                DslError::CompileError {
                    message: "global failure".into(),
                    span: None,
                },
                "compile error: global failure",
            ),
            (
                DslError::EvalError("non-finite path value".into()),
                "eval error: non-finite path value",
            ),
        ];

        for (error, expected) in cases {
            let pricing_error = PricingError::from(error);
            assert_eq!(
                pricing_error,
                PricingError::InvalidInput(expected.to_string())
            );
            assert_eq!(
                pricing_error.to_string(),
                format!("invalid input: {expected}")
            );
        }
    }

    #[test]
    fn annotate_source_reports_one_based_ascii_line_and_column() {
        let source = "product\n  maturity\n  pay";

        assert_eq!(annotate_source(source, Span::new(0, 7)), "line 1, col 1");
        assert_eq!(annotate_source(source, Span::new(7, 7)), "line 1, col 8");
        assert_eq!(annotate_source(source, Span::new(8, 10)), "line 2, col 1");
        assert_eq!(annotate_source(source, Span::new(12, 20)), "line 2, col 5");
        assert_eq!(annotate_source(source, Span::new(21, 24)), "line 3, col 3");
    }

    #[test]
    fn annotate_source_counts_unicode_characters_not_utf8_bytes() {
        let source = "προduct\n  βeta";
        let beta = source.find('β').expect("test fixture contains beta");
        let eta = source.find("eta").expect("test fixture contains eta");

        assert_eq!(
            annotate_source(source, Span::new(beta, beta + 2)),
            "line 2, col 3"
        );
        assert_eq!(
            annotate_source(source, Span::new(eta, eta + 3)),
            "line 2, col 4"
        );
    }

    #[test]
    fn annotate_source_clamps_out_of_range_and_non_boundary_offsets() {
        let source = "α\nβ";
        let inside_alpha = 1;

        assert_eq!(
            annotate_source(source, Span::new(inside_alpha, usize::MAX)),
            "line 1, col 1"
        );
        assert_eq!(
            annotate_source(source, Span::new(usize::MAX, usize::MAX)),
            "line 2, col 2"
        );
        assert_eq!(annotate_source("", Span::new(99, 0)), "line 1, col 1");
    }

    #[test]
    fn annotate_source_handles_crlf_and_ignores_span_end() {
        let source = "first\r\nsecond";
        let second = source.find("second").expect("test fixture contains second");

        assert_eq!(
            annotate_source(source, Span::new(second, second)),
            "line 2, col 1"
        );
        assert_eq!(
            annotate_source(source, Span::new(second + 3, 0)),
            "line 2, col 4"
        );
    }
}
