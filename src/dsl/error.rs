//! DSL error types with span-based diagnostics.

use crate::core::PricingError;
use std::fmt;

/// Source span for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    let before = &source[..span.start.min(source.len())];
    let line = before.chars().filter(|&c| c == '\n').count() + 1;
    let col = before
        .rfind('\n')
        .map_or(span.start, |nl| span.start - nl - 1)
        + 1;
    format!("line {line}, col {col}")
}
