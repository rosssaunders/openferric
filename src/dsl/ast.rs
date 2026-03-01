//! Abstract syntax tree for the DSL parser output.
//!
//! Each AST node carries a source `Span` for error reporting.

use crate::dsl::error::Span;

/// Top-level product definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductDef {
    pub name: String,
    pub span: Span,
    pub body: Vec<ProductItem>,
}

/// Items that can appear inside a `product { ... }` block.
#[derive(Debug, Clone, PartialEq)]
pub enum ProductItem {
    Notional(f64, Span),
    Maturity(f64, Span),
    Underlyings(Vec<UnderlyingDecl>, Span),
    State(Vec<StateDecl>, Span),
    Schedule(ScheduleDef),
}

/// Underlying declaration: `NAME = asset(index)`.
#[derive(Debug, Clone, PartialEq)]
pub struct UnderlyingDecl {
    pub name: String,
    pub asset_index: usize,
    pub span: Span,
}

/// State variable declaration: `name: type = initial_value`.
#[derive(Debug, Clone, PartialEq)]
pub struct StateDecl {
    pub name: String,
    pub type_name: String,
    pub initial_value: AstExpr,
    pub span: Span,
}

/// Schedule definition: `schedule <freq> from <start> to <end> { ... }`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleDef {
    pub frequency: ScheduleFreq,
    pub start: f64,
    pub end: f64,
    pub body: Vec<AstStatement>,
    pub span: Span,
}

/// Schedule frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleFreq {
    /// Every N years.
    Custom(f64),
    Monthly,
    Quarterly,
    SemiAnnual,
    Annual,
}

impl ScheduleFreq {
    /// Returns the period in year fractions.
    pub fn period(&self) -> f64 {
        match self {
            Self::Custom(p) => *p,
            Self::Monthly => 1.0 / 12.0,
            Self::Quarterly => 0.25,
            Self::SemiAnnual => 0.5,
            Self::Annual => 1.0,
        }
    }

    /// Generates observation dates from `start` to `end` inclusive.
    pub fn generate_dates(&self, start: f64, end: f64) -> Vec<f64> {
        let period = self.period();
        if period <= 0.0 {
            return vec![end];
        }
        let mut dates = Vec::new();
        let mut t = start;
        while t <= end + 1e-12 {
            dates.push(t);
            t += period;
        }
        // Ensure the end date is included.
        if dates.is_empty() || (dates.last().unwrap() - end).abs() > 1e-12 {
            dates.push(end);
        }
        dates
    }
}

/// Expression in the AST.
#[derive(Debug, Clone, PartialEq)]
pub struct AstExpr {
    pub kind: AstExprKind,
    pub span: Span,
}

/// Expression variants.
#[derive(Debug, Clone, PartialEq)]
pub enum AstExprKind {
    /// Numeric literal (e.g., `0.08`, `1_000_000`).
    NumberLit(f64),
    /// Boolean literal (`true`, `false`).
    BoolLit(bool),
    /// Identifier reference (variable or keyword).
    Ident(String),
    /// Binary operation.
    BinOp {
        op: AstBinOp,
        lhs: Box<AstExpr>,
        rhs: Box<AstExpr>,
    },
    /// Unary operation.
    UnaryOp {
        op: AstUnaryOp,
        operand: Box<AstExpr>,
    },
    /// Function call: `name(args...)`.
    FnCall { name: String, args: Vec<AstExpr> },
}

/// Binary operators in the AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Unary operators in the AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstUnaryOp {
    Neg,
    Not,
}

/// Statement in the AST.
#[derive(Debug, Clone, PartialEq)]
pub struct AstStatement {
    pub kind: AstStatementKind,
    pub span: Span,
}

/// Statement variants.
#[derive(Debug, Clone, PartialEq)]
pub enum AstStatementKind {
    /// `let name = expr`.
    Let { name: String, expr: AstExpr },
    /// `if cond { ... } [else { ... }]`.
    If {
        condition: AstExpr,
        then_body: Vec<AstStatement>,
        else_body: Vec<AstStatement>,
    },
    /// `pay expr`.
    Pay { amount: AstExpr },
    /// `redeem expr`.
    Redeem { amount: AstExpr },
    /// `set name = expr`.
    SetState { name: String, expr: AstExpr },
    /// `skip`.
    Skip,
}
