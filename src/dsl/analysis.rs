//! Pure analysis module shared between the LSP server and WASM bindings.
//!
//! Provides symbol table construction, diagnostics, completions, hover info,
//! semantic tokens, and goto-definition — all without any `tower_lsp` dependency.

use serde::{Deserialize, Serialize};

use crate::dsl::ast::*;
use crate::dsl::error::{DslError, Span};
use crate::dsl::ir::CompiledProduct;
use crate::dsl::lexer::{self, Token, TokenKind};
use crate::dsl::{compiler, parser};

// ---------------------------------------------------------------------------
// Symbol table types
// ---------------------------------------------------------------------------

/// Kind of symbol in the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolKind {
    Underlying,
    StateVar,
    Local,
    Builtin,
    BuiltinFn,
}

/// Scope in which a symbol is visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum SymbolScope {
    Product,
    Schedule(usize),
}

/// Information about a declared symbol.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub type_hint: &'static str,
    pub def_span: Span,
    #[allow(dead_code)]
    pub scope: SymbolScope,
    pub doc: &'static str,
}

/// A reference to a symbol at a specific location in the source.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolRef {
    pub name: String,
    pub span: Span,
    pub def_span: Span,
    pub kind: SymbolKind,
    pub type_hint: &'static str,
    pub doc: &'static str,
}

/// Symbol table built from the AST.
#[derive(Debug, Default, Serialize)]
pub struct SymbolTable {
    pub declarations: Vec<SymbolInfo>,
    pub references: Vec<SymbolRef>,
}

impl SymbolTable {
    /// Find a declaration whose span contains the given byte offset.
    pub fn declaration_at(&self, offset: usize) -> Option<&SymbolInfo> {
        self.declarations
            .iter()
            .find(|s| offset >= s.def_span.start && offset < s.def_span.end)
    }

    /// Find a reference whose span contains the given byte offset.
    pub fn reference_at(&self, offset: usize) -> Option<&SymbolRef> {
        self.references
            .iter()
            .find(|r| offset >= r.span.start && offset < r.span.end)
    }
}

// ---------------------------------------------------------------------------
// Completion types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionCandidate {
    pub label: String,
    pub kind: CompletionCandidateKind,
    pub detail: Option<String>,
    pub documentation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionCandidateKind {
    Keyword,
    Variable,
    Function,
    EnumMember,
    Constant,
}

// ---------------------------------------------------------------------------
// Hover types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverInfo {
    pub markdown: String,
    pub start: usize,
    pub end: usize,
}

// ---------------------------------------------------------------------------
// Semantic token types
// ---------------------------------------------------------------------------

/// Delta-encoded semantic token data (same layout as LSP SemanticToken).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTokenData {
    pub delta_line: u32,
    pub delta_start: u32,
    pub length: u32,
    pub token_type: u32,
    pub modifiers: u32,
}

/// Token type indices — must match the legend sent to Monaco/LSP.
pub const TOKEN_KEYWORD: u32 = 0;
pub const TOKEN_VARIABLE: u32 = 1;
pub const TOKEN_FUNCTION: u32 = 2;
pub const TOKEN_NUMBER: u32 = 3;
pub const TOKEN_STRING: u32 = 4;
pub const TOKEN_OPERATOR: u32 = 5;
pub const TOKEN_ENUM_MEMBER: u32 = 6;
#[allow(dead_code)]
pub const TOKEN_COMMENT: u32 = 7;

// ---------------------------------------------------------------------------
// Diagnostic info (pure — no tower_lsp Diagnostic)
// ---------------------------------------------------------------------------

/// A single diagnostic produced by the analysis pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticInfo {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

// =========================================================================
// Functions
// =========================================================================

// ---------------------------------------------------------------------------
// parse_and_diagnose
// ---------------------------------------------------------------------------

/// Parse and compile source, returning the AST, compiled product, and diagnostics.
pub fn parse_and_diagnose(
    source: &str,
) -> (
    Option<ProductDef>,
    Option<CompiledProduct>,
    Vec<DiagnosticInfo>,
) {
    let tokens = match lexer::tokenize(source) {
        Ok(t) => t,
        Err(e) => return (None, None, vec![error_to_diagnostic(&e)]),
    };

    let ast = match parser::parse(tokens) {
        Ok(a) => a,
        Err(e) => return (None, None, vec![error_to_diagnostic(&e)]),
    };

    match compiler::compile(&ast) {
        Ok(product) => (Some(ast), Some(product), vec![]),
        Err(e) => (Some(ast), None, vec![error_to_diagnostic(&e)]),
    }
}

fn error_to_diagnostic(error: &DslError) -> DiagnosticInfo {
    let (message, start, end) = match error {
        DslError::LexError { message, span } => (message.clone(), span.start, span.end),
        DslError::ParseError { message, span } => (message.clone(), span.start, span.end),
        DslError::CompileError { message, span } => {
            let (s, e) = span.map_or((0, 1), |sp| (sp.start, sp.end));
            (message.clone(), s, e)
        }
        DslError::EvalError(msg) => (msg.clone(), 0, 1),
    };

    DiagnosticInfo {
        severity: DiagnosticSeverity::Error,
        message,
        start,
        end,
    }
}

// ---------------------------------------------------------------------------
// build_symbol_table
// ---------------------------------------------------------------------------

/// Build a symbol table from a parsed AST.
pub fn build_symbol_table(ast: &ProductDef, source: &str) -> SymbolTable {
    let mut declarations = Vec::new();
    let mut references = Vec::new();

    // Add builtins (no source span — use 0..0).
    let builtins = [
        ("notional", "float", "Product face value"),
        (
            "observation_date",
            "float",
            "Current observation date in year fractions",
        ),
        ("is_final", "bool", "True on the last observation date"),
        ("maturity", "float", "Product maturity in year fractions"),
    ];
    for (name, ty, doc) in &builtins {
        declarations.push(SymbolInfo {
            name: name.to_string(),
            kind: SymbolKind::Builtin,
            type_hint: ty,
            def_span: Span::new(0, 0),
            scope: SymbolScope::Product,
            doc,
        });
    }

    let builtin_fns: &[(&str, &str, &str)] = &[
        (
            "worst_of",
            "worst_of(values: [float]) -> float",
            "Minimum of a vector",
        ),
        (
            "best_of",
            "best_of(values: [float]) -> float",
            "Maximum of a vector",
        ),
        (
            "performances",
            "performances() -> [float]",
            "Spot/initial for each underlying",
        ),
        (
            "price",
            "price(asset_index: int) -> float",
            "Current spot price of underlying",
        ),
        (
            "min",
            "min(a: float, b: float) -> float",
            "Minimum of two values",
        ),
        (
            "max",
            "max(a: float, b: float) -> float",
            "Maximum of two values",
        ),
        ("abs", "abs(x: float) -> float", "Absolute value"),
        ("exp", "exp(x: float) -> float", "Exponential function"),
        ("log", "log(x: float) -> float", "Natural logarithm"),
    ];
    for (name, ty, doc) in builtin_fns {
        declarations.push(SymbolInfo {
            name: name.to_string(),
            kind: SymbolKind::BuiltinFn,
            type_hint: ty,
            def_span: Span::new(0, 0),
            scope: SymbolScope::Product,
            doc,
        });
    }

    // Walk AST items.
    let mut schedule_index = 0usize;
    for item in &ast.body {
        match item {
            ProductItem::Underlyings(decls, _) => {
                for decl in decls {
                    let doc = match decl.underlying_type {
                        crate::dsl::ir::UnderlyingType::Equity => "Underlying equity",
                        crate::dsl::ir::UnderlyingType::Fx => "Underlying FX pair",
                        crate::dsl::ir::UnderlyingType::Commodity => "Underlying commodity",
                        crate::dsl::ir::UnderlyingType::Rate => "Underlying interest rate",
                    };
                    declarations.push(SymbolInfo {
                        name: decl.name.clone(),
                        kind: SymbolKind::Underlying,
                        type_hint: "float",
                        def_span: decl.span,
                        scope: SymbolScope::Product,
                        doc,
                    });
                }
            }
            ProductItem::State(decls, _) => {
                for decl in decls {
                    let type_hint = match decl.type_name.as_str() {
                        "bool" => "bool",
                        "float" => "float",
                        _ => "unknown",
                    };
                    declarations.push(SymbolInfo {
                        name: decl.name.clone(),
                        kind: SymbolKind::StateVar,
                        type_hint,
                        def_span: decl.span,
                        scope: SymbolScope::Product,
                        doc: "State variable (mutable across dates)",
                    });
                }
            }
            ProductItem::Schedule(sched) => {
                let mut scope_decls = declarations.clone();
                walk_statements(
                    &sched.body,
                    &mut scope_decls,
                    &mut references,
                    &mut declarations,
                    SymbolScope::Schedule(schedule_index),
                    source,
                );
                schedule_index += 1;
            }
            _ => {}
        }
    }

    SymbolTable {
        declarations,
        references,
    }
}

fn walk_statements(
    stmts: &[AstStatement],
    scope_decls: &mut Vec<SymbolInfo>,
    refs: &mut Vec<SymbolRef>,
    out_decls: &mut Vec<SymbolInfo>,
    scope: SymbolScope,
    source: &str,
) {
    for stmt in stmts {
        match &stmt.kind {
            AstStatementKind::Let { name, expr } => {
                walk_expr(expr, scope_decls, refs);
                let info = SymbolInfo {
                    name: name.clone(),
                    kind: SymbolKind::Local,
                    type_hint: "float",
                    def_span: stmt.span,
                    scope,
                    doc: "Local variable",
                };
                scope_decls.push(info.clone());
                out_decls.push(info);
            }
            AstStatementKind::If {
                condition,
                then_body,
                else_body,
            } => {
                walk_expr(condition, scope_decls, refs);
                walk_statements(then_body, scope_decls, refs, out_decls, scope, source);
                walk_statements(else_body, scope_decls, refs, out_decls, scope, source);
            }
            AstStatementKind::Pay { amount } => {
                walk_expr(amount, scope_decls, refs);
            }
            AstStatementKind::Redeem { amount } => {
                walk_expr(amount, scope_decls, refs);
            }
            AstStatementKind::SetState { name, expr } => {
                walk_expr(expr, scope_decls, refs);
                if let Some(decl) = scope_decls.iter().find(|d| d.name == *name) {
                    let start = stmt.span.start.min(source.len());
                    let end = stmt.span.end.min(source.len()).max(start);
                    let stmt_text = &source[start..end];
                    let name_offset = stmt_text
                        .find(name.as_str())
                        .map(|i| stmt.span.start + i)
                        .unwrap_or(stmt.span.start + 4);
                    refs.push(SymbolRef {
                        name: name.clone(),
                        span: Span::new(name_offset, name_offset + name.len()),
                        def_span: decl.def_span,
                        kind: decl.kind,
                        type_hint: decl.type_hint,
                        doc: decl.doc,
                    });
                }
            }
            AstStatementKind::Skip => {}
        }
    }
}

fn walk_expr(expr: &AstExpr, scope_decls: &[SymbolInfo], refs: &mut Vec<SymbolRef>) {
    match &expr.kind {
        AstExprKind::Ident(name) => {
            if let Some(d) = scope_decls.iter().find(|d| d.name == *name) {
                refs.push(SymbolRef {
                    name: name.clone(),
                    span: expr.span,
                    def_span: d.def_span,
                    kind: d.kind,
                    type_hint: d.type_hint,
                    doc: d.doc,
                });
            }
        }
        AstExprKind::BinOp { lhs, rhs, .. } => {
            walk_expr(lhs, scope_decls, refs);
            walk_expr(rhs, scope_decls, refs);
        }
        AstExprKind::UnaryOp { operand, .. } => {
            walk_expr(operand, scope_decls, refs);
        }
        AstExprKind::FnCall { name, args } => {
            if let Some(d) = scope_decls.iter().find(|d| d.name == *name) {
                refs.push(SymbolRef {
                    name: name.clone(),
                    span: Span::new(expr.span.start, expr.span.start + name.len()),
                    def_span: d.def_span,
                    kind: d.kind,
                    type_hint: d.type_hint,
                    doc: d.doc,
                });
            }
            for arg in args {
                walk_expr(arg, scope_decls, refs);
            }
        }
        AstExprKind::NumberLit(_) | AstExprKind::BoolLit(_) => {}
    }
}

// ---------------------------------------------------------------------------
// completions
// ---------------------------------------------------------------------------

/// Provide context-aware completions at the given byte offset.
pub fn completions(source: &str, symbols: &SymbolTable, offset: usize) -> Vec<CompletionCandidate> {
    let context = determine_context(source, offset);

    match context {
        Context::TopLevel => top_level_completions(),
        Context::AfterSchedule => frequency_completions(),
        Context::StatementPosition => statement_completions(),
        Context::Expression => expression_completions(symbols),
        Context::AfterSet => state_var_completions(symbols),
        Context::TypePosition => type_completions(),
    }
}

#[derive(Debug)]
enum Context {
    TopLevel,
    AfterSchedule,
    StatementPosition,
    Expression,
    AfterSet,
    TypePosition,
}

fn determine_context(source: &str, offset: usize) -> Context {
    let before = &source[..offset.min(source.len())];

    let line_start = before.rfind('\n').map_or(0, |nl| nl + 1);
    let line = before[line_start..].trim();

    if line.starts_with("set ") && !line.contains('=') {
        return Context::AfterSet;
    }

    if line.starts_with("schedule") && !line.contains("from") {
        return Context::AfterSchedule;
    }

    if line.contains(':') && !line.contains('=') && is_in_state_block(before) {
        return Context::TypePosition;
    }

    if is_in_schedule_body(before) {
        if line.is_empty() || line_is_statement_start(line) {
            return Context::StatementPosition;
        }
        return Context::Expression;
    }

    Context::TopLevel
}

fn is_in_state_block(before: &str) -> bool {
    for line in before.lines().rev() {
        let trimmed = line.trim();
        if trimmed == "state" {
            return true;
        }
        if trimmed.starts_with("schedule")
            || trimmed.starts_with("underlyings")
            || trimmed.starts_with("product")
        {
            return false;
        }
    }
    false
}

fn is_in_schedule_body(before: &str) -> bool {
    for line in before.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with("schedule") {
            return true;
        }
        if trimmed.starts_with("product")
            || trimmed.starts_with("underlyings")
            || trimmed == "state"
        {
            return false;
        }
    }
    false
}

fn line_is_statement_start(line: &str) -> bool {
    let first_word = line.split_whitespace().next().unwrap_or("");
    matches!(
        first_word,
        "" | "let" | "if" | "pay" | "redeem" | "set" | "skip" | "else"
    )
}

fn top_level_completions() -> Vec<CompletionCandidate> {
    vec![
        kw("notional:", "Product face value"),
        kw("maturity:", "Product maturity in year fractions"),
        kw("underlyings", "Declare underlying assets"),
        kw("state", "Declare mutable state variables"),
        kw("schedule", "Define observation schedule"),
    ]
}

fn frequency_completions() -> Vec<CompletionCandidate> {
    vec![
        em("monthly", "Every month (period = 1/12)"),
        em("quarterly", "Every quarter (period = 0.25)"),
        em("semi_annual", "Every 6 months (period = 0.5)"),
        em("annual", "Every year (period = 1.0)"),
    ]
}

fn statement_completions() -> Vec<CompletionCandidate> {
    vec![
        kw("let", "Bind a local variable"),
        kw("if", "Conditional branch"),
        kw("pay", "Record a cashflow payment"),
        kw("redeem", "Final payment and terminate"),
        kw("set", "Update a state variable"),
        kw("skip", "Skip this observation date"),
    ]
}

fn expression_completions(symbols: &SymbolTable) -> Vec<CompletionCandidate> {
    let mut items = Vec::new();
    for sym in &symbols.declarations {
        match sym.kind {
            SymbolKind::Local | SymbolKind::StateVar | SymbolKind::Underlying => {
                items.push(CompletionCandidate {
                    label: sym.name.clone(),
                    kind: CompletionCandidateKind::Variable,
                    detail: Some(format!("{}: {}", sym.name, sym.type_hint)),
                    documentation: Some(sym.doc.to_string()),
                });
            }
            SymbolKind::Builtin => {
                items.push(CompletionCandidate {
                    label: sym.name.clone(),
                    kind: CompletionCandidateKind::Constant,
                    detail: Some(format!("{}: {}", sym.name, sym.type_hint)),
                    documentation: Some(sym.doc.to_string()),
                });
            }
            SymbolKind::BuiltinFn => {
                items.push(CompletionCandidate {
                    label: sym.name.clone(),
                    kind: CompletionCandidateKind::Function,
                    detail: Some(sym.type_hint.to_string()),
                    documentation: Some(sym.doc.to_string()),
                });
            }
        }
    }
    items
}

fn state_var_completions(symbols: &SymbolTable) -> Vec<CompletionCandidate> {
    symbols
        .declarations
        .iter()
        .filter(|s| s.kind == SymbolKind::StateVar)
        .map(|s| CompletionCandidate {
            label: s.name.clone(),
            kind: CompletionCandidateKind::Variable,
            detail: Some(format!("{}: {}", s.name, s.type_hint)),
            documentation: None,
        })
        .collect()
}

fn type_completions() -> Vec<CompletionCandidate> {
    vec![
        kw("bool", "Boolean type"),
        kw("float", "Floating-point number type"),
    ]
}

fn kw(label: &str, detail: &str) -> CompletionCandidate {
    CompletionCandidate {
        label: label.into(),
        kind: CompletionCandidateKind::Keyword,
        detail: Some(detail.into()),
        documentation: None,
    }
}

fn em(label: &str, detail: &str) -> CompletionCandidate {
    CompletionCandidate {
        label: label.into(),
        kind: CompletionCandidateKind::EnumMember,
        detail: Some(detail.into()),
        documentation: None,
    }
}

// ---------------------------------------------------------------------------
// hover_info
// ---------------------------------------------------------------------------

/// Provide hover information at the given byte offset.
pub fn hover_info(source: &str, symbols: &SymbolTable, offset: usize) -> Option<HoverInfo> {
    // Check if cursor is on a reference.
    if let Some(sym_ref) = symbols.reference_at(offset) {
        let markdown =
            format_symbol_hover(&sym_ref.name, sym_ref.kind, sym_ref.type_hint, sym_ref.doc);
        return Some(HoverInfo {
            markdown,
            start: sym_ref.span.start,
            end: sym_ref.span.end,
        });
    }

    // Check if cursor is on a declaration.
    if let Some(sym) = symbols.declaration_at(offset) {
        let markdown = format_symbol_hover(&sym.name, sym.kind, sym.type_hint, sym.doc);
        return Some(HoverInfo {
            markdown,
            start: sym.def_span.start,
            end: sym.def_span.end,
        });
    }

    // Check if cursor is on a keyword.
    keyword_hover(source, offset)
}

fn format_symbol_hover(name: &str, kind: SymbolKind, type_hint: &str, doc: &str) -> String {
    let kind_label = match kind {
        SymbolKind::Underlying => "underlying equity",
        SymbolKind::StateVar => "state variable (mutable across dates)",
        SymbolKind::Local => "local variable",
        SymbolKind::Builtin => "built-in",
        SymbolKind::BuiltinFn => "built-in function",
    };

    match kind {
        SymbolKind::BuiltinFn => {
            format!("`{type_hint}` \u{2014} {doc}")
        }
        _ => {
            format!("`{name}: {type_hint}` \u{2014} {kind_label}")
        }
    }
}

fn keyword_hover(source: &str, offset: usize) -> Option<HoverInfo> {
    let word = word_at_offset(source, offset)?;

    let doc = match word.text {
        "product" => "Define a structured product",
        "notional" => "Product face value",
        "maturity" => "Product maturity in year fractions",
        "underlyings" => "Declare underlying assets for the product",
        "state" => "Declare mutable state variables that persist across observation dates",
        "schedule" => "Define an observation schedule with frequency and date range",
        "let" => "Bind a local variable within the current observation",
        "if" => "Conditional branch",
        "then" => "Begin the body of a conditional branch",
        "else" => "Alternative branch of a conditional",
        "pay" => "Record a cashflow payment at the current observation date",
        "redeem" => "Record final payment and terminate the product",
        "set" => "Update a state variable's value",
        "skip" => "Skip the current observation date without payment",
        "and" => "Logical AND operator",
        "or" => "Logical OR operator",
        "not" => "Logical NOT operator",
        "from" => "Schedule start date",
        "to" => "Schedule end date",
        "monthly" => "Schedule frequency: every month (period = 1/12)",
        "quarterly" => "Schedule frequency: every quarter (period = 0.25)",
        "semi_annual" => "Schedule frequency: every 6 months (period = 0.5)",
        "annual" => "Schedule frequency: every year (period = 1.0)",
        "true" | "false" => "Boolean literal",
        "bool" => "Boolean type",
        "float" => "Floating-point number type",
        "asset" => "Reference an underlying by index: asset(N) (synonym for equity)",
        "equity" => "Equity underlying type: equity(N)",
        "fx" => "FX pair underlying type: fx(N)",
        "commodity" => "Commodity underlying type: commodity(N)",
        "rate" => "Interest rate underlying type: rate(N)",
        _ => return None,
    };

    Some(HoverInfo {
        markdown: format!("`{0}` \u{2014} {doc}", word.text),
        start: word.start,
        end: word.end,
    })
}

struct WordAt<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

fn word_at_offset(source: &str, offset: usize) -> Option<WordAt<'_>> {
    if offset > source.len() {
        return None;
    }
    let bytes = source.as_bytes();
    let mut start = offset;
    while start > 0 && is_ident_char(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = offset;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    if start == end {
        return None;
    }
    Some(WordAt {
        text: &source[start..end],
        start,
        end,
    })
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// goto_definition
// ---------------------------------------------------------------------------

/// Go to the definition of the symbol at the given byte offset.
/// Returns the definition span, or `None` for builtins.
pub fn goto_definition(source: &str, symbols: &SymbolTable, offset: usize) -> Option<Span> {
    let _ = source; // available for future use
    let sym_ref = symbols.reference_at(offset)?;

    if sym_ref.kind == SymbolKind::Builtin || sym_ref.kind == SymbolKind::BuiltinFn {
        return None;
    }

    if sym_ref.def_span.start == 0 && sym_ref.def_span.end == 0 {
        return None;
    }

    Some(sym_ref.def_span)
}

// ---------------------------------------------------------------------------
// semantic_token_data
// ---------------------------------------------------------------------------

/// Compute delta-encoded semantic tokens for the source.
pub fn semantic_token_data(source: &str, symbols: &SymbolTable) -> Vec<SemanticTokenData> {
    let tokens: Vec<Token> = match lexer::tokenize(source) {
        Ok(t) => t,
        Err(_) => return vec![],
    };

    let mut result = Vec::new();
    let mut prev_line = 0u32;
    let mut prev_start = 0u32;

    for token in &tokens {
        let token_type = match &token.kind {
            TokenKind::Product
            | TokenKind::Notional
            | TokenKind::Maturity
            | TokenKind::Underlyings
            | TokenKind::State
            | TokenKind::Schedule
            | TokenKind::From
            | TokenKind::To
            | TokenKind::Let
            | TokenKind::If
            | TokenKind::Then
            | TokenKind::Else
            | TokenKind::Pay
            | TokenKind::Redeem
            | TokenKind::Set
            | TokenKind::Skip
            | TokenKind::Asset
            | TokenKind::Equity
            | TokenKind::Fx
            | TokenKind::Commodity
            | TokenKind::Rate
            | TokenKind::Bool
            | TokenKind::Float
            | TokenKind::True
            | TokenKind::False => TOKEN_KEYWORD,

            TokenKind::And | TokenKind::Or | TokenKind::Not => TOKEN_OPERATOR,

            TokenKind::Monthly
            | TokenKind::Quarterly
            | TokenKind::SemiAnnual
            | TokenKind::Annual => TOKEN_ENUM_MEMBER,

            TokenKind::Number(_) => TOKEN_NUMBER,
            TokenKind::StringLit(_) => TOKEN_STRING,

            TokenKind::Ident(name) => {
                if let Some(sym_ref) = symbols.reference_at(token.span.start) {
                    match sym_ref.kind {
                        SymbolKind::BuiltinFn => TOKEN_FUNCTION,
                        _ => TOKEN_VARIABLE,
                    }
                } else {
                    match name.as_str() {
                        "worst_of" | "best_of" | "performances" | "price" | "min" | "max"
                        | "abs" | "exp" | "log" => TOKEN_FUNCTION,
                        _ => TOKEN_VARIABLE,
                    }
                }
            }

            TokenKind::Indent | TokenKind::Dedent => continue,
            TokenKind::LParen | TokenKind::RParen | TokenKind::Colon | TokenKind::Comma => continue,
            TokenKind::Eq
            | TokenKind::EqEq
            | TokenKind::Ne
            | TokenKind::Lt
            | TokenKind::Le
            | TokenKind::Gt
            | TokenKind::Ge
            | TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash => TOKEN_OPERATOR,
        };

        let (line, col) = offset_to_line_col(source, token.span.start);
        let length = (token.span.end - token.span.start) as u32;

        let delta_line = line - prev_line;
        let delta_start = if delta_line == 0 {
            col - prev_start
        } else {
            col
        };

        result.push(SemanticTokenData {
            delta_line,
            delta_start,
            length,
            token_type,
            modifiers: 0,
        });

        prev_line = line;
        prev_start = col;
    }

    result
}

// ---------------------------------------------------------------------------
// Utility: offset_to_line_col
// ---------------------------------------------------------------------------

/// Convert a byte offset to (line, col), both 0-based.
pub fn offset_to_line_col(source: &str, offset: usize) -> (u32, u32) {
    let offset = offset.min(source.len());
    let before = &source[..offset];
    let line = before.matches('\n').count() as u32;
    let col = before.rfind('\n').map_or(offset, |nl| offset - nl - 1) as u32;
    (line, col)
}

/// Convert (line, col) (both 0-based) to a byte offset.
pub fn line_col_to_offset(source: &str, line: u32, col: u32) -> usize {
    let mut current_line = 0u32;
    for (i, c) in source.char_indices() {
        if current_line == line {
            let col_offset = col as usize;
            return (i + col_offset).min(source.len());
        }
        if c == '\n' {
            current_line += 1;
        }
    }
    source.len()
}
