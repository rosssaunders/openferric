use openferric::dsl::ast::*;
use openferric::dsl::error::Span;

/// Kind of symbol in the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Underlying,
    StateVar,
    Local,
    Builtin,
    BuiltinFn,
}

/// Scope in which a symbol is visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SymbolScope {
    Product,
    Schedule(usize),
}

/// Information about a declared symbol.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct SymbolRef {
    pub name: String,
    pub span: Span,
    pub def_span: Span,
    pub kind: SymbolKind,
    pub type_hint: &'static str,
    pub doc: &'static str,
}

/// Symbol table built from the AST.
#[derive(Debug, Default)]
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
                    declarations.push(SymbolInfo {
                        name: decl.name.clone(),
                        kind: SymbolKind::Underlying,
                        type_hint: "float",
                        def_span: decl.span,
                        scope: SymbolScope::Product,
                        doc: "Underlying equity",
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
                // Clone the product-level declarations as the scope for this schedule.
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

/// Walk statements, collecting local declarations and references.
///
/// `scope_decls` is the set of symbols visible for identifier resolution (including locals added
/// during this walk). `out_decls` is the output list where new declarations are appended.
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
                    // Find the actual position of the identifier name within the statement span.
                    let stmt_text = &source[stmt.span.start..stmt.span.end.min(source.len())];
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
