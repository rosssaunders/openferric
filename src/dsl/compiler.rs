//! Compiler: AST → IR with variable resolution, type checking, and slot assignment.

use crate::dsl::ast::*;
use crate::dsl::error::{DslError, Span};
use crate::dsl::ir::*;
use std::collections::HashMap;

/// Variable scope for the compiler.
struct Scope {
    /// State variables: name → slot index.
    state_vars: HashMap<String, usize>,
    /// Local variables: name → slot index.
    locals: HashMap<String, usize>,
    /// Next available local slot.
    next_local: usize,
    /// Underlying names: name → asset index.
    underlyings: HashMap<String, usize>,
    /// Product notional.
    notional: f64,
    /// Product maturity.
    maturity: f64,
}

impl Scope {
    fn new() -> Self {
        Self {
            state_vars: HashMap::new(),
            locals: HashMap::new(),
            next_local: 0,
            underlyings: HashMap::new(),
            notional: 0.0,
            maturity: 0.0,
        }
    }

    fn alloc_local(&mut self, name: &str) -> usize {
        let slot = self.next_local;
        self.locals.insert(name.to_string(), slot);
        self.next_local += 1;
        slot
    }

    fn reset_locals(&mut self) {
        self.locals.clear();
        self.next_local = 0;
    }
}

/// Compile a parsed AST into a `CompiledProduct`.
pub fn compile(ast: &ProductDef) -> Result<CompiledProduct, DslError> {
    let mut scope = Scope::new();
    let mut notional = None;
    let mut maturity = None;
    let mut underlying_defs = Vec::new();
    let mut state_var_defs = Vec::new();
    let mut schedules = Vec::new();

    // First pass: extract product-level declarations.
    for item in &ast.body {
        match item {
            ProductItem::Notional(v, span) => {
                if notional.is_some() {
                    return Err(DslError::CompileError {
                        message: "duplicate notional declaration".to_string(),
                        span: Some(*span),
                    });
                }
                notional = Some(*v);
            }
            ProductItem::Maturity(v, span) => {
                if maturity.is_some() {
                    return Err(DslError::CompileError {
                        message: "duplicate maturity declaration".to_string(),
                        span: Some(*span),
                    });
                }
                maturity = Some(*v);
            }
            ProductItem::Underlyings(decls, _) => {
                for decl in decls {
                    if scope.underlyings.contains_key(&decl.name) {
                        return Err(DslError::CompileError {
                            message: format!("duplicate underlying '{}'", decl.name),
                            span: Some(decl.span),
                        });
                    }
                    scope
                        .underlyings
                        .insert(decl.name.clone(), decl.asset_index);
                    underlying_defs.push(UnderlyingDef {
                        name: decl.name.clone(),
                        asset_index: decl.asset_index,
                        underlying_type: decl.underlying_type,
                    });
                }
            }
            ProductItem::State(decls, _) => {
                for decl in decls {
                    let slot = state_var_defs.len();
                    if scope.state_vars.contains_key(&decl.name) {
                        return Err(DslError::CompileError {
                            message: format!("duplicate state variable '{}'", decl.name),
                            span: Some(decl.span),
                        });
                    }
                    let initial = compile_initial_value(&decl.initial_value, &decl.type_name)?;
                    scope.state_vars.insert(decl.name.clone(), slot);
                    state_var_defs.push(StateVarDef {
                        name: decl.name.clone(),
                        slot,
                        initial,
                    });
                }
            }
            ProductItem::Schedule(_) => {} // handled in second pass
        }
    }

    let notional = notional.ok_or_else(|| DslError::CompileError {
        message: "missing notional declaration".to_string(),
        span: None,
    })?;
    let maturity = maturity.ok_or_else(|| DslError::CompileError {
        message: "missing maturity declaration".to_string(),
        span: None,
    })?;

    scope.notional = notional;
    scope.maturity = maturity;

    // Second pass: compile schedules.
    for item in &ast.body {
        if let ProductItem::Schedule(sched_def) = item {
            scope.reset_locals();
            let mut dates = sched_def
                .frequency
                .generate_dates(sched_def.start, sched_def.end)
                .map_err(|message| DslError::CompileError {
                    message,
                    span: Some(sched_def.span),
                })?;
            // Observation dates beyond product maturity cannot be simulated
            // (paths end at maturity); drop them so that spots and discount
            // factors stay consistent. A schedule entirely beyond maturity is
            // a hard error.
            dates.retain(|&d| d <= maturity + 1e-9);
            if dates.is_empty() {
                return Err(DslError::CompileError {
                    message: format!(
                        "schedule starts after product maturity {maturity}; no observation date is <= maturity"
                    ),
                    span: Some(sched_def.span),
                });
            }
            let body = compile_statements(&sched_def.body, &mut scope)?;
            schedules.push(Schedule { dates, body });
        }
    }

    let num_underlyings = underlying_defs.len().max(1);

    Ok(CompiledProduct {
        name: ast.name.clone(),
        notional,
        maturity,
        num_underlyings,
        underlyings: underlying_defs,
        state_vars: state_var_defs,
        constants: vec![],
        schedules,
    })
}

fn compile_initial_value(expr: &AstExpr, type_name: &str) -> Result<Value, DslError> {
    match (&expr.kind, type_name) {
        (AstExprKind::BoolLit(b), "bool") => Ok(Value::Bool(*b)),
        (AstExprKind::NumberLit(n), "float") => Ok(Value::F64(*n)),
        (AstExprKind::NumberLit(n), "bool") => Ok(Value::Bool(*n != 0.0)),
        (AstExprKind::BoolLit(b), "float") => Ok(Value::F64(if *b { 1.0 } else { 0.0 })),
        _ => Err(DslError::CompileError {
            message: format!(
                "cannot use {:?} as initial value for type '{type_name}'",
                expr.kind
            ),
            span: Some(expr.span),
        }),
    }
}

fn compile_statements(
    stmts: &[AstStatement],
    scope: &mut Scope,
) -> Result<Vec<Statement>, DslError> {
    stmts.iter().map(|s| compile_statement(s, scope)).collect()
}

fn compile_statement(stmt: &AstStatement, scope: &mut Scope) -> Result<Statement, DslError> {
    match &stmt.kind {
        AstStatementKind::Let { name, expr } => {
            let compiled_expr = compile_expr(expr, scope)?;
            let slot = scope.alloc_local(name);
            Ok(Statement::Let {
                slot,
                expr: compiled_expr,
            })
        }
        AstStatementKind::If {
            condition,
            then_body,
            else_body,
        } => {
            let condition = compile_expr(condition, scope)?;
            // `let` bindings are scoped to their branch: a local declared
            // inside a branch must not be visible after the conditional
            // (it would read 0.0 when the branch was not taken).
            let outer_locals = scope.locals.clone();
            let then_body = compile_statements(then_body, scope)?;
            scope.locals = outer_locals.clone();
            let else_body = compile_statements(else_body, scope)?;
            scope.locals = outer_locals;
            Ok(Statement::If {
                condition,
                then_body,
                else_body,
            })
        }
        AstStatementKind::Pay { amount } => {
            let amount = compile_expr(amount, scope)?;
            Ok(Statement::Pay { amount })
        }
        AstStatementKind::Redeem { amount } => {
            let amount = compile_expr(amount, scope)?;
            Ok(Statement::Redeem { amount })
        }
        AstStatementKind::SetState { name, expr } => {
            let slot =
                scope
                    .state_vars
                    .get(name)
                    .copied()
                    .ok_or_else(|| DslError::CompileError {
                        message: format!("undefined state variable '{name}'"),
                        span: Some(stmt.span),
                    })?;
            let expr = compile_expr(expr, scope)?;
            Ok(Statement::SetState { slot, expr })
        }
        AstStatementKind::Skip => Ok(Statement::Skip),
    }
}

fn compile_expr(expr: &AstExpr, scope: &Scope) -> Result<Expr, DslError> {
    match &expr.kind {
        AstExprKind::NumberLit(n) => Ok(Expr::Literal(Value::F64(*n))),
        AstExprKind::BoolLit(b) => Ok(Expr::Literal(Value::Bool(*b))),
        AstExprKind::Ident(name) => resolve_ident(name, expr.span, scope),
        AstExprKind::BinOp { op, lhs, rhs } => {
            let lhs = compile_expr(lhs, scope)?;
            let rhs = compile_expr(rhs, scope)?;
            let op = match op {
                AstBinOp::Add => BinOp::Add,
                AstBinOp::Sub => BinOp::Sub,
                AstBinOp::Mul => BinOp::Mul,
                AstBinOp::Div => BinOp::Div,
                AstBinOp::Eq => BinOp::Eq,
                AstBinOp::Ne => BinOp::Ne,
                AstBinOp::Lt => BinOp::Lt,
                AstBinOp::Le => BinOp::Le,
                AstBinOp::Gt => BinOp::Gt,
                AstBinOp::Ge => BinOp::Ge,
                AstBinOp::And => BinOp::And,
                AstBinOp::Or => BinOp::Or,
            };
            // Constant-fold literal-only subtrees so they cost a single
            // PUSH_CONST at runtime. Folding mirrors the scalar interpreter
            // semantics exactly (division by zero yields NaN; equality uses
            // an absolute f64::EPSILON tolerance).
            if let (Expr::Literal(a), Expr::Literal(b)) = (&lhs, &rhs) {
                return Ok(Expr::Literal(fold_binop(op, *a, *b)));
            }
            Ok(Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        AstExprKind::UnaryOp { op, operand } => {
            let operand = compile_expr(operand, scope)?;
            let op = match op {
                AstUnaryOp::Neg => UnaryOp::Neg,
                AstUnaryOp::Not => UnaryOp::Not,
            };
            if let Expr::Literal(v) = &operand {
                return Ok(Expr::Literal(fold_unaryop(op, *v)));
            }
            Ok(Expr::UnaryOp {
                op,
                operand: Box::new(operand),
            })
        }
        AstExprKind::FnCall { name, args } => compile_fn_call(name, args, expr.span, scope),
    }
}

/// Fold a binary operation over two literal values.
///
/// Must match the scalar bytecode interpreter in `eval.rs` exactly.
fn fold_binop(op: BinOp, lhs: Value, rhs: Value) -> Value {
    let a = lhs.as_f64();
    let b = rhs.as_f64();
    match op {
        BinOp::Add => Value::F64(a + b),
        BinOp::Sub => Value::F64(a - b),
        BinOp::Mul => Value::F64(a * b),
        BinOp::Div => Value::F64(if b == 0.0 { f64::NAN } else { a / b }),
        BinOp::Eq => Value::Bool((a - b).abs() < f64::EPSILON),
        BinOp::Ne => Value::Bool((a - b).abs() >= f64::EPSILON),
        BinOp::Lt => Value::Bool(a < b),
        BinOp::Le => Value::Bool(a <= b),
        BinOp::Gt => Value::Bool(a > b),
        BinOp::Ge => Value::Bool(a >= b),
        BinOp::And => Value::Bool(lhs.as_bool() && rhs.as_bool()),
        BinOp::Or => Value::Bool(lhs.as_bool() || rhs.as_bool()),
    }
}

/// Fold a unary operation over a literal value.
fn fold_unaryop(op: UnaryOp, v: Value) -> Value {
    match op {
        UnaryOp::Neg => Value::F64(-v.as_f64()),
        UnaryOp::Not => Value::Bool(!v.as_bool()),
    }
}

fn resolve_ident(name: &str, span: Span, scope: &Scope) -> Result<Expr, DslError> {
    // Check local variables first.
    if let Some(&slot) = scope.locals.get(name) {
        return Ok(Expr::LocalVar(slot));
    }
    // Check state variables.
    if let Some(&slot) = scope.state_vars.get(name) {
        return Ok(Expr::StateVar(slot));
    }
    // Declared underlying names resolve to the asset's current spot price,
    // equivalent to `price(asset_index)`. This works identically in the
    // scalar interpreter, the SIMD batch evaluators, and the JIT because it
    // compiles to PUSH_CONST(asset_index) + PRICE.
    if let Some(&asset_index) = scope.underlyings.get(name) {
        return Ok(Expr::Call {
            func: BuiltinFn::Price,
            args: vec![Expr::Literal(Value::F64(asset_index as f64))],
        });
    }
    // Check built-in names.
    match name {
        "notional" => Ok(Expr::Notional),
        "observation_date" => Ok(Expr::ObservationDate),
        "is_final" => Ok(Expr::IsFinal),
        // `maturity` is a product-level constant, baked in at compile time.
        "maturity" => Ok(Expr::Literal(Value::F64(scope.maturity))),
        _ => Err(DslError::CompileError {
            message: format!("undefined variable '{name}'"),
            span: Some(span),
        }),
    }
}

fn compile_fn_call(
    name: &str,
    args: &[AstExpr],
    span: Span,
    scope: &Scope,
) -> Result<Expr, DslError> {
    let func = match name {
        "worst_of" => BuiltinFn::WorstOf,
        "best_of" => BuiltinFn::BestOf,
        "performances" => BuiltinFn::Performances,
        "price" => BuiltinFn::Price,
        "min" => BuiltinFn::Min,
        "max" => BuiltinFn::Max,
        "abs" => BuiltinFn::Abs,
        "exp" => BuiltinFn::Exp,
        "log" => BuiltinFn::Log,
        _ => {
            return Err(DslError::CompileError {
                message: format!("unknown function '{name}'"),
                span: Some(span),
            });
        }
    };

    let compiled_args: Vec<Expr> = args
        .iter()
        .map(|a| compile_expr(a, scope))
        .collect::<Result<_, _>>()?;

    Ok(Expr::Call {
        func,
        args: compiled_args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::lexer::tokenize;
    use crate::dsl::parser::parse;

    fn compile_str(source: &str) -> Result<CompiledProduct, DslError> {
        let tokens = tokenize(source)?;
        let ast = parse(tokens)?;
        compile(&ast)
    }

    #[test]
    fn compile_minimal_product() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 1000
    maturity: 1.0
",
        )
        .unwrap();

        assert_eq!(product.name, "Test");
        assert_eq!(product.notional, 1000.0);
        assert_eq!(product.maturity, 1.0);
        assert!(product.schedules.is_empty());
    }

    #[test]
    fn compile_with_state_and_schedule() {
        let product = compile_str(
            "\
product \"Autocall\"
    notional: 1_000_000
    maturity: 1.5
    underlyings
        SPX = asset(0)
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
",
        )
        .unwrap();

        assert_eq!(product.name, "Autocall");
        assert_eq!(product.notional, 1_000_000.0);
        assert_eq!(product.maturity, 1.5);
        assert_eq!(product.num_underlyings, 1);
        assert_eq!(product.state_vars.len(), 1);
        assert_eq!(product.state_vars[0].name, "ki_hit");
        assert_eq!(product.state_vars[0].initial, Value::Bool(false));

        assert_eq!(product.schedules.len(), 1);
        // Quarterly from 0.25 to 1.5 = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        assert_eq!(product.schedules[0].dates.len(), 6);
        assert!((product.schedules[0].dates[0] - 0.25).abs() < 1e-12);
        assert!((product.schedules[0].dates[5] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn compile_error_undefined_variable() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        pay undefined_var
",
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            DslError::CompileError { message, .. } => {
                assert!(message.contains("undefined variable"));
            }
            _ => panic!("expected CompileError"),
        }
    }

    #[test]
    fn compile_error_undefined_state_var() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        set nonexistent = true
",
        );
        assert!(result.is_err());
    }

    #[test]
    fn underlying_name_resolves_to_price_call() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
        SX5E = asset(1)
    schedule annual from 1.0 to 1.0
        pay SX5E * 0.5
",
        )
        .unwrap();

        let body = &product.schedules[0].body;
        match &body[0] {
            Statement::Pay { amount } => match amount {
                Expr::BinOp {
                    op: BinOp::Mul,
                    lhs,
                    ..
                } => {
                    assert_eq!(
                        **lhs,
                        Expr::Call {
                            func: BuiltinFn::Price,
                            args: vec![Expr::Literal(Value::F64(1.0))],
                        }
                    );
                }
                other => panic!("expected Mul binop, got {other:?}"),
            },
            other => panic!("expected Pay, got {other:?}"),
        }
    }

    #[test]
    fn maturity_resolves_to_constant() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 2.5
    schedule annual from 1.0 to 2.5
        pay notional * 0.08 * maturity
",
        )
        .unwrap();

        let body = &product.schedules[0].body;
        match &body[0] {
            Statement::Pay { amount } => match amount {
                Expr::BinOp {
                    op: BinOp::Mul,
                    rhs,
                    ..
                } => {
                    assert_eq!(**rhs, Expr::Literal(Value::F64(2.5)));
                }
                other => panic!("expected Mul binop, got {other:?}"),
            },
            other => panic!("expected Pay, got {other:?}"),
        }
    }

    #[test]
    fn branch_local_let_is_out_of_scope_after_if() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        if is_final then
            let x = 1.0
            pay x
        pay x
",
        );
        let err = result.unwrap_err();
        match err {
            DslError::CompileError { message, .. } => {
                assert!(message.contains("undefined variable 'x'"), "got: {message}");
            }
            other => panic!("expected CompileError, got {other:?}"),
        }
    }

    #[test]
    fn branch_local_let_not_visible_in_else() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        if is_final then
            let x = 1.0
            pay x
        else
            pay x
",
        );
        assert!(result.is_err());
    }

    #[test]
    fn top_level_let_visible_after_if() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        let x = 2.0
        if is_final then
            pay x
        pay x
",
        );
        assert!(product.is_ok());
    }

    #[test]
    fn tiny_period_schedule_errors_instead_of_oom() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 30.0
    schedule 0.000000001 from 0.0 to 30.0
        pay 1.0
",
        );
        let err = result.unwrap_err();
        match err {
            DslError::CompileError { message, .. } => {
                assert!(message.contains("observation dates"), "got: {message}");
            }
            other => panic!("expected CompileError, got {other:?}"),
        }
    }

    #[test]
    fn schedule_dates_beyond_maturity_are_truncated() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 5.0
        redeem notional
",
        )
        .unwrap();
        assert_eq!(product.schedules[0].dates, vec![1.0]);
    }

    #[test]
    fn schedule_entirely_beyond_maturity_is_error() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 2.0 to 5.0
        redeem notional
",
        );
        assert!(result.is_err());
    }

    #[test]
    fn literal_subtrees_are_constant_folded() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        pay 2.0 * (3.0 + 4.0) - 1.0
",
        )
        .unwrap();

        match &product.schedules[0].body[0] {
            Statement::Pay { amount } => {
                assert_eq!(*amount, Expr::Literal(Value::F64(13.0)));
            }
            other => panic!("expected Pay, got {other:?}"),
        }
    }

    #[test]
    fn constant_folding_division_by_zero_is_nan() {
        let product = compile_str(
            "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        let x = 1.0 / 0.0
        pay 0.0
",
        )
        .unwrap();

        match &product.schedules[0].body[0] {
            Statement::Let { expr, .. } => match expr {
                Expr::Literal(Value::F64(v)) => assert!(v.is_nan()),
                other => panic!("expected folded NaN literal, got {other:?}"),
            },
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn compile_error_duplicate_notional() {
        let result = compile_str(
            "\
product \"Test\"
    notional: 100
    notional: 200
    maturity: 1.0
",
        );
        assert!(result.is_err());
    }
}
