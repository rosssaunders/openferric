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
            let dates = sched_def.frequency.generate_dates(sched_def.start, sched_def.end);
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
            let then_body = compile_statements(then_body, scope)?;
            let else_body = compile_statements(else_body, scope)?;
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
            let slot = scope.state_vars.get(name).copied().ok_or_else(|| {
                DslError::CompileError {
                    message: format!("undefined state variable '{name}'"),
                    span: Some(stmt.span),
                }
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
            Ok(Expr::UnaryOp {
                op,
                operand: Box::new(operand),
            })
        }
        AstExprKind::FnCall { name, args } => compile_fn_call(name, args, expr.span, scope),
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
    // Check built-in names.
    match name {
        "notional" => Ok(Expr::Notional),
        "observation_date" => Ok(Expr::ObservationDate),
        "is_final" => Ok(Expr::IsFinal),
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
