//! Evaluator for compiled DSL products.
//!
//! Walks the IR statement tree for each observation date on each MC path,
//! accumulating discounted cashflows and handling early termination.

use crate::dsl::error::DslError;
use crate::dsl::ir::{BinOp, BuiltinFn, CompiledProduct, Expr, Schedule, Statement, UnaryOp, Value};

/// Per-path evaluation context.
struct EvalContext<'a> {
    /// Current spot prices for each underlying at the current time step.
    spots: &'a [f64],
    /// Initial spot prices (t=0) for computing performances.
    initial_spots: &'a [f64],
    /// Product notional.
    notional: f64,
    /// Current observation date (year fraction).
    observation_date: f64,
    /// Whether this is the final observation in the schedule.
    is_final: bool,
    /// Local variable slots.
    locals: &'a mut [Value],
    /// State variable slots (mutable across observations).
    state: &'a mut [Value],
}

/// Result of evaluating one observation date.
#[derive(Debug)]
enum ObservationResult {
    /// Continue to next observation.
    Continue,
    /// Product redeemed — stop processing further dates.
    Redeemed,
    /// Skip remaining observations.
    Skipped,
}

/// Cashflow generated during evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Cashflow {
    pub time: f64,
    pub amount: f64,
}

/// Evaluate a compiled product on a single MC path.
///
/// `path_spots[t][asset]` gives the spot price of each underlying at each
/// time step. The first index is the time step (0 = initial), the second
/// is the asset index.
///
/// Returns the vector of cashflows generated on this path.
pub fn evaluate_product(
    product: &CompiledProduct,
    path_spots: &[Vec<f64>],
    initial_spots: &[f64],
    num_steps: usize,
    rate: f64,
) -> Result<f64, DslError> {
    let num_locals = product.max_local_slots();
    let mut locals = vec![Value::F64(0.0); num_locals];
    let mut state: Vec<Value> = product.state_vars.iter().map(|sv| sv.initial).collect();

    let mut pv = 0.0;

    for schedule in &product.schedules {
        let outcome = evaluate_schedule(
            product,
            schedule,
            path_spots,
            initial_spots,
            num_steps,
            rate,
            &mut locals,
            &mut state,
            &mut pv,
        )?;
        if matches!(outcome, ObservationResult::Redeemed | ObservationResult::Skipped) {
            break;
        }
    }

    Ok(pv)
}

fn evaluate_schedule(
    product: &CompiledProduct,
    schedule: &Schedule,
    path_spots: &[Vec<f64>],
    initial_spots: &[f64],
    num_steps: usize,
    rate: f64,
    locals: &mut [Value],
    state: &mut [Value],
    pv: &mut f64,
) -> Result<ObservationResult, DslError> {
    let maturity = product.maturity;
    let num_dates = schedule.dates.len();

    for (date_idx, &obs_date) in schedule.dates.iter().enumerate() {
        let is_final = date_idx == num_dates - 1;

        // Map observation date to path step index.
        let step_idx = if maturity > 0.0 {
            ((obs_date / maturity) * num_steps as f64).round() as usize
        } else {
            0
        };
        let step_idx = step_idx.min(path_spots.len() - 1);

        let spots = &path_spots[step_idx];
        // Reset locals for each observation date.
        for local in locals.iter_mut() {
            *local = Value::F64(0.0);
        }

        let mut ctx = EvalContext {
            spots,
            initial_spots,
            notional: product.notional,
            observation_date: obs_date,
            is_final,
            locals,
            state,
        };

        let mut cashflows = Vec::new();
        let result = execute_statements(&schedule.body, &mut ctx, &mut cashflows)?;

        // Discount and accumulate cashflows.
        for cf in &cashflows {
            let cf_df = (-rate * cf.time).exp();
            *pv += cf.amount * cf_df;
        }

        match result {
            ObservationResult::Redeemed | ObservationResult::Skipped => return Ok(result),
            ObservationResult::Continue => {}
        }
    }

    Ok(ObservationResult::Continue)
}

fn execute_statements(
    stmts: &[Statement],
    ctx: &mut EvalContext<'_>,
    cashflows: &mut Vec<Cashflow>,
) -> Result<ObservationResult, DslError> {
    for stmt in stmts {
        let result = execute_statement(stmt, ctx, cashflows)?;
        match result {
            ObservationResult::Continue => {}
            other => return Ok(other),
        }
    }
    Ok(ObservationResult::Continue)
}

fn execute_statement(
    stmt: &Statement,
    ctx: &mut EvalContext<'_>,
    cashflows: &mut Vec<Cashflow>,
) -> Result<ObservationResult, DslError> {
    match stmt {
        Statement::Let { slot, expr } => {
            let val = eval_expr(expr, ctx)?;
            ctx.locals[*slot] = val;
            Ok(ObservationResult::Continue)
        }
        Statement::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond = eval_expr(condition, ctx)?;
            if cond.as_bool() {
                execute_statements(then_body, ctx, cashflows)
            } else {
                execute_statements(else_body, ctx, cashflows)
            }
        }
        Statement::Pay { amount } => {
            let val = eval_expr(amount, ctx)?.as_f64();
            cashflows.push(Cashflow {
                time: ctx.observation_date,
                amount: val,
            });
            Ok(ObservationResult::Continue)
        }
        Statement::Redeem { amount } => {
            let val = eval_expr(amount, ctx)?.as_f64();
            cashflows.push(Cashflow {
                time: ctx.observation_date,
                amount: val,
            });
            Ok(ObservationResult::Redeemed)
        }
        Statement::SetState { slot, expr } => {
            let val = eval_expr(expr, ctx)?;
            ctx.state[*slot] = val;
            Ok(ObservationResult::Continue)
        }
        Statement::Skip => Ok(ObservationResult::Skipped),
    }
}

#[inline]
fn eval_expr(expr: &Expr, ctx: &mut EvalContext<'_>) -> Result<Value, DslError> {
    match expr {
        Expr::Literal(v) => Ok(*v),
        Expr::LocalVar(slot) => Ok(ctx.locals[*slot]),
        Expr::StateVar(slot) => Ok(ctx.state[*slot]),
        Expr::Notional => Ok(Value::F64(ctx.notional)),
        Expr::ObservationDate => Ok(Value::F64(ctx.observation_date)),
        Expr::IsFinal => Ok(Value::Bool(ctx.is_final)),
        Expr::BinOp { op, lhs, rhs } => {
            let l = eval_expr(lhs, ctx)?;
            let r = eval_expr(rhs, ctx)?;
            Ok(eval_binop(*op, l, r))
        }
        Expr::UnaryOp { op, operand } => {
            let v = eval_expr(operand, ctx)?;
            Ok(eval_unaryop(*op, v))
        }
        Expr::Call { func, args } => eval_builtin(*func, args, ctx),
    }
}

#[inline]
fn eval_binop(op: BinOp, lhs: Value, rhs: Value) -> Value {
    let l = lhs.as_f64();
    let r = rhs.as_f64();
    match op {
        BinOp::Add => Value::F64(l + r),
        BinOp::Sub => Value::F64(l - r),
        BinOp::Mul => Value::F64(l * r),
        BinOp::Div => Value::F64(if r == 0.0 { f64::NAN } else { l / r }),
        BinOp::Eq => Value::Bool((l - r).abs() < f64::EPSILON),
        BinOp::Ne => Value::Bool((l - r).abs() >= f64::EPSILON),
        BinOp::Lt => Value::Bool(l < r),
        BinOp::Le => Value::Bool(l <= r),
        BinOp::Gt => Value::Bool(l > r),
        BinOp::Ge => Value::Bool(l >= r),
        BinOp::And => Value::Bool(lhs.as_bool() && rhs.as_bool()),
        BinOp::Or => Value::Bool(lhs.as_bool() || rhs.as_bool()),
    }
}

#[inline]
fn eval_unaryop(op: UnaryOp, val: Value) -> Value {
    match op {
        UnaryOp::Neg => Value::F64(-val.as_f64()),
        UnaryOp::Not => Value::Bool(!val.as_bool()),
    }
}

fn eval_builtin(
    func: BuiltinFn,
    args: &[Expr],
    ctx: &mut EvalContext<'_>,
) -> Result<Value, DslError> {
    match func {
        BuiltinFn::Performances => {
            // Returns a pseudo-value; only meaningful as argument to worst_of/best_of.
            // We encode the performance vector as the first performance value.
            // This is handled specially by worst_of/best_of.
            Err(DslError::EvalError(
                "performances() cannot be used standalone; wrap in worst_of() or best_of()"
                    .to_string(),
            ))
        }
        BuiltinFn::WorstOf => {
            // If the argument is performances(), compute worst-of performance.
            if args.len() == 1
                && matches!(
                    &args[0],
                    Expr::Call {
                        func: BuiltinFn::Performances,
                        ..
                    }
                )
            {
                let wof = compute_worst_of_performance(ctx);
                return Ok(Value::F64(wof));
            }
            // Otherwise, evaluate all args and take min.
            let mut min_val = f64::INFINITY;
            for arg in args {
                let v = eval_expr(arg, ctx)?.as_f64();
                if v < min_val {
                    min_val = v;
                }
            }
            Ok(Value::F64(min_val))
        }
        BuiltinFn::BestOf => {
            if args.len() == 1
                && matches!(
                    &args[0],
                    Expr::Call {
                        func: BuiltinFn::Performances,
                        ..
                    }
                )
            {
                let bof = compute_best_of_performance(ctx);
                return Ok(Value::F64(bof));
            }
            let mut max_val = f64::NEG_INFINITY;
            for arg in args {
                let v = eval_expr(arg, ctx)?.as_f64();
                if v > max_val {
                    max_val = v;
                }
            }
            Ok(Value::F64(max_val))
        }
        BuiltinFn::Price => {
            if args.is_empty() {
                return Err(DslError::EvalError("price() requires an asset index".to_string()));
            }
            let idx = eval_expr(&args[0], ctx)?.as_f64() as usize;
            if idx >= ctx.spots.len() {
                return Err(DslError::EvalError(format!(
                    "asset index {idx} out of range (have {} assets)",
                    ctx.spots.len()
                )));
            }
            Ok(Value::F64(ctx.spots[idx]))
        }
        BuiltinFn::Min => {
            if args.len() != 2 {
                return Err(DslError::EvalError("min() requires 2 arguments".to_string()));
            }
            let a = eval_expr(&args[0], ctx)?.as_f64();
            let b = eval_expr(&args[1], ctx)?.as_f64();
            Ok(Value::F64(a.min(b)))
        }
        BuiltinFn::Max => {
            if args.len() != 2 {
                return Err(DslError::EvalError("max() requires 2 arguments".to_string()));
            }
            let a = eval_expr(&args[0], ctx)?.as_f64();
            let b = eval_expr(&args[1], ctx)?.as_f64();
            Ok(Value::F64(a.max(b)))
        }
        BuiltinFn::Abs => {
            if args.len() != 1 {
                return Err(DslError::EvalError("abs() requires 1 argument".to_string()));
            }
            let v = eval_expr(&args[0], ctx)?.as_f64();
            Ok(Value::F64(v.abs()))
        }
        BuiltinFn::Exp => {
            if args.len() != 1 {
                return Err(DslError::EvalError("exp() requires 1 argument".to_string()));
            }
            let v = eval_expr(&args[0], ctx)?.as_f64();
            Ok(Value::F64(v.exp()))
        }
        BuiltinFn::Log => {
            if args.len() != 1 {
                return Err(DslError::EvalError("log() requires 1 argument".to_string()));
            }
            let v = eval_expr(&args[0], ctx)?.as_f64();
            Ok(Value::F64(v.ln()))
        }
    }
}

#[inline]
fn compute_worst_of_performance(ctx: &EvalContext<'_>) -> f64 {
    let mut wof = f64::INFINITY;
    for (i, &spot) in ctx.spots.iter().enumerate() {
        if i < ctx.initial_spots.len() && ctx.initial_spots[i] > 0.0 {
            let perf = spot / ctx.initial_spots[i];
            if perf < wof {
                wof = perf;
            }
        }
    }
    wof
}

#[inline]
fn compute_best_of_performance(ctx: &EvalContext<'_>) -> f64 {
    let mut bof = f64::NEG_INFINITY;
    for (i, &spot) in ctx.spots.iter().enumerate() {
        if i < ctx.initial_spots.len() && ctx.initial_spots[i] > 0.0 {
            let perf = spot / ctx.initial_spots[i];
            if perf > bof {
                bof = perf;
            }
        }
    }
    bof
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::ir::*;

    /// Helper: build a simple product that pays notional * coupon_rate * obs_date
    /// if worst-of performance >= autocall_barrier, and redeems notional.
    fn make_simple_autocallable(
        notional: f64,
        maturity: f64,
        dates: Vec<f64>,
        autocall_barrier: f64,
        coupon_rate: f64,
        ki_barrier: f64,
    ) -> CompiledProduct {
        // State var 0: ki_hit (bool)
        let state_vars = vec![StateVarDef {
            name: "ki_hit".to_string(),
            slot: 0,
            initial: Value::Bool(false),
        }];

        // Schedule body:
        // let wof = worst_of(performances())       -- slot 0
        // if wof <= ki_barrier { set ki_hit = true }
        // if wof >= autocall_barrier and not is_final {
        //   pay notional * coupon_rate * observation_date
        //   redeem notional
        // }
        // if is_final {
        //   pay notional * coupon_rate * maturity
        //   if ki_hit and wof < 1.0 {
        //     redeem notional * wof
        //   } else {
        //     redeem notional
        //   }
        // }
        let wof_expr = Expr::Call {
            func: BuiltinFn::WorstOf,
            args: vec![Expr::Call {
                func: BuiltinFn::Performances,
                args: vec![],
            }],
        };

        let body = vec![
            // let wof = worst_of(performances())
            Statement::Let {
                slot: 0,
                expr: wof_expr,
            },
            // if wof <= ki_barrier { set ki_hit = true }
            Statement::If {
                condition: Expr::BinOp {
                    op: BinOp::Le,
                    lhs: Box::new(Expr::LocalVar(0)),
                    rhs: Box::new(Expr::Literal(Value::F64(ki_barrier))),
                },
                then_body: vec![Statement::SetState {
                    slot: 0,
                    expr: Expr::Literal(Value::Bool(true)),
                }],
                else_body: vec![],
            },
            // if wof >= autocall_barrier and not is_final { pay + redeem }
            Statement::If {
                condition: Expr::BinOp {
                    op: BinOp::And,
                    lhs: Box::new(Expr::BinOp {
                        op: BinOp::Ge,
                        lhs: Box::new(Expr::LocalVar(0)),
                        rhs: Box::new(Expr::Literal(Value::F64(autocall_barrier))),
                    }),
                    rhs: Box::new(Expr::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(Expr::IsFinal),
                    }),
                },
                then_body: vec![
                    Statement::Pay {
                        amount: Expr::BinOp {
                            op: BinOp::Mul,
                            lhs: Box::new(Expr::BinOp {
                                op: BinOp::Mul,
                                lhs: Box::new(Expr::Notional),
                                rhs: Box::new(Expr::Literal(Value::F64(coupon_rate))),
                            }),
                            rhs: Box::new(Expr::ObservationDate),
                        },
                    },
                    Statement::Redeem {
                        amount: Expr::Notional,
                    },
                ],
                else_body: vec![],
            },
            // if is_final { ... }
            Statement::If {
                condition: Expr::IsFinal,
                then_body: vec![
                    // pay coupon
                    Statement::Pay {
                        amount: Expr::BinOp {
                            op: BinOp::Mul,
                            lhs: Box::new(Expr::BinOp {
                                op: BinOp::Mul,
                                lhs: Box::new(Expr::Notional),
                                rhs: Box::new(Expr::Literal(Value::F64(coupon_rate))),
                            }),
                            rhs: Box::new(Expr::Literal(Value::F64(maturity))),
                        },
                    },
                    // if ki_hit and wof < 1.0 { redeem notional * wof } else { redeem notional }
                    Statement::If {
                        condition: Expr::BinOp {
                            op: BinOp::And,
                            lhs: Box::new(Expr::StateVar(0)),
                            rhs: Box::new(Expr::BinOp {
                                op: BinOp::Lt,
                                lhs: Box::new(Expr::LocalVar(0)),
                                rhs: Box::new(Expr::Literal(Value::F64(1.0))),
                            }),
                        },
                        then_body: vec![Statement::Redeem {
                            amount: Expr::BinOp {
                                op: BinOp::Mul,
                                lhs: Box::new(Expr::Notional),
                                rhs: Box::new(Expr::LocalVar(0)),
                            },
                        }],
                        else_body: vec![Statement::Redeem {
                            amount: Expr::Notional,
                        }],
                    },
                ],
                else_body: vec![],
            },
        ];

        CompiledProduct {
            name: "Test Autocallable".to_string(),
            notional,
            maturity,
            num_underlyings: 1,
            underlyings: vec![UnderlyingDef {
                name: "SPX".to_string(),
                asset_index: 0,
            }],
            state_vars,
            constants: vec![],
            schedules: vec![Schedule { dates, body }],
        }
    }

    #[test]
    fn autocallable_early_redemption_when_spot_above_barrier() {
        let product = make_simple_autocallable(
            1_000_000.0, // notional
            1.5,         // maturity
            vec![0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            1.0,  // autocall barrier
            0.08, // coupon rate
            0.60, // ki barrier
        );

        let initial_spots = vec![100.0];
        // All dates spot = 105 (above autocall barrier of 100%)
        // Should autocall at first date (0.25)
        let num_steps = 6;
        let path_spots: Vec<Vec<f64>> = (0..=num_steps).map(|_| vec![105.0]).collect();

        let pv = evaluate_product(&product, &path_spots, &initial_spots, num_steps, 0.05).unwrap();

        // At t=0.25: pays coupon = 1M * 0.08 * 0.25 = 20000, redeems 1M
        // PV = (20000 + 1000000) * exp(-0.05 * 0.25)
        let expected = (20_000.0 + 1_000_000.0) * (-0.05 * 0.25f64).exp();
        assert!(
            (pv - expected).abs() < 1.0,
            "expected {expected}, got {pv}"
        );
    }

    #[test]
    fn autocallable_ki_hit_with_final_wof_below_one() {
        let product = make_simple_autocallable(
            1_000_000.0,
            1.5,
            vec![0.5, 1.0, 1.5],
            1.0,
            0.08,
            0.60,
        );

        let initial_spots = vec![100.0];
        let num_steps = 3;

        // t=0: 100, t=0.5: 55 (ki hit, below 60%), t=1.0: 70, t=1.5: 80
        // wof at each: 0.55, 0.70, 0.80 — all below autocall (1.0)
        // ki_hit = true at t=0.5
        // At maturity: wof=0.80 < 1.0 and ki_hit → redeem notional * wof
        let path_spots = vec![
            vec![100.0], // step 0
            vec![55.0],  // step 1 (t=0.5)
            vec![70.0],  // step 2 (t=1.0)
            vec![80.0],  // step 3 (t=1.5)
        ];

        let pv = evaluate_product(&product, &path_spots, &initial_spots, num_steps, 0.05).unwrap();

        // At maturity t=1.5:
        // coupon = 1M * 0.08 * 1.5 = 120000
        // redemption = 1M * 0.80 = 800000
        let coupon_pv = 120_000.0 * (-0.05 * 1.5f64).exp();
        let redeem_pv = 800_000.0 * (-0.05 * 1.5f64).exp();
        let expected = coupon_pv + redeem_pv;

        assert!(
            (pv - expected).abs() < 1.0,
            "expected {expected}, got {pv}"
        );
    }

    #[test]
    fn autocallable_no_ki_hit_returns_full_notional() {
        let product = make_simple_autocallable(
            1_000_000.0,
            1.0,
            vec![0.5, 1.0],
            1.0,
            0.08,
            0.60,
        );

        let initial_spots = vec![100.0];
        let num_steps = 2;

        // Spot stays at 90 (above ki barrier of 60, below autocall of 100%)
        let path_spots = vec![
            vec![100.0], // step 0
            vec![90.0],  // step 1 (t=0.5), perf = 0.90
            vec![90.0],  // step 2 (t=1.0), perf = 0.90
        ];

        let pv = evaluate_product(&product, &path_spots, &initial_spots, num_steps, 0.05).unwrap();

        // At maturity: wof=0.90 < 1.0 but ki_hit=false → redeem notional (full)
        // coupon = 1M * 0.08 * 1.0 = 80000
        let coupon_pv = 80_000.0 * (-0.05 * 1.0f64).exp();
        let redeem_pv = 1_000_000.0 * (-0.05 * 1.0f64).exp();
        let expected = coupon_pv + redeem_pv;

        assert!(
            (pv - expected).abs() < 1.0,
            "expected {expected}, got {pv}"
        );
    }

    #[test]
    fn multi_asset_worst_of_uses_min_performance() {
        // 3 assets, dates at [0.5, 1.0]
        let product = CompiledProduct {
            name: "WoF Test".to_string(),
            notional: 100.0,
            maturity: 1.0,
            num_underlyings: 3,
            underlyings: vec![
                UnderlyingDef { name: "A".to_string(), asset_index: 0 },
                UnderlyingDef { name: "B".to_string(), asset_index: 1 },
                UnderlyingDef { name: "C".to_string(), asset_index: 2 },
            ],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![
                    Statement::Let {
                        slot: 0,
                        expr: Expr::Call {
                            func: BuiltinFn::WorstOf,
                            args: vec![Expr::Call {
                                func: BuiltinFn::Performances,
                                args: vec![],
                            }],
                        },
                    },
                    // redeem notional * wof
                    Statement::Redeem {
                        amount: Expr::BinOp {
                            op: BinOp::Mul,
                            lhs: Box::new(Expr::Notional),
                            rhs: Box::new(Expr::LocalVar(0)),
                        },
                    },
                ],
            }],
        };

        let initial_spots = vec![100.0, 200.0, 50.0];
        let num_steps = 1;
        // At maturity: A=110 (1.10), B=180 (0.90), C=55 (1.10)
        // worst-of = 0.90
        let path_spots = vec![
            vec![100.0, 200.0, 50.0], // t=0
            vec![110.0, 180.0, 55.0], // t=1.0
        ];

        let pv = evaluate_product(&product, &path_spots, &initial_spots, num_steps, 0.0).unwrap();
        // redeem 100 * 0.90 = 90, rate=0 so no discounting
        assert!((pv - 90.0).abs() < 0.01, "expected 90.0, got {pv}");
    }
}
