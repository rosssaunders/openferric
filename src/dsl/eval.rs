//! Evaluator for compiled DSL products.
//!
//! Walks the IR statement tree for each observation date on each MC path,
//! accumulating discounted cashflows and handling early termination.

use crate::dsl::error::DslError;
use crate::dsl::ir::{BinOp, BuiltinFn, CompiledProduct, Expr, Statement, UnaryOp};

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
    /// Discount factor for the current observation date.
    discount_factor: f64,
    /// Local variable slots.
    locals: &'a mut [f64],
    /// State variable slots (mutable across observations).
    state: &'a mut [f64],
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

#[derive(Debug, Clone)]
pub(crate) struct ProductExecutionPlan {
    step_to_snapshot: Vec<usize>,
    schedules: Vec<ScheduleExecutionPlan>,
    snapshot_count: usize,
    max_stack: usize,
}

impl ProductExecutionPlan {
    #[inline]
    pub(crate) fn snapshot_count(&self) -> usize {
        self.snapshot_count
    }

    #[inline]
    pub(crate) fn snapshot_index_for_step(&self, step: usize) -> Option<usize> {
        let idx = *self.step_to_snapshot.get(step)?;
        (idx != usize::MAX).then_some(idx)
    }

    #[inline]
    pub(crate) fn max_stack(&self) -> usize {
        self.max_stack
    }
}

#[derive(Debug, Clone)]
struct ScheduleExecutionPlan {
    observations: Vec<ObservationPoint>,
    program: Vec<ExecOp>,
}

#[derive(Debug, Clone, Copy)]
struct ObservationPoint {
    snapshot_index: usize,
    observation_date: f64,
    discount_factor: f64,
    is_final: bool,
}

#[derive(Debug, Clone, Copy)]
enum ExecOp {
    PushLiteral(f64),
    PushLocal(usize),
    PushState(usize),
    PushNotional,
    PushObservationDate,
    PushIsFinal,
    ApplyBinOp(BinOp),
    ApplyUnaryOp(UnaryOp),
    WorstOfPerformances,
    BestOfPerformances,
    WorstOf(usize),
    BestOf(usize),
    Price,
    Min,
    Max,
    Abs,
    Exp,
    Log,
    StoreLocal(usize),
    StoreState(usize),
    Pay,
    Redeem,
    JumpIfFalse(usize),
    Jump(usize),
    Skip,
}

#[derive(Debug, Default)]
struct ProgramBuilder {
    program: Vec<ExecOp>,
    stack_depth: usize,
    max_stack: usize,
}

impl ProgramBuilder {
    fn emit(&mut self, op: ExecOp, stack_delta: isize) -> usize {
        let idx = self.program.len();
        self.program.push(op);
        self.adjust_stack(stack_delta);
        idx
    }

    fn patch_jump(&mut self, idx: usize, target: usize) {
        match &mut self.program[idx] {
            ExecOp::JumpIfFalse(dst) | ExecOp::Jump(dst) => *dst = target,
            _ => unreachable!("attempted to patch non-jump opcode"),
        }
    }

    fn adjust_stack(&mut self, delta: isize) {
        if delta < 0 {
            self.stack_depth -= (-delta) as usize;
        } else {
            self.stack_depth += delta as usize;
            self.max_stack = self.max_stack.max(self.stack_depth);
        }
    }
}

struct ValueStack<'a> {
    values: &'a mut [f64],
    len: usize,
}

impl<'a> ValueStack<'a> {
    fn new(values: &'a mut [f64]) -> Self {
        Self { values, len: 0 }
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn push(&mut self, value: f64) {
        debug_assert!(self.len < self.values.len());
        self.values[self.len] = value;
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) -> f64 {
        debug_assert!(self.len > 0);
        self.len -= 1;
        self.values[self.len]
    }
}

pub(crate) fn build_execution_plan(
    product: &CompiledProduct,
    num_steps: usize,
    rate: f64,
) -> Result<ProductExecutionPlan, DslError> {
    let maturity = product.maturity;
    let step_scale = if maturity > 0.0 {
        num_steps as f64 / maturity
    } else {
        0.0
    };
    let mut step_to_snapshot = vec![usize::MAX; num_steps + 1];
    let mut schedules = Vec::with_capacity(product.schedules.len());
    let mut snapshot_count = 0usize;
    let mut max_stack = 0usize;

    for schedule in &product.schedules {
        let num_dates = schedule.dates.len();
        let mut observations = Vec::with_capacity(num_dates);
        for (date_idx, &obs_date) in schedule.dates.iter().enumerate() {
            let mut step_idx = if maturity > 0.0 {
                (obs_date * step_scale).round() as usize
            } else {
                0
            };
            step_idx = step_idx.min(num_steps);

            let snapshot_index = if step_to_snapshot[step_idx] == usize::MAX {
                let idx = snapshot_count;
                step_to_snapshot[step_idx] = idx;
                snapshot_count += 1;
                idx
            } else {
                step_to_snapshot[step_idx]
            };

            observations.push(ObservationPoint {
                snapshot_index,
                observation_date: obs_date,
                discount_factor: (-rate * obs_date).exp(),
                is_final: date_idx + 1 == num_dates,
            });
        }

        let mut builder = ProgramBuilder::default();
        compile_statement_block(&schedule.body, &mut builder)?;
        debug_assert_eq!(builder.stack_depth, 0);
        max_stack = max_stack.max(builder.max_stack);
        schedules.push(ScheduleExecutionPlan {
            observations,
            program: builder.program,
        });
    }

    Ok(ProductExecutionPlan {
        step_to_snapshot,
        schedules,
        snapshot_count,
        max_stack,
    })
}

/// Evaluate a compiled product on a single MC path.
///
/// `path_spots[t][asset]` gives the spot price of each underlying at each
/// time step. The first index is the time step (0 = initial), the second
/// is the asset index.
///
/// Returns the present value generated on this path.
pub fn evaluate_product(
    product: &CompiledProduct,
    path_spots: &[Vec<f64>],
    initial_spots: &[f64],
    num_steps: usize,
    rate: f64,
) -> Result<f64, DslError> {
    let plan = build_execution_plan(product, num_steps, rate)?;
    let num_locals = product.max_local_slots();
    let mut locals = vec![0.0; num_locals];
    let mut state = vec![0.0; product.state_vars.len()];
    let mut stack = vec![0.0; plan.max_stack()];

    evaluate_product_in_place(
        product,
        &plan,
        path_spots,
        initial_spots,
        &mut locals,
        &mut state,
        &mut stack,
    )
}

pub(crate) fn evaluate_product_in_place(
    product: &CompiledProduct,
    plan: &ProductExecutionPlan,
    path_spots: &[Vec<f64>],
    initial_spots: &[f64],
    locals: &mut [f64],
    state: &mut [f64],
    stack: &mut [f64],
) -> Result<f64, DslError> {
    let mut observation_spots = vec![Vec::new(); plan.snapshot_count()];
    for (step_idx, spots) in path_spots.iter().enumerate() {
        if let Some(snapshot_index) = plan.snapshot_index_for_step(step_idx) {
            observation_spots[snapshot_index] = spots.clone();
        }
    }

    evaluate_product_with_plan_in_place(
        product,
        plan,
        &observation_spots,
        initial_spots,
        locals,
        state,
        stack,
    )
}

pub(crate) fn evaluate_product_with_plan_in_place(
    product: &CompiledProduct,
    plan: &ProductExecutionPlan,
    observation_spots: &[Vec<f64>],
    initial_spots: &[f64],
    locals: &mut [f64],
    state: &mut [f64],
    stack: &mut [f64],
) -> Result<f64, DslError> {
    if state.len() != product.state_vars.len() {
        return Err(DslError::EvalError(format!(
            "state scratch length {} does not match product state count {}",
            state.len(),
            product.state_vars.len()
        )));
    }
    if stack.len() < plan.max_stack() {
        return Err(DslError::EvalError(format!(
            "stack scratch length {} is smaller than required stack {}",
            stack.len(),
            plan.max_stack()
        )));
    }
    for (dst, sv) in state.iter_mut().zip(product.state_vars.iter()) {
        *dst = sv.initial.as_f64();
    }

    let mut pv = 0.0;
    let mut value_stack = ValueStack::new(stack);

    for schedule_plan in &plan.schedules {
        let outcome = execute_schedule(
            product,
            schedule_plan,
            observation_spots,
            initial_spots,
            locals,
            state,
            &mut value_stack,
            &mut pv,
        )?;
        if matches!(
            outcome,
            ObservationResult::Redeemed | ObservationResult::Skipped
        ) {
            break;
        }
    }

    Ok(pv)
}

fn execute_schedule(
    product: &CompiledProduct,
    plan: &ScheduleExecutionPlan,
    observation_spots: &[Vec<f64>],
    initial_spots: &[f64],
    locals: &mut [f64],
    state: &mut [f64],
    stack: &mut ValueStack<'_>,
    pv: &mut f64,
) -> Result<ObservationResult, DslError> {
    for observation in &plan.observations {
        let spots = observation_spots
            .get(observation.snapshot_index)
            .ok_or_else(|| {
                DslError::EvalError(format!(
                    "missing observation snapshot {}",
                    observation.snapshot_index
                ))
            })?;
        // Reset locals for each observation date.
        locals.fill(0.0);

        let mut ctx = EvalContext {
            spots,
            initial_spots,
            notional: product.notional,
            observation_date: observation.observation_date,
            is_final: observation.is_final,
            discount_factor: observation.discount_factor,
            locals,
            state,
        };

        let result = execute_program(&plan.program, &mut ctx, stack, pv)?;

        match result {
            ObservationResult::Redeemed | ObservationResult::Skipped => return Ok(result),
            ObservationResult::Continue => {}
        }
    }

    Ok(ObservationResult::Continue)
}

fn compile_statement_block(
    stmts: &[Statement],
    builder: &mut ProgramBuilder,
) -> Result<(), DslError> {
    for stmt in stmts {
        compile_statement(stmt, builder)?;
    }
    Ok(())
}

fn compile_statement(stmt: &Statement, builder: &mut ProgramBuilder) -> Result<(), DslError> {
    let entry_stack = builder.stack_depth;

    match stmt {
        Statement::Let { slot, expr } => {
            compile_expr(expr, builder)?;
            builder.emit(ExecOp::StoreLocal(*slot), -1);
        }
        Statement::If {
            condition,
            then_body,
            else_body,
        } => {
            compile_expr(condition, builder)?;
            let jump_if_false = builder.emit(ExecOp::JumpIfFalse(usize::MAX), -1);
            compile_statement_block(then_body, builder)?;
            if else_body.is_empty() {
                builder.patch_jump(jump_if_false, builder.program.len());
            } else {
                let jump_end = builder.emit(ExecOp::Jump(usize::MAX), 0);
                let else_start = builder.program.len();
                builder.patch_jump(jump_if_false, else_start);
                compile_statement_block(else_body, builder)?;
                builder.patch_jump(jump_end, builder.program.len());
            }
        }
        Statement::Pay { amount } => {
            compile_expr(amount, builder)?;
            builder.emit(ExecOp::Pay, -1);
        }
        Statement::Redeem { amount } => {
            compile_expr(amount, builder)?;
            builder.emit(ExecOp::Redeem, -1);
        }
        Statement::SetState { slot, expr } => {
            compile_expr(expr, builder)?;
            builder.emit(ExecOp::StoreState(*slot), -1);
        }
        Statement::Skip => {
            builder.emit(ExecOp::Skip, 0);
        }
    }

    debug_assert_eq!(builder.stack_depth, entry_stack);
    Ok(())
}

fn compile_expr(expr: &Expr, builder: &mut ProgramBuilder) -> Result<(), DslError> {
    match expr {
        Expr::Literal(v) => {
            builder.emit(ExecOp::PushLiteral(v.as_f64()), 1);
        }
        Expr::LocalVar(slot) => {
            builder.emit(ExecOp::PushLocal(*slot), 1);
        }
        Expr::StateVar(slot) => {
            builder.emit(ExecOp::PushState(*slot), 1);
        }
        Expr::Notional => {
            builder.emit(ExecOp::PushNotional, 1);
        }
        Expr::ObservationDate => {
            builder.emit(ExecOp::PushObservationDate, 1);
        }
        Expr::IsFinal => {
            builder.emit(ExecOp::PushIsFinal, 1);
        }
        Expr::BinOp { op, lhs, rhs } => {
            compile_expr(lhs, builder)?;
            compile_expr(rhs, builder)?;
            builder.emit(ExecOp::ApplyBinOp(*op), -1);
        }
        Expr::UnaryOp { op, operand } => {
            compile_expr(operand, builder)?;
            builder.emit(ExecOp::ApplyUnaryOp(*op), 0);
        }
        Expr::Call { func, args } => compile_builtin(*func, args, builder)?,
    }

    Ok(())
}

fn compile_builtin(
    func: BuiltinFn,
    args: &[Expr],
    builder: &mut ProgramBuilder,
) -> Result<(), DslError> {
    match func {
        BuiltinFn::Performances => {
            return Err(DslError::EvalError(
                "performances() cannot be used standalone; wrap in worst_of() or best_of()"
                    .to_string(),
            ));
        }
        BuiltinFn::WorstOf => {
            if args.len() == 1
                && matches!(
                    &args[0],
                    Expr::Call {
                        func: BuiltinFn::Performances,
                        ..
                    }
                )
            {
                builder.emit(ExecOp::WorstOfPerformances, 1);
            } else {
                for arg in args {
                    compile_expr(arg, builder)?;
                }
                builder.emit(ExecOp::WorstOf(args.len()), 1 - args.len() as isize);
            }
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
                builder.emit(ExecOp::BestOfPerformances, 1);
            } else {
                for arg in args {
                    compile_expr(arg, builder)?;
                }
                builder.emit(ExecOp::BestOf(args.len()), 1 - args.len() as isize);
            }
        }
        BuiltinFn::Price => {
            if args.len() != 1 {
                return Err(DslError::EvalError(
                    "price() requires an asset index".to_string(),
                ));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(ExecOp::Price, 0);
        }
        BuiltinFn::Min => {
            if args.len() != 2 {
                return Err(DslError::EvalError(
                    "min() requires 2 arguments".to_string(),
                ));
            }
            compile_expr(&args[0], builder)?;
            compile_expr(&args[1], builder)?;
            builder.emit(ExecOp::Min, -1);
        }
        BuiltinFn::Max => {
            if args.len() != 2 {
                return Err(DslError::EvalError(
                    "max() requires 2 arguments".to_string(),
                ));
            }
            compile_expr(&args[0], builder)?;
            compile_expr(&args[1], builder)?;
            builder.emit(ExecOp::Max, -1);
        }
        BuiltinFn::Abs => {
            if args.len() != 1 {
                return Err(DslError::EvalError("abs() requires 1 argument".to_string()));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(ExecOp::Abs, 0);
        }
        BuiltinFn::Exp => {
            if args.len() != 1 {
                return Err(DslError::EvalError("exp() requires 1 argument".to_string()));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(ExecOp::Exp, 0);
        }
        BuiltinFn::Log => {
            if args.len() != 1 {
                return Err(DslError::EvalError("log() requires 1 argument".to_string()));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(ExecOp::Log, 0);
        }
    }

    Ok(())
}

fn execute_program(
    program: &[ExecOp],
    ctx: &mut EvalContext<'_>,
    stack: &mut ValueStack<'_>,
    pv: &mut f64,
) -> Result<ObservationResult, DslError> {
    stack.clear();
    let mut pc = 0usize;

    while pc < program.len() {
        match program[pc] {
            ExecOp::PushLiteral(value) => stack.push(value),
            ExecOp::PushLocal(slot) => stack.push(ctx.locals[slot]),
            ExecOp::PushState(slot) => stack.push(ctx.state[slot]),
            ExecOp::PushNotional => stack.push(ctx.notional),
            ExecOp::PushObservationDate => stack.push(ctx.observation_date),
            ExecOp::PushIsFinal => stack.push(bool_to_f64(ctx.is_final)),
            ExecOp::ApplyBinOp(op) => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(eval_binop(op, lhs, rhs));
            }
            ExecOp::ApplyUnaryOp(op) => {
                let value = stack.pop();
                stack.push(eval_unaryop(op, value));
            }
            ExecOp::WorstOfPerformances => {
                stack.push(compute_worst_of_performance(ctx));
            }
            ExecOp::BestOfPerformances => {
                stack.push(compute_best_of_performance(ctx));
            }
            ExecOp::WorstOf(arg_count) => {
                let mut min_val = f64::INFINITY;
                for _ in 0..arg_count {
                    let value = stack.pop();
                    if value < min_val {
                        min_val = value;
                    }
                }
                stack.push(min_val);
            }
            ExecOp::BestOf(arg_count) => {
                let mut max_val = f64::NEG_INFINITY;
                for _ in 0..arg_count {
                    let value = stack.pop();
                    if value > max_val {
                        max_val = value;
                    }
                }
                stack.push(max_val);
            }
            ExecOp::Price => {
                let idx = stack.pop() as usize;
                if idx >= ctx.spots.len() {
                    return Err(DslError::EvalError(format!(
                        "asset index {idx} out of range (have {} assets)",
                        ctx.spots.len()
                    )));
                }
                stack.push(ctx.spots[idx]);
            }
            ExecOp::Min => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs.min(rhs));
            }
            ExecOp::Max => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs.max(rhs));
            }
            ExecOp::Abs => {
                let value = stack.pop();
                stack.push(value.abs());
            }
            ExecOp::Exp => {
                let value = stack.pop();
                stack.push(value.exp());
            }
            ExecOp::Log => {
                let value = stack.pop();
                stack.push(value.ln());
            }
            ExecOp::StoreLocal(slot) => {
                ctx.locals[slot] = stack.pop();
            }
            ExecOp::StoreState(slot) => {
                ctx.state[slot] = stack.pop();
            }
            ExecOp::Pay => {
                let value = stack.pop();
                *pv += value * ctx.discount_factor;
            }
            ExecOp::Redeem => {
                let value = stack.pop();
                *pv += value * ctx.discount_factor;
                return Ok(ObservationResult::Redeemed);
            }
            ExecOp::JumpIfFalse(target) => {
                if stack.pop() == 0.0 {
                    pc = target;
                    continue;
                }
            }
            ExecOp::Jump(target) => {
                pc = target;
                continue;
            }
            ExecOp::Skip => return Ok(ObservationResult::Skipped),
        }

        pc += 1;
    }

    debug_assert_eq!(stack.len, 0);
    Ok(ObservationResult::Continue)
}

#[inline]
fn eval_binop(op: BinOp, lhs: f64, rhs: f64) -> f64 {
    match op {
        BinOp::Add => lhs + rhs,
        BinOp::Sub => lhs - rhs,
        BinOp::Mul => lhs * rhs,
        BinOp::Div => {
            if rhs == 0.0 {
                f64::NAN
            } else {
                lhs / rhs
            }
        }
        BinOp::Eq => bool_to_f64((lhs - rhs).abs() < f64::EPSILON),
        BinOp::Ne => bool_to_f64((lhs - rhs).abs() >= f64::EPSILON),
        BinOp::Lt => bool_to_f64(lhs < rhs),
        BinOp::Le => bool_to_f64(lhs <= rhs),
        BinOp::Gt => bool_to_f64(lhs > rhs),
        BinOp::Ge => bool_to_f64(lhs >= rhs),
        BinOp::And => bool_to_f64(lhs != 0.0 && rhs != 0.0),
        BinOp::Or => bool_to_f64(lhs != 0.0 || rhs != 0.0),
    }
}

#[inline]
fn eval_unaryop(op: UnaryOp, val: f64) -> f64 {
    match op {
        UnaryOp::Neg => -val,
        UnaryOp::Not => bool_to_f64(val == 0.0),
    }
}

#[inline]
fn bool_to_f64(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
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
                underlying_type: Default::default(),
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
        assert!((pv - expected).abs() < 1.0, "expected {expected}, got {pv}");
    }

    #[test]
    fn autocallable_ki_hit_with_final_wof_below_one() {
        let product =
            make_simple_autocallable(1_000_000.0, 1.5, vec![0.5, 1.0, 1.5], 1.0, 0.08, 0.60);

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

        assert!((pv - expected).abs() < 1.0, "expected {expected}, got {pv}");
    }

    #[test]
    fn autocallable_no_ki_hit_returns_full_notional() {
        let product = make_simple_autocallable(1_000_000.0, 1.0, vec![0.5, 1.0], 1.0, 0.08, 0.60);

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

        assert!((pv - expected).abs() < 1.0, "expected {expected}, got {pv}");
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
                UnderlyingDef {
                    name: "A".to_string(),
                    asset_index: 0,
                    underlying_type: Default::default(),
                },
                UnderlyingDef {
                    name: "B".to_string(),
                    asset_index: 1,
                    underlying_type: Default::default(),
                },
                UnderlyingDef {
                    name: "C".to_string(),
                    asset_index: 2,
                    underlying_type: Default::default(),
                },
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
