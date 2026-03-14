//! Evaluator for compiled DSL products.
//!
//! Walks the IR statement tree for each observation date on each MC path,
//! accumulating discounted cashflows and handling early termination.
//!
//! Bytecode is encoded as 4-byte packed [`Instruction`] values with a
//! separate constant pool for f64 literals.

use crate::dsl::error::DslError;
use crate::dsl::ir::{BinOp, BuiltinFn, CompiledProduct, Expr, Statement, UnaryOp, Value};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::math::simd_math::{fast_exp_f64x4, ln_f64x4, load_f64x4, store_f64x4};
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::math::simd_neon::{load_f64x2, simd_exp_f64x2, simd_ln_f64x2, store_f64x2};

// ── Bool-as-f64 convention ────────────────────────────────────────
const TRUE_F64: f64 = 1.0;
const FALSE_F64: f64 = 0.0;

#[inline]
fn bool_to_f64(b: bool) -> f64 {
    if b { TRUE_F64 } else { FALSE_F64 }
}

#[inline]
fn f64_to_bool(v: f64) -> bool {
    v != 0.0
}

// ── Packed instruction format ──────────────────────────────────────

/// A single 4-byte packed VM instruction.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Instruction {
    opcode: u8,
    flags: u8,
    operand: u16,
}

const _: () = assert!(std::mem::size_of::<Instruction>() == 4);

/// Compiled bytecode program with an associated constant pool.
#[derive(Debug, Clone)]
struct Program {
    code: Vec<Instruction>,
    constants: Vec<f64>,
}

/// Opcode constants organised by family.
mod opcode {
    // 0x00–0x0F: Control
    pub const JUMP: u8 = 0x01;
    pub const JUMP_FALSE: u8 = 0x02;
    pub const SKIP: u8 = 0x03;

    // 0x10–0x1F: Load
    pub const PUSH_CONST: u8 = 0x10;
    pub const PUSH_LOCAL: u8 = 0x11;
    pub const PUSH_STATE: u8 = 0x12;
    pub const PUSH_NOTIONAL: u8 = 0x13;
    pub const PUSH_DATE: u8 = 0x14;
    pub const PUSH_IS_FINAL: u8 = 0x15;
    pub const PUSH_TRUE: u8 = 0x16;
    pub const PUSH_FALSE: u8 = 0x17;

    // 0x20–0x2F: Store
    pub const STORE_LOCAL: u8 = 0x20;
    pub const STORE_STATE: u8 = 0x21;

    // 0x30–0x3F: Arithmetic
    pub const ADD: u8 = 0x30;
    pub const SUB: u8 = 0x31;
    pub const MUL: u8 = 0x32;
    pub const DIV: u8 = 0x33;
    pub const NEG: u8 = 0x34;
    pub const ABS: u8 = 0x35;
    pub const EXP: u8 = 0x36;
    pub const LOG: u8 = 0x37;
    pub const MIN: u8 = 0x38;
    pub const MAX: u8 = 0x39;

    // 0x40–0x4F: Comparison / Logic
    pub const EQ: u8 = 0x40;
    pub const NE: u8 = 0x41;
    pub const LT: u8 = 0x42;
    pub const LE: u8 = 0x43;
    pub const GT: u8 = 0x44;
    pub const GE: u8 = 0x45;
    pub const AND: u8 = 0x46;
    pub const OR: u8 = 0x47;
    pub const NOT: u8 = 0x48;

    // 0x50–0x5F: Domain
    pub const PAY: u8 = 0x50;
    pub const REDEEM: u8 = 0x51;
    pub const PRICE: u8 = 0x52;
    pub const WORST_OF: u8 = 0x53;
    pub const BEST_OF: u8 = 0x54;
    pub const WORST_OF_PERF: u8 = 0x55;
    pub const BEST_OF_PERF: u8 = 0x56;
}

// ── Evaluation context ─────────────────────────────────────────────

/// Per-path evaluation context.
struct EvalContext<'a> {
    spots: &'a [f64],
    initial_spots: &'a [f64],
    notional: f64,
    observation_date: f64,
    is_final: bool,
    discount_factor: f64,
    locals: &'a mut [f64],
    state: &'a mut [f64],
}

/// Result of evaluating one observation date.
#[derive(Debug)]
enum ObservationResult {
    Continue,
    Redeemed,
    Skipped,
}

/// Cashflow generated during evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Cashflow {
    pub time: f64,
    pub amount: f64,
}

// ── Execution plan ─────────────────────────────────────────────────

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
    program: Program,
}

#[derive(Debug, Clone, Copy)]
struct ObservationPoint {
    snapshot_index: usize,
    observation_date: f64,
    discount_factor: f64,
    is_final: bool,
}

// ── Program builder ────────────────────────────────────────────────

#[derive(Debug, Default)]
struct ProgramBuilder {
    code: Vec<Instruction>,
    constants: Vec<f64>,
    stack_depth: usize,
    max_stack: usize,
}

impl ProgramBuilder {
    fn emit(&mut self, opcode: u8, operand: u16, stack_delta: isize) -> usize {
        let idx = self.code.len();
        self.code.push(Instruction {
            opcode,
            flags: 0,
            operand,
        });
        self.adjust_stack(stack_delta);
        idx
    }

    fn add_constant(&mut self, value: f64) -> u16 {
        let bits = value.to_bits();
        for (i, c) in self.constants.iter().enumerate() {
            if c.to_bits() == bits {
                return i as u16;
            }
        }
        let idx = self.constants.len();
        self.constants.push(value);
        idx as u16
    }

    fn patch_jump(&mut self, idx: usize, target: usize) {
        debug_assert!(
            self.code[idx].opcode == opcode::JUMP || self.code[idx].opcode == opcode::JUMP_FALSE,
            "attempted to patch non-jump opcode"
        );
        self.code[idx].operand = target as u16;
    }

    fn build(self) -> Program {
        Program {
            code: self.code,
            constants: self.constants,
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

// ── Value stack ────────────────────────────────────────────────────

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

// ── Plan construction ──────────────────────────────────────────────

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
            program: builder.build(),
        });
    }

    Ok(ProductExecutionPlan {
        step_to_snapshot,
        schedules,
        snapshot_count,
        max_stack,
    })
}

// ── Public evaluation entry points ─────────────────────────────────

pub fn evaluate_product(
    product: &CompiledProduct,
    path_spots: &[Vec<f64>],
    initial_spots: &[f64],
    num_steps: usize,
    rate: f64,
) -> Result<f64, DslError> {
    let plan = build_execution_plan(product, num_steps, rate)?;
    let num_locals = product.max_local_slots();
    let mut locals = vec![0.0_f64; num_locals];
    let mut state = vec![0.0_f64; product.state_vars.len()];
    let mut stack = vec![0.0_f64; plan.max_stack()];

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
    execute_schedule_from_observation(
        product,
        plan,
        observation_spots,
        initial_spots,
        locals,
        state,
        stack,
        pv,
        0,
    )
}

fn execute_schedule_from_observation(
    product: &CompiledProduct,
    plan: &ScheduleExecutionPlan,
    observation_spots: &[Vec<f64>],
    initial_spots: &[f64],
    locals: &mut [f64],
    state: &mut [f64],
    stack: &mut ValueStack<'_>,
    pv: &mut f64,
    start_observation: usize,
) -> Result<ObservationResult, DslError> {
    for observation in plan.observations.iter().skip(start_observation) {
        let spots = observation_spots
            .get(observation.snapshot_index)
            .ok_or_else(|| {
                DslError::EvalError(format!(
                    "missing observation snapshot {}",
                    observation.snapshot_index
                ))
            })?;
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

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn find_branch_merge(code: &[Instruction], false_target: usize) -> usize {
    if false_target > 0 && false_target <= code.len() {
        let candidate = code[false_target - 1];
        if candidate.opcode == opcode::JUMP {
            let merge = candidate.operand as usize;
            if merge >= false_target {
                return merge;
            }
        }
    }
    false_target
}

#[inline]
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn validate_batch_scratch<const LANES: usize>(
    product: &CompiledProduct,
    plan: &ProductExecutionPlan,
    locals_len: usize,
    state: &[[f64; LANES]],
    stack_len: usize,
) -> Result<(), DslError> {
    if locals_len < product.max_local_slots() {
        return Err(DslError::EvalError(format!(
            "locals scratch length {} is smaller than required locals {}",
            locals_len,
            product.max_local_slots()
        )));
    }
    if state.len() != product.state_vars.len() {
        return Err(DslError::EvalError(format!(
            "state scratch length {} does not match product state count {}",
            state.len(),
            product.state_vars.len()
        )));
    }
    if stack_len < plan.max_stack() {
        return Err(DslError::EvalError(format!(
            "stack scratch length {} is smaller than required stack {}",
            stack_len,
            plan.max_stack()
        )));
    }
    Ok(())
}

#[inline]
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn init_state_batch<const LANES: usize>(
    product: &CompiledProduct,
    state: &mut [[f64; LANES]],
) -> Result<(), DslError> {
    if state.len() != product.state_vars.len() {
        return Err(DslError::EvalError(format!(
            "state scratch length {} does not match product state count {}",
            state.len(),
            product.state_vars.len()
        )));
    }
    for (dst, sv) in state.iter_mut().zip(product.state_vars.iter()) {
        dst.fill(sv.initial.as_f64());
    }
    Ok(())
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const SIMD_BATCH_LANES_X86: usize = 4;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
struct BatchEvalContextX86<'a> {
    spots: &'a [[f64; SIMD_BATCH_LANES_X86]],
    initial_spots: &'a [f64],
    notional: f64,
    observation_date: f64,
    is_final: bool,
    discount_factor: f64,
    locals: &'a mut [[f64; SIMD_BATCH_LANES_X86]],
    state: &'a mut [[f64; SIMD_BATCH_LANES_X86]],
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
struct SimdValueStackX86<'a> {
    values: &'a mut [[f64; SIMD_BATCH_LANES_X86]],
    len: usize,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<'a> SimdValueStackX86<'a> {
    fn new(values: &'a mut [[f64; SIMD_BATCH_LANES_X86]]) -> Self {
        Self { values, len: 0 }
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    unsafe fn push_reg(&mut self, value: std::arch::x86_64::__m256d) {
        debug_assert!(self.len < self.values.len());
        unsafe { store_f64x4(&mut self.values[self.len], 0, value) };
        self.len += 1;
    }

    #[inline]
    fn push_array(&mut self, value: [f64; SIMD_BATCH_LANES_X86]) {
        debug_assert!(self.len < self.values.len());
        self.values[self.len] = value;
        self.len += 1;
    }

    #[inline]
    unsafe fn pop_reg(&mut self) -> std::arch::x86_64::__m256d {
        debug_assert!(self.len > 0);
        self.len -= 1;
        unsafe { load_f64x4(&self.values[self.len], 0) }
    }

    #[inline]
    fn pop_array(&mut self) -> [f64; SIMD_BATCH_LANES_X86] {
        debug_assert!(self.len > 0);
        self.len -= 1;
        self.values[self.len]
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn x86_splat(value: f64) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::_mm256_set1_pd;
    unsafe { _mm256_set1_pd(value) }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn x86_abs(value: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_andnot_pd, _mm256_set1_pd};
    unsafe { _mm256_andnot_pd(_mm256_set1_pd(-0.0), value) }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn x86_bool_mask(value: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_CMP_NEQ_OQ, _mm256_cmp_pd, _mm256_setzero_pd};
    unsafe { _mm256_cmp_pd(value, _mm256_setzero_pd(), _CMP_NEQ_OQ) }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn x86_mask_from_bits(bits: u8) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_castsi256_pd, _mm256_set_epi64x};
    let bit = |lane: usize| {
        if bits & (1 << lane) != 0 {
            -1_i64
        } else {
            0_i64
        }
    };
    unsafe { _mm256_castsi256_pd(_mm256_set_epi64x(bit(3), bit(2), bit(1), bit(0))) }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn masked_write_x86(
    dst: &mut [f64; SIMD_BATCH_LANES_X86],
    value: std::arch::x86_64::__m256d,
    active_mask: std::arch::x86_64::__m256d,
) {
    use std::arch::x86_64::_mm256_blendv_pd;
    let current = unsafe { load_f64x4(dst, 0) };
    let blended = unsafe { _mm256_blendv_pd(current, value, active_mask) };
    unsafe { store_f64x4(dst, 0, blended) };
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) fn evaluate_product_with_plan_batch_x86(
    product: &CompiledProduct,
    plan: &ProductExecutionPlan,
    observation_spots: &[Vec<[f64; SIMD_BATCH_LANES_X86]>],
    initial_spots: &[f64],
    locals: &mut [[f64; SIMD_BATCH_LANES_X86]],
    state: &mut [[f64; SIMD_BATCH_LANES_X86]],
    stack: &mut [[f64; SIMD_BATCH_LANES_X86]],
) -> Result<[f64; SIMD_BATCH_LANES_X86], DslError> {
    validate_batch_scratch(product, plan, locals.len(), state, stack.len())?;
    init_state_batch(product, state)?;

    let mut pv = [0.0; SIMD_BATCH_LANES_X86];
    let mut value_stack = SimdValueStackX86::new(stack);
    let mut live_bits = 0b1111_u8;

    for schedule_plan in &plan.schedules {
        for observation in &schedule_plan.observations {
            if live_bits == 0 {
                return Ok(pv);
            }

            for slot in locals.iter_mut() {
                slot.fill(0.0);
            }

            let spots = observation_spots
                .get(observation.snapshot_index)
                .ok_or_else(|| {
                    DslError::EvalError(format!(
                        "missing observation snapshot {}",
                        observation.snapshot_index
                    ))
                })?;

            let mut ctx = BatchEvalContextX86 {
                spots,
                initial_spots,
                notional: product.notional,
                observation_date: observation.observation_date,
                is_final: observation.is_final,
                discount_factor: observation.discount_factor,
                locals,
                state,
            };
            value_stack.clear();
            live_bits = unsafe {
                execute_program_batch_x86_range(
                    &schedule_plan.program,
                    &mut ctx,
                    &mut value_stack,
                    &mut pv,
                    live_bits,
                    0,
                    schedule_plan.program.code.len(),
                )
            }?;
            debug_assert_eq!(value_stack.len, 0);
        }
    }

    Ok(pv)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn execute_program_batch_x86_range(
    program: &Program,
    ctx: &mut BatchEvalContextX86<'_>,
    stack: &mut SimdValueStackX86<'_>,
    pv: &mut [f64; SIMD_BATCH_LANES_X86],
    mut active_bits: u8,
    start_pc: usize,
    end_pc: usize,
) -> Result<u8, DslError> {
    use std::arch::x86_64::*;

    let code = &program.code;
    let constants = &program.constants;
    let one = unsafe { x86_splat(TRUE_F64) };
    let zero = _mm256_setzero_pd();
    let nan = unsafe { x86_splat(f64::NAN) };
    let discount = unsafe { x86_splat(ctx.discount_factor) };
    let mut pc = start_pc;

    while pc < end_pc && active_bits != 0 {
        let active_mask = unsafe { x86_mask_from_bits(active_bits) };
        let inst = code[pc];
        match inst.opcode {
            opcode::PUSH_CONST => unsafe {
                stack.push_reg(x86_splat(constants[inst.operand as usize]))
            },
            opcode::PUSH_TRUE => unsafe { stack.push_reg(one) },
            opcode::PUSH_FALSE => unsafe { stack.push_reg(zero) },
            opcode::PUSH_LOCAL => unsafe {
                stack.push_reg(load_f64x4(&ctx.locals[inst.operand as usize], 0))
            },
            opcode::PUSH_STATE => unsafe {
                stack.push_reg(load_f64x4(&ctx.state[inst.operand as usize], 0))
            },
            opcode::PUSH_NOTIONAL => unsafe { stack.push_reg(x86_splat(ctx.notional)) },
            opcode::PUSH_DATE => unsafe { stack.push_reg(x86_splat(ctx.observation_date)) },
            opcode::PUSH_IS_FINAL => unsafe {
                stack.push_reg(x86_splat(bool_to_f64(ctx.is_final)))
            },

            opcode::ADD => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_add_pd(lhs, rhs));
            },
            opcode::SUB => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_sub_pd(lhs, rhs));
            },
            opcode::MUL => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_mul_pd(lhs, rhs));
            },
            opcode::DIV => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let quotient = _mm256_div_pd(lhs, rhs);
                let zero_mask = _mm256_cmp_pd(rhs, zero, _CMP_EQ_OQ);
                stack.push_reg(_mm256_blendv_pd(quotient, nan, zero_mask));
            },
            opcode::NEG => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(_mm256_sub_pd(zero, value));
            },
            opcode::ABS => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(x86_abs(value));
            },
            opcode::EXP => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(fast_exp_f64x4(value));
            },
            opcode::LOG => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(ln_f64x4(value));
            },
            opcode::MIN => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_min_pd(lhs, rhs));
            },
            opcode::MAX => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_max_pd(lhs, rhs));
            },

            opcode::EQ => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let diff = x86_abs(_mm256_sub_pd(lhs, rhs));
                let mask = _mm256_cmp_pd(diff, x86_splat(f64::EPSILON), _CMP_LT_OQ);
                stack.push_reg(_mm256_and_pd(mask, one));
            },
            opcode::NE => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let diff = x86_abs(_mm256_sub_pd(lhs, rhs));
                let mask = _mm256_cmp_pd(diff, x86_splat(f64::EPSILON), _CMP_GE_OQ);
                stack.push_reg(_mm256_and_pd(mask, one));
            },
            opcode::LT => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_and_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ), one));
            },
            opcode::LE => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_and_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ), one));
            },
            opcode::GT => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_and_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GT_OQ), one));
            },
            opcode::GE => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(_mm256_and_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GE_OQ), one));
            },
            opcode::AND => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let mask = _mm256_and_pd(x86_bool_mask(lhs), x86_bool_mask(rhs));
                stack.push_reg(_mm256_and_pd(mask, one));
            },
            opcode::OR => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let mask = _mm256_or_pd(x86_bool_mask(lhs), x86_bool_mask(rhs));
                stack.push_reg(_mm256_and_pd(mask, one));
            },
            opcode::NOT => unsafe {
                let value = stack.pop_reg();
                let mask = _mm256_cmp_pd(value, zero, _CMP_EQ_OQ);
                stack.push_reg(_mm256_and_pd(mask, one));
            },

            opcode::WORST_OF_PERF => unsafe {
                stack.push_reg(compute_worst_of_performance_batch_x86(
                    ctx.spots,
                    ctx.initial_spots,
                ));
            },
            opcode::BEST_OF_PERF => unsafe {
                stack.push_reg(compute_best_of_performance_batch_x86(
                    ctx.spots,
                    ctx.initial_spots,
                ));
            },
            opcode::WORST_OF => unsafe {
                let arg_count = inst.operand as usize;
                let mut min_value = x86_splat(f64::INFINITY);
                for _ in 0..arg_count {
                    min_value = _mm256_min_pd(min_value, stack.pop_reg());
                }
                stack.push_reg(min_value);
            },
            opcode::BEST_OF => unsafe {
                let arg_count = inst.operand as usize;
                let mut max_value = x86_splat(f64::NEG_INFINITY);
                for _ in 0..arg_count {
                    max_value = _mm256_max_pd(max_value, stack.pop_reg());
                }
                stack.push_reg(max_value);
            },
            opcode::PRICE => {
                let idxs = stack.pop_array();
                let mut values = [0.0; SIMD_BATCH_LANES_X86];
                for lane in 0..SIMD_BATCH_LANES_X86 {
                    let idx = idxs[lane] as usize;
                    if idx >= ctx.spots.len() {
                        return Err(DslError::EvalError(format!(
                            "asset index {idx} out of range (have {} assets)",
                            ctx.spots.len()
                        )));
                    }
                    values[lane] = ctx.spots[idx][lane];
                }
                stack.push_array(values);
            }

            opcode::STORE_LOCAL => unsafe {
                let value = stack.pop_reg();
                masked_write_x86(&mut ctx.locals[inst.operand as usize], value, active_mask);
            },
            opcode::STORE_STATE => unsafe {
                let value = stack.pop_reg();
                masked_write_x86(&mut ctx.state[inst.operand as usize], value, active_mask);
            },

            opcode::PAY => unsafe {
                let value = stack.pop_reg();
                let add = _mm256_and_pd(_mm256_mul_pd(value, discount), active_mask);
                let pv_reg = _mm256_add_pd(load_f64x4(pv, 0), add);
                store_f64x4(pv, 0, pv_reg);
            },
            opcode::REDEEM => unsafe {
                let value = stack.pop_reg();
                let add = _mm256_and_pd(_mm256_mul_pd(value, discount), active_mask);
                let pv_reg = _mm256_add_pd(load_f64x4(pv, 0), add);
                store_f64x4(pv, 0, pv_reg);
                active_bits = 0;
                break;
            },

            opcode::JUMP_FALSE => unsafe {
                let cond = stack.pop_reg();
                let false_bits = (_mm256_movemask_pd(_mm256_and_pd(
                    _mm256_cmp_pd(cond, zero, _CMP_EQ_OQ),
                    active_mask,
                )) as u8)
                    & active_bits;
                if false_bits == active_bits {
                    pc = inst.operand as usize;
                    continue;
                }
                if false_bits != 0 {
                    let true_bits = active_bits & !false_bits;
                    let merge_pc = find_branch_merge(code, inst.operand as usize);
                    let saved_len = stack.len;
                    let true_remaining = execute_program_batch_x86_range(
                        program,
                        ctx,
                        stack,
                        pv,
                        true_bits,
                        pc + 1,
                        merge_pc,
                    )?;
                    debug_assert_eq!(stack.len, saved_len);
                    let false_remaining = execute_program_batch_x86_range(
                        program,
                        ctx,
                        stack,
                        pv,
                        false_bits,
                        inst.operand as usize,
                        merge_pc,
                    )?;
                    debug_assert_eq!(stack.len, saved_len);
                    active_bits = true_remaining | false_remaining;
                    pc = merge_pc;
                    continue;
                }
            },
            opcode::JUMP => {
                pc = inst.operand as usize;
                continue;
            }
            opcode::SKIP => {
                active_bits = 0;
                break;
            }

            _ => {
                return Err(DslError::EvalError(format!(
                    "unknown opcode 0x{:02x}",
                    inst.opcode
                )));
            }
        }

        pc += 1;
    }

    Ok(active_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_worst_of_performance_batch_x86(
    spots: &[[f64; SIMD_BATCH_LANES_X86]],
    initial_spots: &[f64],
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;

    let mut wof = unsafe { x86_splat(f64::INFINITY) };
    for (asset, lane_spots) in spots.iter().enumerate() {
        if asset < initial_spots.len() && initial_spots[asset] > 0.0 {
            let spot = unsafe { load_f64x4(lane_spots, 0) };
            let inv_initial = unsafe { x86_splat(1.0 / initial_spots[asset]) };
            let perf = _mm256_mul_pd(spot, inv_initial);
            wof = _mm256_min_pd(wof, perf);
        }
    }
    wof
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_best_of_performance_batch_x86(
    spots: &[[f64; SIMD_BATCH_LANES_X86]],
    initial_spots: &[f64],
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;

    let mut bof = unsafe { x86_splat(f64::NEG_INFINITY) };
    for (asset, lane_spots) in spots.iter().enumerate() {
        if asset < initial_spots.len() && initial_spots[asset] > 0.0 {
            let spot = unsafe { load_f64x4(lane_spots, 0) };
            let inv_initial = unsafe { x86_splat(1.0 / initial_spots[asset]) };
            let perf = _mm256_mul_pd(spot, inv_initial);
            bof = _mm256_max_pd(bof, perf);
        }
    }
    bof
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
const SIMD_BATCH_LANES_NEON: usize = 2;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
struct BatchEvalContextNeon<'a> {
    spots: &'a [[f64; SIMD_BATCH_LANES_NEON]],
    initial_spots: &'a [f64],
    notional: f64,
    observation_date: f64,
    is_final: bool,
    discount_factor: f64,
    locals: &'a mut [[f64; SIMD_BATCH_LANES_NEON]],
    state: &'a mut [[f64; SIMD_BATCH_LANES_NEON]],
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
struct SimdValueStackNeon<'a> {
    values: &'a mut [[f64; SIMD_BATCH_LANES_NEON]],
    len: usize,
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
impl<'a> SimdValueStackNeon<'a> {
    fn new(values: &'a mut [[f64; SIMD_BATCH_LANES_NEON]]) -> Self {
        Self { values, len: 0 }
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    unsafe fn push_reg(&mut self, value: std::arch::aarch64::float64x2_t) {
        debug_assert!(self.len < self.values.len());
        unsafe { store_f64x2(&mut self.values[self.len], 0, value) };
        self.len += 1;
    }

    #[inline]
    fn push_array(&mut self, value: [f64; SIMD_BATCH_LANES_NEON]) {
        debug_assert!(self.len < self.values.len());
        self.values[self.len] = value;
        self.len += 1;
    }

    #[inline]
    unsafe fn pop_reg(&mut self) -> std::arch::aarch64::float64x2_t {
        debug_assert!(self.len > 0);
        self.len -= 1;
        unsafe { load_f64x2(&self.values[self.len], 0) }
    }

    #[inline]
    fn pop_array(&mut self) -> [f64; SIMD_BATCH_LANES_NEON] {
        debug_assert!(self.len > 0);
        self.len -= 1;
        self.values[self.len]
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn neon_mask_bits(mask: std::arch::aarch64::uint64x2_t) -> u8 {
    use std::arch::aarch64::*;
    let mut lanes = [0_u64; SIMD_BATCH_LANES_NEON];
    unsafe { vst1q_u64(lanes.as_mut_ptr(), mask) };
    let mut bits = 0_u8;
    for (lane, value) in lanes.iter().enumerate() {
        if *value != 0 {
            bits |= 1 << lane;
        }
    }
    bits
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn neon_mask_from_bits(bits: u8) -> std::arch::aarch64::uint64x2_t {
    use std::arch::aarch64::*;
    let lanes = [
        if bits & 0b01 != 0 { u64::MAX } else { 0 },
        if bits & 0b10 != 0 { u64::MAX } else { 0 },
    ];
    unsafe { vld1q_u64(lanes.as_ptr()) }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn neon_bool_mask(value: std::arch::aarch64::float64x2_t) -> std::arch::aarch64::uint64x2_t {
    use std::arch::aarch64::*;
    unsafe { veorq_u64(vceqq_f64(value, vdupq_n_f64(0.0)), vdupq_n_u64(u64::MAX)) }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn masked_write_neon(
    dst: &mut [f64; SIMD_BATCH_LANES_NEON],
    value: std::arch::aarch64::float64x2_t,
    active_mask: std::arch::aarch64::uint64x2_t,
) {
    use std::arch::aarch64::*;
    let current = unsafe { load_f64x2(dst, 0) };
    let blended = unsafe { vbslq_f64(active_mask, value, current) };
    unsafe { store_f64x2(dst, 0, blended) };
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) fn evaluate_product_with_plan_batch_neon(
    product: &CompiledProduct,
    plan: &ProductExecutionPlan,
    observation_spots: &[Vec<[f64; SIMD_BATCH_LANES_NEON]>],
    initial_spots: &[f64],
    locals: &mut [[f64; SIMD_BATCH_LANES_NEON]],
    state: &mut [[f64; SIMD_BATCH_LANES_NEON]],
    stack: &mut [[f64; SIMD_BATCH_LANES_NEON]],
) -> Result<[f64; SIMD_BATCH_LANES_NEON], DslError> {
    validate_batch_scratch(product, plan, locals.len(), state, stack.len())?;
    init_state_batch(product, state)?;

    let mut pv = [0.0; SIMD_BATCH_LANES_NEON];
    let mut value_stack = SimdValueStackNeon::new(stack);
    let mut live_bits = 0b11_u8;

    for schedule_plan in &plan.schedules {
        for observation in &schedule_plan.observations {
            if live_bits == 0 {
                return Ok(pv);
            }

            for slot in locals.iter_mut() {
                slot.fill(0.0);
            }

            let spots = observation_spots
                .get(observation.snapshot_index)
                .ok_or_else(|| {
                    DslError::EvalError(format!(
                        "missing observation snapshot {}",
                        observation.snapshot_index
                    ))
                })?;

            let mut ctx = BatchEvalContextNeon {
                spots,
                initial_spots,
                notional: product.notional,
                observation_date: observation.observation_date,
                is_final: observation.is_final,
                discount_factor: observation.discount_factor,
                locals,
                state,
            };
            value_stack.clear();
            live_bits = unsafe {
                execute_program_batch_neon_range(
                    &schedule_plan.program,
                    &mut ctx,
                    &mut value_stack,
                    &mut pv,
                    live_bits,
                    0,
                    schedule_plan.program.code.len(),
                )
            }?;
            debug_assert_eq!(value_stack.len, 0);
        }
    }

    Ok(pv)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn execute_program_batch_neon_range(
    program: &Program,
    ctx: &mut BatchEvalContextNeon<'_>,
    stack: &mut SimdValueStackNeon<'_>,
    pv: &mut [f64; SIMD_BATCH_LANES_NEON],
    mut active_bits: u8,
    start_pc: usize,
    end_pc: usize,
) -> Result<u8, DslError> {
    use std::arch::aarch64::*;

    let code = &program.code;
    let constants = &program.constants;
    let one = vdupq_n_f64(TRUE_F64);
    let zero = vdupq_n_f64(0.0);
    let nan = vdupq_n_f64(f64::NAN);
    let discount = vdupq_n_f64(ctx.discount_factor);
    let mut pc = start_pc;

    while pc < end_pc && active_bits != 0 {
        let active_mask = unsafe { neon_mask_from_bits(active_bits) };
        let inst = code[pc];
        match inst.opcode {
            opcode::PUSH_CONST => unsafe {
                stack.push_reg(vdupq_n_f64(constants[inst.operand as usize]))
            },
            opcode::PUSH_TRUE => unsafe { stack.push_reg(one) },
            opcode::PUSH_FALSE => unsafe { stack.push_reg(zero) },
            opcode::PUSH_LOCAL => unsafe {
                stack.push_reg(load_f64x2(&ctx.locals[inst.operand as usize], 0))
            },
            opcode::PUSH_STATE => unsafe {
                stack.push_reg(load_f64x2(&ctx.state[inst.operand as usize], 0))
            },
            opcode::PUSH_NOTIONAL => unsafe { stack.push_reg(vdupq_n_f64(ctx.notional)) },
            opcode::PUSH_DATE => unsafe { stack.push_reg(vdupq_n_f64(ctx.observation_date)) },
            opcode::PUSH_IS_FINAL => unsafe {
                stack.push_reg(vdupq_n_f64(bool_to_f64(ctx.is_final)))
            },

            opcode::ADD => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vaddq_f64(lhs, rhs));
            },
            opcode::SUB => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vsubq_f64(lhs, rhs));
            },
            opcode::MUL => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vmulq_f64(lhs, rhs));
            },
            opcode::DIV => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let quotient = vdivq_f64(lhs, rhs);
                let zero_mask = vceqq_f64(rhs, zero);
                stack.push_reg(vbslq_f64(zero_mask, nan, quotient));
            },
            opcode::NEG => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(vnegq_f64(value));
            },
            opcode::ABS => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(vabsq_f64(value));
            },
            opcode::EXP => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(simd_exp_f64x2(value));
            },
            opcode::LOG => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(simd_ln_f64x2(value));
            },
            opcode::MIN => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vminq_f64(lhs, rhs));
            },
            opcode::MAX => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vmaxq_f64(lhs, rhs));
            },

            opcode::EQ => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let diff = vabsq_f64(vsubq_f64(lhs, rhs));
                let mask = vcltq_f64(diff, vdupq_n_f64(f64::EPSILON));
                stack.push_reg(vbslq_f64(mask, one, zero));
            },
            opcode::NE => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let diff = vabsq_f64(vsubq_f64(lhs, rhs));
                let mask = vcgeq_f64(diff, vdupq_n_f64(f64::EPSILON));
                stack.push_reg(vbslq_f64(mask, one, zero));
            },
            opcode::LT => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vbslq_f64(vcltq_f64(lhs, rhs), one, zero));
            },
            opcode::LE => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vbslq_f64(vcleq_f64(lhs, rhs), one, zero));
            },
            opcode::GT => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vbslq_f64(vcgtq_f64(lhs, rhs), one, zero));
            },
            opcode::GE => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                stack.push_reg(vbslq_f64(vcgeq_f64(lhs, rhs), one, zero));
            },
            opcode::AND => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let mask = vandq_u64(neon_bool_mask(lhs), neon_bool_mask(rhs));
                stack.push_reg(vbslq_f64(mask, one, zero));
            },
            opcode::OR => unsafe {
                let rhs = stack.pop_reg();
                let lhs = stack.pop_reg();
                let mask = vorrq_u64(neon_bool_mask(lhs), neon_bool_mask(rhs));
                stack.push_reg(vbslq_f64(mask, one, zero));
            },
            opcode::NOT => unsafe {
                let value = stack.pop_reg();
                stack.push_reg(vbslq_f64(vceqq_f64(value, zero), one, zero));
            },

            opcode::WORST_OF_PERF => unsafe {
                stack.push_reg(compute_worst_of_performance_batch_neon(
                    ctx.spots,
                    ctx.initial_spots,
                ));
            },
            opcode::BEST_OF_PERF => unsafe {
                stack.push_reg(compute_best_of_performance_batch_neon(
                    ctx.spots,
                    ctx.initial_spots,
                ));
            },
            opcode::WORST_OF => unsafe {
                let arg_count = inst.operand as usize;
                let mut min_value = vdupq_n_f64(f64::INFINITY);
                for _ in 0..arg_count {
                    min_value = vminq_f64(min_value, stack.pop_reg());
                }
                stack.push_reg(min_value);
            },
            opcode::BEST_OF => unsafe {
                let arg_count = inst.operand as usize;
                let mut max_value = vdupq_n_f64(f64::NEG_INFINITY);
                for _ in 0..arg_count {
                    max_value = vmaxq_f64(max_value, stack.pop_reg());
                }
                stack.push_reg(max_value);
            },
            opcode::PRICE => {
                let idxs = stack.pop_array();
                let mut values = [0.0; SIMD_BATCH_LANES_NEON];
                for lane in 0..SIMD_BATCH_LANES_NEON {
                    let idx = idxs[lane] as usize;
                    if idx >= ctx.spots.len() {
                        return Err(DslError::EvalError(format!(
                            "asset index {idx} out of range (have {} assets)",
                            ctx.spots.len()
                        )));
                    }
                    values[lane] = ctx.spots[idx][lane];
                }
                stack.push_array(values);
            }

            opcode::STORE_LOCAL => unsafe {
                let value = stack.pop_reg();
                masked_write_neon(&mut ctx.locals[inst.operand as usize], value, active_mask);
            },
            opcode::STORE_STATE => unsafe {
                let value = stack.pop_reg();
                masked_write_neon(&mut ctx.state[inst.operand as usize], value, active_mask);
            },

            opcode::PAY => unsafe {
                let value = stack.pop_reg();
                let add = vbslq_f64(active_mask, vmulq_f64(value, discount), zero);
                let pv_reg = vaddq_f64(load_f64x2(pv, 0), add);
                store_f64x2(pv, 0, pv_reg);
            },
            opcode::REDEEM => unsafe {
                let value = stack.pop_reg();
                let add = vbslq_f64(active_mask, vmulq_f64(value, discount), zero);
                let pv_reg = vaddq_f64(load_f64x2(pv, 0), add);
                store_f64x2(pv, 0, pv_reg);
                active_bits = 0;
                break;
            },

            opcode::JUMP_FALSE => unsafe {
                let cond = stack.pop_reg();
                let false_mask = vandq_u64(vceqq_f64(cond, zero), active_mask);
                let false_bits = neon_mask_bits(false_mask) & active_bits;
                if false_bits == active_bits {
                    pc = inst.operand as usize;
                    continue;
                }
                if false_bits != 0 {
                    let true_bits = active_bits & !false_bits;
                    let merge_pc = find_branch_merge(code, inst.operand as usize);
                    let saved_len = stack.len;
                    let true_remaining = execute_program_batch_neon_range(
                        program,
                        ctx,
                        stack,
                        pv,
                        true_bits,
                        pc + 1,
                        merge_pc,
                    )?;
                    debug_assert_eq!(stack.len, saved_len);
                    let false_remaining = execute_program_batch_neon_range(
                        program,
                        ctx,
                        stack,
                        pv,
                        false_bits,
                        inst.operand as usize,
                        merge_pc,
                    )?;
                    debug_assert_eq!(stack.len, saved_len);
                    active_bits = true_remaining | false_remaining;
                    pc = merge_pc;
                    continue;
                }
            },
            opcode::JUMP => {
                pc = inst.operand as usize;
                continue;
            }
            opcode::SKIP => {
                active_bits = 0;
                break;
            }

            _ => {
                return Err(DslError::EvalError(format!(
                    "unknown opcode 0x{:02x}",
                    inst.opcode
                )));
            }
        }

        pc += 1;
    }

    Ok(active_bits)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn compute_worst_of_performance_batch_neon(
    spots: &[[f64; SIMD_BATCH_LANES_NEON]],
    initial_spots: &[f64],
) -> std::arch::aarch64::float64x2_t {
    use std::arch::aarch64::*;

    let mut wof = vdupq_n_f64(f64::INFINITY);
    for (asset, lane_spots) in spots.iter().enumerate() {
        if asset < initial_spots.len() && initial_spots[asset] > 0.0 {
            let spot = unsafe { load_f64x2(lane_spots, 0) };
            let perf = vmulq_f64(spot, vdupq_n_f64(1.0 / initial_spots[asset]));
            wof = vminq_f64(wof, perf);
        }
    }
    wof
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn compute_best_of_performance_batch_neon(
    spots: &[[f64; SIMD_BATCH_LANES_NEON]],
    initial_spots: &[f64],
) -> std::arch::aarch64::float64x2_t {
    use std::arch::aarch64::*;

    let mut bof = vdupq_n_f64(f64::NEG_INFINITY);
    for (asset, lane_spots) in spots.iter().enumerate() {
        if asset < initial_spots.len() && initial_spots[asset] > 0.0 {
            let spot = unsafe { load_f64x2(lane_spots, 0) };
            let perf = vmulq_f64(spot, vdupq_n_f64(1.0 / initial_spots[asset]));
            bof = vmaxq_f64(bof, perf);
        }
    }
    bof
}

// ── Compilation: IR → packed bytecode ──────────────────────────────

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
            builder.emit(opcode::STORE_LOCAL, *slot as u16, -1);
        }
        Statement::If {
            condition,
            then_body,
            else_body,
        } => {
            compile_expr(condition, builder)?;
            let jump_if_false = builder.emit(opcode::JUMP_FALSE, u16::MAX, -1);
            compile_statement_block(then_body, builder)?;
            if else_body.is_empty() {
                builder.patch_jump(jump_if_false, builder.code.len());
            } else {
                let jump_end = builder.emit(opcode::JUMP, u16::MAX, 0);
                let else_start = builder.code.len();
                builder.patch_jump(jump_if_false, else_start);
                compile_statement_block(else_body, builder)?;
                builder.patch_jump(jump_end, builder.code.len());
            }
        }
        Statement::Pay { amount } => {
            compile_expr(amount, builder)?;
            builder.emit(opcode::PAY, 0, -1);
        }
        Statement::Redeem { amount } => {
            compile_expr(amount, builder)?;
            builder.emit(opcode::REDEEM, 0, -1);
        }
        Statement::SetState { slot, expr } => {
            compile_expr(expr, builder)?;
            builder.emit(opcode::STORE_STATE, *slot as u16, -1);
        }
        Statement::Skip => {
            builder.emit(opcode::SKIP, 0, 0);
        }
    }

    debug_assert_eq!(builder.stack_depth, entry_stack);
    Ok(())
}

fn compile_expr(expr: &Expr, builder: &mut ProgramBuilder) -> Result<(), DslError> {
    match expr {
        Expr::Literal(v) => match v {
            Value::F64(x) => {
                let idx = builder.add_constant(*x);
                builder.emit(opcode::PUSH_CONST, idx, 1);
            }
            Value::Bool(true) => {
                builder.emit(opcode::PUSH_TRUE, 0, 1);
            }
            Value::Bool(false) => {
                builder.emit(opcode::PUSH_FALSE, 0, 1);
            }
        },
        Expr::LocalVar(slot) => {
            builder.emit(opcode::PUSH_LOCAL, *slot as u16, 1);
        }
        Expr::StateVar(slot) => {
            builder.emit(opcode::PUSH_STATE, *slot as u16, 1);
        }
        Expr::Notional => {
            builder.emit(opcode::PUSH_NOTIONAL, 0, 1);
        }
        Expr::ObservationDate => {
            builder.emit(opcode::PUSH_DATE, 0, 1);
        }
        Expr::IsFinal => {
            builder.emit(opcode::PUSH_IS_FINAL, 0, 1);
        }
        Expr::BinOp { op, lhs, rhs } => {
            compile_expr(lhs, builder)?;
            compile_expr(rhs, builder)?;
            builder.emit(binop_opcode(*op), 0, -1);
        }
        Expr::UnaryOp { op, operand } => {
            compile_expr(operand, builder)?;
            builder.emit(unaryop_opcode(*op), 0, 0);
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
                builder.emit(opcode::WORST_OF_PERF, 0, 1);
            } else {
                for arg in args {
                    compile_expr(arg, builder)?;
                }
                builder.emit(opcode::WORST_OF, args.len() as u16, 1 - args.len() as isize);
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
                builder.emit(opcode::BEST_OF_PERF, 0, 1);
            } else {
                for arg in args {
                    compile_expr(arg, builder)?;
                }
                builder.emit(opcode::BEST_OF, args.len() as u16, 1 - args.len() as isize);
            }
        }
        BuiltinFn::Price => {
            if args.len() != 1 {
                return Err(DslError::EvalError(
                    "price() requires an asset index".to_string(),
                ));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(opcode::PRICE, 0, 0);
        }
        BuiltinFn::Min => {
            if args.len() != 2 {
                return Err(DslError::EvalError(
                    "min() requires 2 arguments".to_string(),
                ));
            }
            compile_expr(&args[0], builder)?;
            compile_expr(&args[1], builder)?;
            builder.emit(opcode::MIN, 0, -1);
        }
        BuiltinFn::Max => {
            if args.len() != 2 {
                return Err(DslError::EvalError(
                    "max() requires 2 arguments".to_string(),
                ));
            }
            compile_expr(&args[0], builder)?;
            compile_expr(&args[1], builder)?;
            builder.emit(opcode::MAX, 0, -1);
        }
        BuiltinFn::Abs => {
            if args.len() != 1 {
                return Err(DslError::EvalError("abs() requires 1 argument".to_string()));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(opcode::ABS, 0, 0);
        }
        BuiltinFn::Exp => {
            if args.len() != 1 {
                return Err(DslError::EvalError("exp() requires 1 argument".to_string()));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(opcode::EXP, 0, 0);
        }
        BuiltinFn::Log => {
            if args.len() != 1 {
                return Err(DslError::EvalError("log() requires 1 argument".to_string()));
            }
            compile_expr(&args[0], builder)?;
            builder.emit(opcode::LOG, 0, 0);
        }
    }

    Ok(())
}

#[inline]
fn binop_opcode(op: BinOp) -> u8 {
    match op {
        BinOp::Add => opcode::ADD,
        BinOp::Sub => opcode::SUB,
        BinOp::Mul => opcode::MUL,
        BinOp::Div => opcode::DIV,
        BinOp::Eq => opcode::EQ,
        BinOp::Ne => opcode::NE,
        BinOp::Lt => opcode::LT,
        BinOp::Le => opcode::LE,
        BinOp::Gt => opcode::GT,
        BinOp::Ge => opcode::GE,
        BinOp::And => opcode::AND,
        BinOp::Or => opcode::OR,
    }
}

#[inline]
fn unaryop_opcode(op: UnaryOp) -> u8 {
    match op {
        UnaryOp::Neg => opcode::NEG,
        UnaryOp::Not => opcode::NOT,
    }
}

// ── Bytecode interpreter ───────────────────────────────────────────

fn execute_program(
    program: &Program,
    ctx: &mut EvalContext<'_>,
    stack: &mut ValueStack<'_>,
    pv: &mut f64,
) -> Result<ObservationResult, DslError> {
    stack.clear();
    let code = &program.code;
    let constants = &program.constants;
    let mut pc = 0usize;

    while pc < code.len() {
        let inst = code[pc];
        match inst.opcode {
            // ── Load ───────────────────────────────────────────
            opcode::PUSH_CONST => stack.push(constants[inst.operand as usize]),
            opcode::PUSH_TRUE => stack.push(TRUE_F64),
            opcode::PUSH_FALSE => stack.push(FALSE_F64),
            opcode::PUSH_LOCAL => stack.push(ctx.locals[inst.operand as usize]),
            opcode::PUSH_STATE => stack.push(ctx.state[inst.operand as usize]),
            opcode::PUSH_NOTIONAL => stack.push(ctx.notional),
            opcode::PUSH_DATE => stack.push(ctx.observation_date),
            opcode::PUSH_IS_FINAL => stack.push(bool_to_f64(ctx.is_final)),

            // ── Arithmetic ─────────────────────────────────────
            opcode::ADD => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs + rhs);
            }
            opcode::SUB => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs - rhs);
            }
            opcode::MUL => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs * rhs);
            }
            opcode::DIV => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(if rhs == 0.0 { f64::NAN } else { lhs / rhs });
            }
            opcode::NEG => {
                let v = stack.pop();
                stack.push(-v);
            }
            opcode::ABS => {
                let v = stack.pop();
                stack.push(v.abs());
            }
            opcode::EXP => {
                let v = stack.pop();
                stack.push(v.exp());
            }
            opcode::LOG => {
                let v = stack.pop();
                stack.push(v.ln());
            }
            opcode::MIN => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs.min(rhs));
            }
            opcode::MAX => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(lhs.max(rhs));
            }

            // ── Comparison / Logic ─────────────────────────────
            opcode::EQ => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64((lhs - rhs).abs() < f64::EPSILON));
            }
            opcode::NE => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64((lhs - rhs).abs() >= f64::EPSILON));
            }
            opcode::LT => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64(lhs < rhs));
            }
            opcode::LE => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64(lhs <= rhs));
            }
            opcode::GT => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64(lhs > rhs));
            }
            opcode::GE => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64(lhs >= rhs));
            }
            opcode::AND => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64(f64_to_bool(lhs) && f64_to_bool(rhs)));
            }
            opcode::OR => {
                let rhs = stack.pop();
                let lhs = stack.pop();
                stack.push(bool_to_f64(f64_to_bool(lhs) || f64_to_bool(rhs)));
            }
            opcode::NOT => {
                let v = stack.pop();
                stack.push(bool_to_f64(!f64_to_bool(v)));
            }

            // ── Domain ─────────────────────────────────────────
            opcode::WORST_OF_PERF => {
                stack.push(compute_worst_of_performance(ctx));
            }
            opcode::BEST_OF_PERF => {
                stack.push(compute_best_of_performance(ctx));
            }
            opcode::WORST_OF => {
                let arg_count = inst.operand as usize;
                let mut min_val = f64::INFINITY;
                for _ in 0..arg_count {
                    let v = stack.pop();
                    if v < min_val {
                        min_val = v;
                    }
                }
                stack.push(min_val);
            }
            opcode::BEST_OF => {
                let arg_count = inst.operand as usize;
                let mut max_val = f64::NEG_INFINITY;
                for _ in 0..arg_count {
                    let v = stack.pop();
                    if v > max_val {
                        max_val = v;
                    }
                }
                stack.push(max_val);
            }
            opcode::PRICE => {
                let idx = stack.pop() as usize;
                if idx >= ctx.spots.len() {
                    return Err(DslError::EvalError(format!(
                        "asset index {idx} out of range (have {} assets)",
                        ctx.spots.len()
                    )));
                }
                stack.push(ctx.spots[idx]);
            }

            // ── Store ──────────────────────────────────────────
            opcode::STORE_LOCAL => {
                ctx.locals[inst.operand as usize] = stack.pop();
            }
            opcode::STORE_STATE => {
                ctx.state[inst.operand as usize] = stack.pop();
            }

            // ── Side-effects ───────────────────────────────────
            opcode::PAY => {
                let v = stack.pop();
                *pv += v * ctx.discount_factor;
            }
            opcode::REDEEM => {
                let v = stack.pop();
                *pv += v * ctx.discount_factor;
                return Ok(ObservationResult::Redeemed);
            }

            // ── Control flow ───────────────────────────────────
            opcode::JUMP_FALSE => {
                if stack.pop() == 0.0 {
                    pc = inst.operand as usize;
                    continue;
                }
            }
            opcode::JUMP => {
                pc = inst.operand as usize;
                continue;
            }
            opcode::SKIP => return Ok(ObservationResult::Skipped),

            _ => {
                return Err(DslError::EvalError(format!(
                    "unknown opcode 0x{:02x}",
                    inst.opcode
                )));
            }
        }

        pc += 1;
    }

    debug_assert_eq!(stack.len, 0);
    Ok(ObservationResult::Continue)
}

// ── Helpers ────────────────────────────────────────────────────────

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

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::ir::*;

    #[test]
    fn instruction_is_four_bytes() {
        assert_eq!(std::mem::size_of::<Instruction>(), 4);
    }

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

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn batched_x86_matches_scalar_when_lanes_diverge() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        let product = make_simple_autocallable(1_000_000.0, 1.0, vec![0.5, 1.0], 1.0, 0.08, 0.60);
        let plan = build_execution_plan(&product, 2, 0.05).unwrap();
        let initial_spots = vec![100.0];

        let lane_paths = [
            vec![vec![100.0], vec![105.0], vec![105.0]],
            vec![vec![100.0], vec![90.0], vec![90.0]],
            vec![vec![100.0], vec![55.0], vec![80.0]],
            vec![vec![100.0], vec![100.0], vec![100.0]],
        ];

        let mut observation_spots =
            vec![vec![[0.0; SIMD_BATCH_LANES_X86]; initial_spots.len()]; plan.snapshot_count()];
        for (step, _) in lane_paths[0].iter().enumerate() {
            if let Some(snapshot_index) = plan.snapshot_index_for_step(step) {
                for asset in 0..initial_spots.len() {
                    for lane in 0..SIMD_BATCH_LANES_X86 {
                        observation_spots[snapshot_index][asset][lane] =
                            lane_paths[lane][step][asset];
                    }
                }
            }
        }

        let mut locals = vec![[0.0; SIMD_BATCH_LANES_X86]; product.max_local_slots()];
        let mut state = vec![[0.0; SIMD_BATCH_LANES_X86]; product.state_vars.len()];
        let mut stack = vec![[0.0; SIMD_BATCH_LANES_X86]; plan.max_stack()];

        let batched = evaluate_product_with_plan_batch_x86(
            &product,
            &plan,
            &observation_spots,
            &initial_spots,
            &mut locals,
            &mut state,
            &mut stack,
        )
        .unwrap();

        for lane in 0..SIMD_BATCH_LANES_X86 {
            let mut lane_locals = vec![0.0; product.max_local_slots()];
            let mut lane_state = vec![0.0; product.state_vars.len()];
            let mut lane_stack = vec![0.0; plan.max_stack()];
            let mut scalar_observation_spots =
                vec![vec![0.0; initial_spots.len()]; plan.snapshot_count()];
            for (step, _) in lane_paths[lane].iter().enumerate() {
                if let Some(snapshot_index) = plan.snapshot_index_for_step(step) {
                    scalar_observation_spots[snapshot_index] = lane_paths[lane][step].clone();
                }
            }
            let scalar = evaluate_product_with_plan_in_place(
                &product,
                &plan,
                &scalar_observation_spots,
                &initial_spots,
                &mut lane_locals,
                &mut lane_state,
                &mut lane_stack,
            )
            .unwrap();
            assert!(
                (batched[lane] - scalar).abs() < 1e-8,
                "lane {lane}: expected {scalar}, got {}",
                batched[lane]
            );
        }
    }
}
