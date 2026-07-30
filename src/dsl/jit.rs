//! Cranelift JIT compiler for DSL bytecode programs.
//!
//! Compiles the stack-machine bytecode produced by `eval.rs` into native code
//! via Cranelift, targeting hot observation-date loops in Monte Carlo pricing.
//!
//! # Architecture
//!
//! The JIT compiler translates each bytecode `Program` into a single native
//! function with a C calling convention. The compiled function receives raw
//! pointers to the evaluation context (spots, locals, state, constants, PV
//! accumulator) and returns a `u8` status code (0 = Continue, 1 = Redeemed,
//! 2 = Skipped).
//!
//! The bytecode stack machine is translated to SSA form by maintaining a
//! compile-time stack of Cranelift `Value`s -- the standard approach for
//! JIT-compiling stack-based VMs.
//!
//! # Feature gate
//!
//! This module is gated behind `#[cfg(feature = "jit")]`.

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::types::{F64, I8, I64};
use cranelift_codegen::ir::{self, AbiParam, InstBuilder, MemFlags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use super::error::DslError;
use super::eval::{
    Instruction, ObservationResult, ProductExecutionPlan, Program,
    build_execution_plan_for_validated_product, opcode,
};
use super::ir::CompiledProduct;

const RESULT_CONTINUE: u8 = 0;
const RESULT_REDEEMED: u8 = 1;
const RESULT_SKIPPED: u8 = 2;
const RESULT_INVALID_PRICE_INDEX: u8 = 3;

/// Whether the current Cranelift JIT configuration is supported.
///
/// Cranelift 0.116's JIT module requires x86-64 PLT support. Constructing it
/// on another architecture can panic inside the dependency, so policy
/// selection and direct compilation both reject those targets first.
pub(crate) const fn jit_platform_supported() -> bool {
    cfg!(target_arch = "x86_64")
}

fn ensure_jit_platform_supported() -> Result<(), String> {
    if jit_platform_supported() {
        Ok(())
    } else {
        Err(format!(
            "JIT execution is unsupported on {}; the current native backend requires x86_64",
            std::env::consts::ARCH
        ))
    }
}

// ---- External helper functions (called from JIT code) ----

/// Compute worst-of performance across all assets.
///
/// # Safety
///
/// `spots` and `initial_spots` must point to valid arrays of at least
/// `num_assets` f64 values.
unsafe extern "C" fn jit_worst_of_perf(
    spots: *const f64,
    initial_spots: *const f64,
    num_assets: u64,
) -> f64 {
    let n = num_assets as usize;
    let mut wof = f64::INFINITY;
    for i in 0..n {
        let init = unsafe { *initial_spots.add(i) };
        if init > 0.0 {
            let perf = unsafe { *spots.add(i) } / init;
            // NaN-propagating minimum, matching the interpreter backends.
            if perf.is_nan() {
                return f64::NAN;
            }
            if perf < wof {
                wof = perf;
            }
        }
    }
    wof
}

/// Compute best-of performance across all assets.
///
/// # Safety
///
/// `spots` and `initial_spots` must point to valid arrays of at least
/// `num_assets` f64 values.
unsafe extern "C" fn jit_best_of_perf(
    spots: *const f64,
    initial_spots: *const f64,
    num_assets: u64,
) -> f64 {
    let n = num_assets as usize;
    let mut bof = f64::NEG_INFINITY;
    for i in 0..n {
        let init = unsafe { *initial_spots.add(i) };
        if init > 0.0 {
            let perf = unsafe { *spots.add(i) } / init;
            // NaN-propagating maximum, matching the interpreter backends.
            if perf.is_nan() {
                return f64::NAN;
            }
            if perf > bof {
                bof = perf;
            }
        }
    }
    bof
}

// ---- JIT-compiled function type ----

type JitObservationFn = unsafe extern "C" fn(
    spots: *const f64,
    initial_spots: *const f64,
    num_assets: u64,
    notional: f64,
    obs_date: f64,
    is_final: u64,
    discount: f64,
    locals: *mut f64,
    state: *mut f64,
    constants: *const f64,
    pv: *mut f64,
) -> u8;

// ---- JitCompiledProgram ----

/// A JIT-compiled bytecode program ready for repeated execution.
///
/// Holds the `JITModule` (which owns the executable memory) and the
/// function pointer to the compiled native code.
pub struct JitCompiledProgram {
    _module: JITModule,
    fn_ptr: JitObservationFn,
}

// SAFETY: The JIT module's code memory is allocated and owned; the function
// pointer is a simple code pointer that is safe to call from any thread.
unsafe impl Send for JitCompiledProgram {}
unsafe impl Sync for JitCompiledProgram {}

impl JitCompiledProgram {
    /// Compile a bytecode program to native code.
    ///
    /// The `code` slice contains `(opcode, flags, operand)` tuples matching
    /// the `Instruction` layout in `eval.rs`. The `constants` slice provides
    /// the f64 constant pool referenced by `PUSH_CONST` operands.
    pub fn compile(code: &[(u8, u8, u16)], constants: &[f64]) -> Result<Self, String> {
        let instructions: Vec<Instruction> = code
            .iter()
            .map(|&(opc, flags, operand)| Instruction {
                opcode: opc,
                flags,
                operand,
            })
            .collect();

        compile_program(&instructions, constants)
    }

    fn compile_bytecode(program: &Program) -> Result<Self, String> {
        compile_program(&program.code, &program.constants)
    }

    /// Execute the compiled program.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `spots`, `initial_spots`, `locals`,
    /// `state`, and `pv` are valid and appropriately sized.
    ///
    /// # Panics
    ///
    /// Panics if the compiled program produces an invalid asset index or an
    /// unknown status code. Use [`Self::execute_checked`] to handle those
    /// failures without panicking.
    pub unsafe fn execute(
        &self,
        spots: &[f64],
        initial_spots: &[f64],
        notional: f64,
        obs_date: f64,
        is_final: bool,
        discount: f64,
        locals: &mut [f64],
        state: &mut [f64],
        pv: &mut f64,
    ) -> ObservationResult {
        unsafe {
            self.execute_checked(
                spots,
                initial_spots,
                notional,
                obs_date,
                is_final,
                discount,
                locals,
                state,
                pv,
            )
        }
        .unwrap_or_else(|error| panic!("JIT execution failed: {error}"))
    }

    /// Execute the compiled program and report invalid runtime status codes.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `spots`, `initial_spots`, `locals`,
    /// `state`, and `pv` are valid and appropriately sized.
    pub unsafe fn execute_checked(
        &self,
        spots: &[f64],
        initial_spots: &[f64],
        notional: f64,
        obs_date: f64,
        is_final: bool,
        discount: f64,
        locals: &mut [f64],
        state: &mut [f64],
        pv: &mut f64,
    ) -> Result<ObservationResult, DslError> {
        let result = unsafe {
            (self.fn_ptr)(
                spots.as_ptr(),
                initial_spots.as_ptr(),
                spots.len() as u64,
                notional,
                obs_date,
                u64::from(is_final),
                discount,
                locals.as_mut_ptr(),
                state.as_mut_ptr(),
                std::ptr::null(),
                pv as *mut f64,
            )
        };
        match result {
            RESULT_CONTINUE => Ok(ObservationResult::Continue),
            RESULT_REDEEMED => Ok(ObservationResult::Redeemed),
            RESULT_SKIPPED => Ok(ObservationResult::Skipped),
            RESULT_INVALID_PRICE_INDEX => Err(DslError::EvalError(
                "asset index is not a non-negative integer or is out of range".to_string(),
            )),
            _ => Err(DslError::EvalError(format!(
                "JIT returned unknown status code {result}"
            ))),
        }
    }
}

/// A reusable, native-code execution plan for one compiled DSL product.
///
/// Construction performs both bytecode planning and JIT compilation. Keep this
/// value and call [`Self::evaluate_path_with_scratch`] for repeated paths so
/// compilation and allocation are amortized rather than paid inside a hot
/// Monte Carlo loop.
pub struct JitProductEvaluator {
    plan: ProductExecutionPlan,
    programs: Vec<JitCompiledProgram>,
    notional: f64,
    initial_state: Vec<f64>,
    num_underlyings: usize,
    num_locals: usize,
    num_steps: usize,
}

/// Reusable memory for [`JitProductEvaluator`] path evaluation.
///
/// Scratch values are evaluator-specific. Create them with
/// [`JitProductEvaluator::new_scratch`].
pub struct JitEvaluationScratch {
    observation_spots: Vec<Vec<f64>>,
    locals: Vec<f64>,
    state: Vec<f64>,
}

impl JitProductEvaluator {
    /// Build and JIT-compile a product execution plan.
    pub fn compile(
        product: &CompiledProduct,
        num_steps: usize,
        rate: f64,
    ) -> Result<Self, DslError> {
        product.validate()?;
        ensure_jit_platform_supported().map_err(DslError::EvalError)?;

        if num_steps == 0 {
            return Err(DslError::EvalError(
                "JIT evaluation requires num_steps > 0".to_string(),
            ));
        }
        if product.maturity <= 0.0 || !product.maturity.is_finite() {
            return Err(DslError::EvalError(
                "JIT evaluation requires a positive finite maturity".to_string(),
            ));
        }
        if !rate.is_finite() {
            return Err(DslError::EvalError(
                "JIT evaluation requires a finite rate".to_string(),
            ));
        }

        let plan = build_execution_plan_for_validated_product(product, num_steps, rate)?;
        let programs = plan
            .schedules
            .iter()
            .map(|schedule| {
                JitCompiledProgram::compile_bytecode(&schedule.program)
                    .map_err(|err| DslError::EvalError(format!("JIT compilation failed: {err}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            plan,
            programs,
            notional: product.notional,
            initial_state: product
                .state_vars
                .iter()
                .map(|state| state.initial.as_f64())
                .collect(),
            num_underlyings: product.num_underlyings,
            num_locals: product.max_local_slots(),
            num_steps,
        })
    }

    /// Allocate reusable scratch memory sized for this evaluator.
    pub fn new_scratch(&self) -> JitEvaluationScratch {
        JitEvaluationScratch {
            observation_spots: vec![vec![0.0; self.num_underlyings]; self.plan.snapshot_count()],
            locals: vec![0.0; self.num_locals],
            state: self.initial_state.clone(),
        }
    }

    pub(crate) fn snapshot_count(&self) -> usize {
        self.plan.snapshot_count()
    }

    pub(crate) fn snapshot_index_for_step(&self, step: usize) -> Option<usize> {
        self.plan.snapshot_index_for_step(step)
    }

    /// Evaluate one simulated path, allocating scratch memory for convenience.
    ///
    /// For repeated paths, prefer [`Self::evaluate_path_with_scratch`].
    pub fn evaluate_path(
        &self,
        path_spots: &[Vec<f64>],
        initial_spots: &[f64],
    ) -> Result<f64, DslError> {
        let mut scratch = self.new_scratch();
        self.evaluate_path_with_scratch(path_spots, initial_spots, &mut scratch)
    }

    /// Evaluate one simulated path using caller-reused scratch memory.
    pub fn evaluate_path_with_scratch(
        &self,
        path_spots: &[Vec<f64>],
        initial_spots: &[f64],
        scratch: &mut JitEvaluationScratch,
    ) -> Result<f64, DslError> {
        self.validate_inputs(path_spots, initial_spots, scratch)?;

        for (step, spots) in path_spots.iter().enumerate() {
            if let Some(snapshot) = self.plan.snapshot_index_for_step(step) {
                scratch.observation_spots[snapshot].copy_from_slice(spots);
            }
        }
        self.evaluate_observations_with_scratch(
            &scratch.observation_spots,
            initial_spots,
            &mut scratch.locals,
            &mut scratch.state,
        )
    }

    pub(crate) fn evaluate_captured_observations(
        &self,
        observation_spots: &[Vec<f64>],
        initial_spots: &[f64],
        scratch: &mut JitEvaluationScratch,
    ) -> Result<f64, DslError> {
        if initial_spots.len() != self.num_underlyings
            || observation_spots.len() != self.plan.snapshot_count()
            || observation_spots
                .iter()
                .any(|spots| spots.len() != self.num_underlyings)
        {
            return Err(DslError::EvalError(
                "JIT observation snapshots do not match the compiled product".to_string(),
            ));
        }
        self.validate_scratch(scratch)?;
        self.evaluate_observations_with_scratch(
            observation_spots,
            initial_spots,
            &mut scratch.locals,
            &mut scratch.state,
        )
    }

    fn evaluate_observations_with_scratch(
        &self,
        observation_spots: &[Vec<f64>],
        initial_spots: &[f64],
        locals: &mut [f64],
        state: &mut [f64],
    ) -> Result<f64, DslError> {
        state.copy_from_slice(&self.initial_state);

        let mut pv = 0.0;
        for (schedule, program) in self.plan.schedules.iter().zip(&self.programs) {
            for observation in &schedule.observations {
                locals.fill(0.0);
                let spots = &observation_spots[observation.snapshot_index];

                // SAFETY: all slices were sized from the source product and
                // validated above. The program is compiled from that same
                // product's validated bytecode.
                let result = unsafe {
                    program.execute_checked(
                        spots,
                        initial_spots,
                        self.notional,
                        observation.observation_date,
                        observation.is_final,
                        observation.discount_factor,
                        locals,
                        state,
                        &mut pv,
                    )
                }?;

                if matches!(
                    result,
                    ObservationResult::Redeemed | ObservationResult::Skipped
                ) {
                    return ensure_finite_pv(pv);
                }
            }
        }

        ensure_finite_pv(pv)
    }

    fn validate_inputs(
        &self,
        path_spots: &[Vec<f64>],
        initial_spots: &[f64],
        scratch: &JitEvaluationScratch,
    ) -> Result<(), DslError> {
        let expected_steps = self.num_steps + 1;
        if path_spots.len() != expected_steps {
            return Err(DslError::EvalError(format!(
                "JIT path has {} steps, expected {expected_steps}",
                path_spots.len()
            )));
        }
        if initial_spots.len() != self.num_underlyings {
            return Err(DslError::EvalError(format!(
                "JIT initial spots have {} assets, expected {}",
                initial_spots.len(),
                self.num_underlyings
            )));
        }
        if let Some((asset, _)) = initial_spots
            .iter()
            .enumerate()
            .find(|(_, spot)| !spot.is_finite())
        {
            return Err(DslError::EvalError(format!(
                "JIT initial_spots[{asset}] must be finite"
            )));
        }
        if let Some((step, spots)) = path_spots
            .iter()
            .enumerate()
            .find(|(_, spots)| spots.len() != self.num_underlyings)
        {
            return Err(DslError::EvalError(format!(
                "JIT path step {step} has {} assets, expected {}",
                spots.len(),
                self.num_underlyings
            )));
        }
        if let Some((step, asset)) = path_spots.iter().enumerate().find_map(|(step, spots)| {
            spots
                .iter()
                .position(|spot| !spot.is_finite())
                .map(|asset| (step, asset))
        }) {
            return Err(DslError::EvalError(format!(
                "JIT path_spots[{step}][{asset}] must be finite"
            )));
        }
        self.validate_scratch(scratch)
    }

    fn validate_scratch(&self, scratch: &JitEvaluationScratch) -> Result<(), DslError> {
        if scratch.observation_spots.len() != self.plan.snapshot_count()
            || scratch
                .observation_spots
                .iter()
                .any(|spots| spots.len() != self.num_underlyings)
            || scratch.locals.len() != self.num_locals
            || scratch.state.len() != self.initial_state.len()
        {
            return Err(DslError::EvalError(
                "JIT scratch belongs to a different product evaluator".to_string(),
            ));
        }
        Ok(())
    }
}

fn ensure_finite_pv(pv: f64) -> Result<f64, DslError> {
    if pv.is_finite() {
        Ok(pv)
    } else {
        Err(DslError::EvalError(
            "JIT product evaluation produced a non-finite present value".to_string(),
        ))
    }
}

// ---- Compile-time value stack ----

struct CompileStack {
    values: Vec<ir::Value>,
}

impl CompileStack {
    fn new() -> Self {
        Self {
            values: Vec::with_capacity(32),
        }
    }

    fn push(&mut self, val: ir::Value) {
        self.values.push(val);
    }

    fn pop(&mut self) -> ir::Value {
        self.values
            .pop()
            .expect("JIT compile stack underflow -- bytecode is malformed")
    }

    fn clear(&mut self) {
        self.values.clear();
    }
}

// ---- Parameter indices ----

const PARAM_SPOTS: usize = 0;
const PARAM_INITIAL_SPOTS: usize = 1;
const PARAM_NUM_ASSETS: usize = 2;
const PARAM_NOTIONAL: usize = 3;
const PARAM_OBS_DATE: usize = 4;
const PARAM_IS_FINAL: usize = 5;
const PARAM_DISCOUNT: usize = 6;
const PARAM_LOCALS: usize = 7;
const PARAM_STATE: usize = 8;
const PARAM_CONSTANTS: usize = 9;
const PARAM_PV: usize = 10;

// ---- Core compilation logic ----

fn compile_program(
    instructions: &[Instruction],
    constants: &[f64],
) -> Result<JitCompiledProgram, String> {
    ensure_jit_platform_supported()?;
    validate_program(instructions, constants)?;

    // 1. Create the JIT module.
    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| format!("failed to create JIT builder: {e}"))?;

    jit_builder.symbol("jit_worst_of_perf", jit_worst_of_perf as *const u8);
    jit_builder.symbol("jit_best_of_perf", jit_best_of_perf as *const u8);
    jit_builder.symbol("exp", libm_ffi::exp_ptr());
    jit_builder.symbol("log", libm_ffi::log_ptr());

    let mut module = JITModule::new(jit_builder);

    // 2. Build the function signature.
    let ptr_type = module.target_config().pointer_type();

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // spots
    sig.params.push(AbiParam::new(ptr_type)); // initial_spots
    sig.params.push(AbiParam::new(I64)); // num_assets
    sig.params.push(AbiParam::new(F64)); // notional
    sig.params.push(AbiParam::new(F64)); // obs_date
    sig.params.push(AbiParam::new(I64)); // is_final
    sig.params.push(AbiParam::new(F64)); // discount
    sig.params.push(AbiParam::new(ptr_type)); // locals
    sig.params.push(AbiParam::new(ptr_type)); // state
    sig.params.push(AbiParam::new(ptr_type)); // constants
    sig.params.push(AbiParam::new(ptr_type)); // pv
    sig.returns.push(AbiParam::new(I8)); // result code

    let func_id = module
        .declare_function("jit_observation", Linkage::Local, &sig)
        .map_err(|e| format!("failed to declare function: {e}"))?;

    // 3. Declare external functions.
    let exp_func_id = declare_ext_f64_unary(&mut module, "exp")?;
    let log_func_id = declare_ext_f64_unary(&mut module, "log")?;
    let wof_perf_func_id = declare_ext_perf_helper(&mut module, "jit_worst_of_perf")?;
    let bof_perf_func_id = declare_ext_perf_helper(&mut module, "jit_best_of_perf")?;

    // 4. Build the function body.
    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut fn_builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let block_map = build_block_map(instructions, &mut builder);

        let entry_block = block_map[&0];
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);

        let params: Vec<ir::Value> = builder.block_params(entry_block).to_vec();

        let var_spots = Variable::from_u32(0);
        let var_initial_spots = Variable::from_u32(1);
        let var_num_assets = Variable::from_u32(2);
        let var_notional = Variable::from_u32(3);
        let var_obs_date = Variable::from_u32(4);
        let var_is_final = Variable::from_u32(5);
        let var_discount = Variable::from_u32(6);
        let var_locals = Variable::from_u32(7);
        let var_state = Variable::from_u32(8);
        let var_constants = Variable::from_u32(9);
        let var_pv = Variable::from_u32(10);

        builder.declare_var(var_spots, ptr_type);
        builder.declare_var(var_initial_spots, ptr_type);
        builder.declare_var(var_num_assets, I64);
        builder.declare_var(var_notional, F64);
        builder.declare_var(var_obs_date, F64);
        builder.declare_var(var_is_final, I64);
        builder.declare_var(var_discount, F64);
        builder.declare_var(var_locals, ptr_type);
        builder.declare_var(var_state, ptr_type);
        builder.declare_var(var_constants, ptr_type);
        builder.declare_var(var_pv, ptr_type);

        builder.def_var(var_spots, params[PARAM_SPOTS]);
        builder.def_var(var_initial_spots, params[PARAM_INITIAL_SPOTS]);
        builder.def_var(var_num_assets, params[PARAM_NUM_ASSETS]);
        builder.def_var(var_notional, params[PARAM_NOTIONAL]);
        builder.def_var(var_obs_date, params[PARAM_OBS_DATE]);
        builder.def_var(var_is_final, params[PARAM_IS_FINAL]);
        builder.def_var(var_discount, params[PARAM_DISCOUNT]);
        builder.def_var(var_locals, params[PARAM_LOCALS]);
        builder.def_var(var_state, params[PARAM_STATE]);
        builder.def_var(var_constants, params[PARAM_CONSTANTS]);
        builder.def_var(var_pv, params[PARAM_PV]);

        let exp_ref = module.declare_func_in_func(exp_func_id, builder.func);
        let log_ref = module.declare_func_in_func(log_func_id, builder.func);
        let wof_perf_ref = module.declare_func_in_func(wof_perf_func_id, builder.func);
        let bof_perf_ref = module.declare_func_in_func(bof_perf_func_id, builder.func);

        let return_continue_block = builder.create_block();
        let return_redeemed_block = builder.create_block();
        let return_skipped_block = builder.create_block();
        let return_invalid_price_index_block = builder.create_block();

        let mut stack = CompileStack::new();
        let mem = MemFlags::new();
        let mut block_terminated = false;

        let mut pc = 0usize;
        while pc < instructions.len() {
            if pc > 0
                && let Some(&target_block) = block_map.get(&pc)
            {
                // JUMP_FALSE already switches to the fall-through block
                // (which is this pc's block); avoid emitting a self-jump
                // and re-switching to an already-current block.
                if builder.current_block() != Some(target_block) {
                    if !block_terminated {
                        builder.ins().jump(target_block, &[]);
                    }
                    builder.switch_to_block(target_block);
                    // Valid DSL control flow reaches statement boundaries
                    // with an empty stack. Clear any values left by
                    // unreachable bytecode lowered into a dead block.
                    stack.clear();
                    block_terminated = false;
                }
            }

            let inst = instructions[pc];

            match inst.opcode {
                opcode::PUSH_CONST => {
                    let idx = inst.operand as usize;
                    if idx >= constants.len() {
                        return Err(format!(
                            "PUSH_CONST operand {idx} out of range (have {} constants)",
                            constants.len()
                        ));
                    }
                    let val = builder.ins().f64const(constants[idx]);
                    stack.push(val);
                }
                opcode::PUSH_TRUE => {
                    stack.push(builder.ins().f64const(1.0));
                }
                opcode::PUSH_FALSE => {
                    stack.push(builder.ins().f64const(0.0));
                }
                opcode::PUSH_LOCAL => {
                    let ptr = builder.use_var(var_locals);
                    let offset = (inst.operand as i32) * 8;
                    let val = builder.ins().load(F64, mem, ptr, offset);
                    stack.push(val);
                }
                opcode::PUSH_STATE => {
                    let ptr = builder.use_var(var_state);
                    let offset = (inst.operand as i32) * 8;
                    let val = builder.ins().load(F64, mem, ptr, offset);
                    stack.push(val);
                }
                opcode::PUSH_NOTIONAL => {
                    stack.push(builder.use_var(var_notional));
                }
                opcode::PUSH_DATE => {
                    stack.push(builder.use_var(var_obs_date));
                }
                opcode::PUSH_IS_FINAL => {
                    let i = builder.use_var(var_is_final);
                    let val = builder.ins().fcvt_from_uint(F64, i);
                    stack.push(val);
                }

                opcode::ADD => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    stack.push(builder.ins().fadd(lhs, rhs));
                }
                opcode::SUB => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    stack.push(builder.ins().fsub(lhs, rhs));
                }
                opcode::MUL => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    stack.push(builder.ins().fmul(lhs, rhs));
                }
                opcode::DIV => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let zero = builder.ins().f64const(0.0);
                    let is_zero = builder.ins().fcmp(FloatCC::Equal, rhs, zero);
                    let quotient = builder.ins().fdiv(lhs, rhs);
                    let nan = builder.ins().f64const(f64::NAN);
                    stack.push(builder.ins().select(is_zero, nan, quotient));
                }
                opcode::NEG => {
                    let v = stack.pop();
                    stack.push(builder.ins().fneg(v));
                }
                opcode::ABS => {
                    let v = stack.pop();
                    stack.push(builder.ins().fabs(v));
                }
                opcode::EXP => {
                    let v = stack.pop();
                    let call = builder.ins().call(exp_ref, &[v]);
                    stack.push(builder.inst_results(call)[0]);
                }
                opcode::LOG => {
                    let v = stack.pop();
                    let call = builder.ins().call(log_ref, &[v]);
                    stack.push(builder.inst_results(call)[0]);
                }
                // Cranelift fmin/fmax propagate NaN, matching the scalar
                // interpreter's nan_min/nan_max and the SIMD backends.
                opcode::MIN => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    stack.push(builder.ins().fmin(lhs, rhs));
                }
                opcode::MAX => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    stack.push(builder.ins().fmax(lhs, rhs));
                }

                opcode::EQ => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let diff = builder.ins().fsub(lhs, rhs);
                    let abs_diff = builder.ins().fabs(diff);
                    let eps = builder.ins().f64const(f64::EPSILON);
                    let cmp = builder.ins().fcmp(FloatCC::LessThan, abs_diff, eps);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    stack.push(builder.ins().select(cmp, one, zero));
                }
                opcode::NE => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let diff = builder.ins().fsub(lhs, rhs);
                    let abs_diff = builder.ins().fabs(diff);
                    let eps = builder.ins().f64const(f64::EPSILON);
                    let cmp = builder
                        .ins()
                        .fcmp(FloatCC::GreaterThanOrEqual, abs_diff, eps);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    stack.push(builder.ins().select(cmp, one, zero));
                }
                opcode::LT => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let cmp = builder.ins().fcmp(FloatCC::LessThan, lhs, rhs);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    stack.push(builder.ins().select(cmp, one, zero));
                }
                opcode::LE => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    stack.push(builder.ins().select(cmp, one, zero));
                }
                opcode::GT => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let cmp = builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    stack.push(builder.ins().select(cmp, one, zero));
                }
                opcode::GE => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    stack.push(builder.ins().select(cmp, one, zero));
                }
                opcode::AND => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let zero = builder.ins().f64const(0.0);
                    let lhs_b = builder.ins().fcmp(FloatCC::NotEqual, lhs, zero);
                    let rhs_b = builder.ins().fcmp(FloatCC::NotEqual, rhs, zero);
                    let both = builder.ins().band(lhs_b, rhs_b);
                    let one = builder.ins().f64const(1.0);
                    stack.push(builder.ins().select(both, one, zero));
                }
                opcode::OR => {
                    let rhs = stack.pop();
                    let lhs = stack.pop();
                    let zero = builder.ins().f64const(0.0);
                    let lhs_b = builder.ins().fcmp(FloatCC::NotEqual, lhs, zero);
                    let rhs_b = builder.ins().fcmp(FloatCC::NotEqual, rhs, zero);
                    let either = builder.ins().bor(lhs_b, rhs_b);
                    let one = builder.ins().f64const(1.0);
                    stack.push(builder.ins().select(either, one, zero));
                }
                opcode::NOT => {
                    let v = stack.pop();
                    let zero = builder.ins().f64const(0.0);
                    let is_false = builder.ins().fcmp(FloatCC::Equal, v, zero);
                    let one = builder.ins().f64const(1.0);
                    stack.push(builder.ins().select(is_false, one, zero));
                }

                opcode::STORE_LOCAL => {
                    let v = stack.pop();
                    let ptr = builder.use_var(var_locals);
                    let offset = (inst.operand as i32) * 8;
                    builder.ins().store(mem, v, ptr, offset);
                }
                opcode::STORE_STATE => {
                    let v = stack.pop();
                    let ptr = builder.use_var(var_state);
                    let offset = (inst.operand as i32) * 8;
                    builder.ins().store(mem, v, ptr, offset);
                }

                opcode::PAY => {
                    let v = stack.pop();
                    let disc = builder.use_var(var_discount);
                    let discounted = builder.ins().fmul(v, disc);
                    let pv_ptr = builder.use_var(var_pv);
                    let cur = builder.ins().load(F64, mem, pv_ptr, 0);
                    let new_pv = builder.ins().fadd(cur, discounted);
                    builder.ins().store(mem, new_pv, pv_ptr, 0);
                }
                opcode::REDEEM => {
                    let v = stack.pop();
                    let disc = builder.use_var(var_discount);
                    let discounted = builder.ins().fmul(v, disc);
                    let pv_ptr = builder.use_var(var_pv);
                    let cur = builder.ins().load(F64, mem, pv_ptr, 0);
                    let new_pv = builder.ins().fadd(cur, discounted);
                    builder.ins().store(mem, new_pv, pv_ptr, 0);
                    builder.ins().jump(return_redeemed_block, &[]);
                    // Continue lowering any remaining instructions into a
                    // fresh (unreachable) block. The new block is unfilled,
                    // so `block_terminated` must be reset to false.
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
                    block_terminated = false;
                }
                opcode::SKIP => {
                    builder.ins().jump(return_skipped_block, &[]);
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
                    block_terminated = false;
                }

                opcode::PRICE => {
                    // Validate before conversion/loading. NaN and negative
                    // values can trap `fcvt_to_uint`, while fractional and
                    // out-of-range values violate the scalar/SIMD PRICE
                    // contract. All invalid cases return the checked JIT
                    // status without touching the spots array.
                    let idx_f64 = stack.pop();
                    let zero = builder.ins().f64const(0.0);
                    let non_negative =
                        builder
                            .ins()
                            .fcmp(FloatCC::GreaterThanOrEqual, idx_f64, zero);
                    let num_assets = builder.use_var(var_num_assets);
                    let num_assets_f64 = builder.ins().fcvt_from_uint(F64, num_assets);
                    let below_num_assets =
                        builder
                            .ins()
                            .fcmp(FloatCC::LessThan, idx_f64, num_assets_f64);
                    let in_range = builder.ins().band(non_negative, below_num_assets);
                    let integer_check_block = builder.create_block();
                    let load_block = builder.create_block();
                    builder.ins().brif(
                        in_range,
                        integer_check_block,
                        &[],
                        return_invalid_price_index_block,
                        &[],
                    );
                    builder.switch_to_block(integer_check_block);

                    let idx_i64 = builder.ins().fcvt_to_uint(I64, idx_f64);
                    let round_trip = builder.ins().fcvt_from_uint(F64, idx_i64);
                    let is_integer = builder.ins().fcmp(FloatCC::Equal, idx_f64, round_trip);
                    builder.ins().brif(
                        is_integer,
                        load_block,
                        &[],
                        return_invalid_price_index_block,
                        &[],
                    );
                    builder.switch_to_block(load_block);

                    let eight = builder.ins().iconst(I64, 8);
                    let byte_off = builder.ins().imul(idx_i64, eight);
                    let s_ptr = builder.use_var(var_spots);
                    let addr = builder.ins().iadd(s_ptr, byte_off);
                    stack.push(builder.ins().load(F64, mem, addr, 0));
                }
                opcode::WORST_OF => {
                    let n = inst.operand as usize;
                    if n == 0 {
                        stack.push(builder.ins().f64const(f64::INFINITY));
                    } else {
                        let mut acc = stack.pop();
                        for _ in 1..n {
                            let v = stack.pop();
                            acc = builder.ins().fmin(v, acc);
                        }
                        stack.push(acc);
                    }
                }
                opcode::BEST_OF => {
                    let n = inst.operand as usize;
                    if n == 0 {
                        stack.push(builder.ins().f64const(f64::NEG_INFINITY));
                    } else {
                        let mut acc = stack.pop();
                        for _ in 1..n {
                            let v = stack.pop();
                            acc = builder.ins().fmax(v, acc);
                        }
                        stack.push(acc);
                    }
                }
                opcode::WORST_OF_PERF => {
                    let s = builder.use_var(var_spots);
                    let is = builder.use_var(var_initial_spots);
                    let na = builder.use_var(var_num_assets);
                    let call = builder.ins().call(wof_perf_ref, &[s, is, na]);
                    stack.push(builder.inst_results(call)[0]);
                }
                opcode::BEST_OF_PERF => {
                    let s = builder.use_var(var_spots);
                    let is = builder.use_var(var_initial_spots);
                    let na = builder.use_var(var_num_assets);
                    let call = builder.ins().call(bof_perf_ref, &[s, is, na]);
                    stack.push(builder.inst_results(call)[0]);
                }

                opcode::JUMP => {
                    let target_pc = inst.operand as usize;
                    let target_block = block_map[&target_pc];
                    builder.ins().jump(target_block, &[]);
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
                    block_terminated = false;
                }
                opcode::JUMP_FALSE => {
                    let target_pc = inst.operand as usize;
                    let cond = stack.pop();
                    let zero = builder.ins().f64const(0.0);
                    let is_false = builder.ins().fcmp(FloatCC::Equal, cond, zero);

                    let false_block = block_map[&target_pc];
                    let true_block = block_map
                        .get(&(pc + 1))
                        .copied()
                        .unwrap_or_else(|| builder.create_block());

                    builder
                        .ins()
                        .brif(is_false, false_block, &[], true_block, &[]);
                    // The branch terminates the current block. Continue
                    // lowering in the fall-through block immediately; the
                    // loop's target-block check avoids switching to it twice.
                    builder.switch_to_block(true_block);
                    block_terminated = false;
                }

                _ => {
                    return Err(format!("unknown opcode 0x{:02x} at pc {pc}", inst.opcode));
                }
            }

            pc += 1;
        }

        // If the program ends with JUMP_FALSE, its fall-through block (pc+1 ==
        // instructions.len()) *is* the end block from the block map and is the
        // current block here; the fall-through jump below already terminates
        // it, so the end-block epilogue must not fill it a second time.
        let last_block = builder.current_block();
        if !block_terminated {
            builder.ins().jump(return_continue_block, &[]);
        }

        // A jump may target the end of the program (pc == instructions.len(),
        // e.g. the exit of an if/else); that block was created in the block
        // map but never lowered, so terminate it with a fall-through to
        // Continue -- unless it was already filled above.
        if let Some(&end_block) = block_map.get(&instructions.len())
            && Some(end_block) != last_block
        {
            builder.switch_to_block(end_block);
            builder.ins().jump(return_continue_block, &[]);
        }

        builder.switch_to_block(return_continue_block);
        let rc = builder.ins().iconst(I8, i64::from(RESULT_CONTINUE));
        builder.ins().return_(&[rc]);

        builder.switch_to_block(return_redeemed_block);
        let rr = builder.ins().iconst(I8, i64::from(RESULT_REDEEMED));
        builder.ins().return_(&[rr]);

        builder.switch_to_block(return_skipped_block);
        let rs = builder.ins().iconst(I8, i64::from(RESULT_SKIPPED));
        builder.ins().return_(&[rs]);

        builder.switch_to_block(return_invalid_price_index_block);
        let ri = builder
            .ins()
            .iconst(I8, i64::from(RESULT_INVALID_PRICE_INDEX));
        builder.ins().return_(&[ri]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    // 5. Compile and extract function pointer.
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("failed to define function: {e}"))?;

    module
        .finalize_definitions()
        .map_err(|e| format!("failed to finalize definitions: {e}"))?;

    let code_ptr = module.get_finalized_function(func_id);
    let fn_ptr: JitObservationFn = unsafe { std::mem::transmute(code_ptr) };

    Ok(JitCompiledProgram {
        _module: module,
        fn_ptr,
    })
}

/// Validate bytecode before handing it to Cranelift.
///
/// Besides producing useful errors for malformed public `compile` inputs, this
/// keeps the code generator's compile-time stack operations and raw memory
/// accesses from being reached with structurally invalid bytecode. The DSL
/// compiler only emits forward, statement-level branches with an empty stack,
/// which is the control-flow form supported by this JIT.
fn validate_program(instructions: &[Instruction], constants: &[f64]) -> Result<(), String> {
    for (pc, inst) in instructions.iter().enumerate() {
        if inst.flags != 0 {
            return Err(format!(
                "unsupported instruction flags 0x{:02x} at pc {pc}",
                inst.flags
            ));
        }

        match inst.opcode {
            opcode::JUMP | opcode::JUMP_FALSE => {
                let target = inst.operand as usize;
                if target > instructions.len() {
                    return Err(format!(
                        "jump target {target} out of range at pc {pc} (program length {})",
                        instructions.len()
                    ));
                }
                if target <= pc {
                    return Err(format!(
                        "backward/self jump from pc {pc} to {target} is not supported"
                    ));
                }
            }
            opcode::PUSH_CONST => {
                let idx = inst.operand as usize;
                if idx >= constants.len() {
                    return Err(format!(
                        "PUSH_CONST operand {idx} out of range (have {} constants)",
                        constants.len()
                    ));
                }
            }
            opcode::SKIP
            | opcode::PUSH_LOCAL
            | opcode::PUSH_STATE
            | opcode::PUSH_NOTIONAL
            | opcode::PUSH_DATE
            | opcode::PUSH_IS_FINAL
            | opcode::PUSH_TRUE
            | opcode::PUSH_FALSE
            | opcode::STORE_LOCAL
            | opcode::STORE_STATE
            | opcode::ADD
            | opcode::SUB
            | opcode::MUL
            | opcode::DIV
            | opcode::NEG
            | opcode::ABS
            | opcode::EXP
            | opcode::LOG
            | opcode::MIN
            | opcode::MAX
            | opcode::EQ
            | opcode::NE
            | opcode::LT
            | opcode::LE
            | opcode::GT
            | opcode::GE
            | opcode::AND
            | opcode::OR
            | opcode::NOT
            | opcode::PAY
            | opcode::REDEEM
            | opcode::PRICE
            | opcode::WORST_OF
            | opcode::BEST_OF
            | opcode::WORST_OF_PERF
            | opcode::BEST_OF_PERF => {}
            _ => {
                return Err(format!("unknown opcode 0x{:02x} at pc {pc}", inst.opcode));
            }
        }
    }

    let mut depths = vec![None; instructions.len() + 1];
    let mut work = vec![(0usize, 0usize)];

    while let Some((pc, depth)) = work.pop() {
        if let Some(existing) = depths[pc] {
            if existing != depth {
                return Err(format!(
                    "inconsistent stack depth at pc {pc}: {existing} versus {depth}"
                ));
            }
            continue;
        }
        depths[pc] = Some(depth);

        if pc == instructions.len() {
            if depth != 0 {
                return Err(format!(
                    "program exits with {depth} value(s) left on the stack"
                ));
            }
            continue;
        }

        let inst = instructions[pc];
        let (required, delta) = stack_effect(inst);
        if depth < required {
            return Err(format!(
                "stack underflow at pc {pc}: opcode 0x{:02x} needs {required} value(s), has {depth}",
                inst.opcode
            ));
        }
        let next_depth = (depth as isize + delta) as usize;

        let mut add_successor = |successor: usize| -> Result<(), String> {
            if matches!(inst.opcode, opcode::JUMP | opcode::JUMP_FALSE) && next_depth != 0 {
                return Err(format!(
                    "control flow at pc {pc} carries {next_depth} stack value(s); only statement-level branches are supported"
                ));
            }
            work.push((successor, next_depth));
            Ok(())
        };

        match inst.opcode {
            opcode::JUMP => add_successor(inst.operand as usize)?,
            opcode::JUMP_FALSE => {
                add_successor(inst.operand as usize)?;
                add_successor(pc + 1)?;
            }
            opcode::REDEEM | opcode::SKIP => {}
            _ => add_successor(pc + 1)?,
        }
    }

    Ok(())
}

#[inline]
fn stack_effect(inst: Instruction) -> (usize, isize) {
    match inst.opcode {
        opcode::PUSH_CONST
        | opcode::PUSH_LOCAL
        | opcode::PUSH_STATE
        | opcode::PUSH_NOTIONAL
        | opcode::PUSH_DATE
        | opcode::PUSH_IS_FINAL
        | opcode::PUSH_TRUE
        | opcode::PUSH_FALSE
        | opcode::WORST_OF_PERF
        | opcode::BEST_OF_PERF => (0, 1),
        opcode::STORE_LOCAL
        | opcode::STORE_STATE
        | opcode::PAY
        | opcode::REDEEM
        | opcode::JUMP_FALSE => (1, -1),
        opcode::ADD
        | opcode::SUB
        | opcode::MUL
        | opcode::DIV
        | opcode::MIN
        | opcode::MAX
        | opcode::EQ
        | opcode::NE
        | opcode::LT
        | opcode::LE
        | opcode::GT
        | opcode::GE
        | opcode::AND
        | opcode::OR => (2, -1),
        opcode::NEG | opcode::ABS | opcode::EXP | opcode::LOG | opcode::NOT | opcode::PRICE => {
            (1, 0)
        }
        opcode::WORST_OF | opcode::BEST_OF => {
            let count = inst.operand as usize;
            (count, 1 - count as isize)
        }
        opcode::JUMP | opcode::SKIP => (0, 0),
        _ => unreachable!("opcodes are checked before stack validation"),
    }
}

fn declare_ext_f64_unary(module: &mut JITModule, name: &str) -> Result<FuncId, String> {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));
    module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| format!("failed to declare '{name}': {e}"))
}

fn declare_ext_perf_helper(module: &mut JITModule, name: &str) -> Result<FuncId, String> {
    let ptr_type = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(I64));
    sig.returns.push(AbiParam::new(F64));
    module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| format!("failed to declare '{name}': {e}"))
}

fn build_block_map(
    instructions: &[Instruction],
    builder: &mut FunctionBuilder<'_>,
) -> std::collections::HashMap<usize, ir::Block> {
    let mut targets = std::collections::HashSet::new();
    targets.insert(0usize);

    for (pc, inst) in instructions.iter().enumerate() {
        match inst.opcode {
            opcode::JUMP => {
                targets.insert(inst.operand as usize);
            }
            opcode::JUMP_FALSE => {
                targets.insert(inst.operand as usize);
                targets.insert(pc + 1);
            }
            _ => {}
        }
    }

    let mut block_map = std::collections::HashMap::new();
    for &target_pc in &targets {
        block_map.insert(target_pc, builder.create_block());
    }
    block_map
}

// ---- libm FFI ----

mod libm_ffi {
    // We need function pointers to exp and log (ln) from libm/libc.
    // On macOS/Linux these are available in the C runtime.

    unsafe extern "C" {
        safe fn exp(x: f64) -> f64;
        safe fn log(x: f64) -> f64;
    }

    pub fn exp_ptr() -> *const u8 {
        exp as *const u8
    }

    pub fn log_ptr() -> *const u8 {
        log as *const u8
    }
}

// ---- Tests ----

#[cfg(all(test, not(target_arch = "x86_64")))]
mod unsupported_platform_tests {
    use super::*;

    #[test]
    fn direct_jit_compilation_returns_unsupported_without_initializing_backend() {
        let error = JitCompiledProgram::compile(&[], &[])
            .err()
            .expect("non-x86_64 JIT compilation must be unsupported");
        assert!(
            error.contains("unsupported"),
            "unexpected JIT platform error: {error}"
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::dsl::{eval::evaluate_product, parse_and_compile};

    fn inst(opcode: u8, operand: u16) -> (u8, u8, u16) {
        (opcode, 0, operand)
    }

    #[test]
    fn jit_push_const_and_pay() {
        let code = vec![inst(opcode::PUSH_CONST, 0), inst(opcode::PAY, 0)];
        let compiled = JitCompiledProgram::compile(&code, &[100.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1e6,
                0.5,
                false,
                0.9,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };

        assert!(matches!(result, ObservationResult::Continue));
        assert!((pv - 90.0).abs() < 1e-10, "expected 90.0, got {pv}");
    }

    #[test]
    fn jit_push_const_and_redeem() {
        let code = vec![inst(opcode::PUSH_CONST, 0), inst(opcode::REDEEM, 0)];
        let compiled = JitCompiledProgram::compile(&code, &[50.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 10.0;

        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1e6,
                1.0,
                false,
                0.95,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };

        assert!(matches!(result, ObservationResult::Redeemed));
        assert!((pv - 57.5).abs() < 1e-10, "expected 57.5, got {pv}");
    }

    #[test]
    fn jit_arithmetic() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::ADD, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[3.0, 4.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 7.0).abs() < 1e-10, "expected 7.0, got {pv}");
    }

    #[test]
    fn jit_locals_roundtrip() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_LOCAL, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[42.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 42.0).abs() < 1e-10, "expected 42.0, got {pv}");
        assert!(
            (locals[0] - 42.0).abs() < 1e-10,
            "locals[0] expected 42.0, got {}",
            locals[0]
        );
    }

    #[test]
    fn jit_conditional_false_branch() {
        let code = vec![
            inst(opcode::PUSH_FALSE, 0),
            inst(opcode::JUMP_FALSE, 4),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[100.0, 200.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 200.0).abs() < 1e-10, "expected 200.0, got {pv}");
    }

    #[test]
    fn jit_conditional_true_branch() {
        let code = vec![
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::JUMP_FALSE, 4),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[100.0, 200.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 300.0).abs() < 1e-10, "expected 300.0, got {pv}");
    }

    #[test]
    fn jit_comparison_lt() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::LT, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[3.0, 5.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            (pv - 1.0).abs() < 1e-10,
            "3 < 5 should be true (1.0), got {pv}"
        );
    }

    #[test]
    fn jit_worst_of_perf_test() {
        let code = vec![inst(opcode::WORST_OF_PERF, 0), inst(opcode::PAY, 0)];
        let compiled = JitCompiledProgram::compile(&code, &[]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[120.0, 90.0],
                &[100.0, 100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 0.9).abs() < 1e-10, "expected 0.9, got {pv}");
    }

    #[test]
    fn jit_notional_and_date() {
        let code = vec![
            inst(opcode::PUSH_NOTIONAL, 0),
            inst(opcode::PUSH_DATE, 0),
            inst(opcode::MUL, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1e6,
                0.5,
                false,
                0.95,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 475_000.0).abs() < 1e-6, "expected 475000, got {pv}");
    }

    #[test]
    fn jit_skip() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
            inst(opcode::SKIP, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[10.0, 20.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };

        assert!(matches!(result, ObservationResult::Skipped));
        assert!((pv - 10.0).abs() < 1e-10, "expected 10.0, got {pv}");
    }

    #[test]
    fn jit_if_else_pattern() {
        // if true then pay 100 else pay 200
        let code = vec![
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::JUMP_FALSE, 5),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
            inst(opcode::JUMP, 7),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[100.0, 200.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 100.0).abs() < 1e-10, "expected 100.0, got {pv}");
    }

    #[test]
    fn jit_div_by_zero_returns_nan() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::DIV, 0),
            inst(opcode::STORE_LOCAL, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[5.0, 0.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(locals[0].is_nan(), "expected NaN, got {}", locals[0]);
    }

    #[test]
    fn jit_and_or_not() {
        let code = vec![
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::AND, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_FALSE, 0),
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::OR, 0),
            inst(opcode::STORE_LOCAL, 1),
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::NOT, 0),
            inst(opcode::STORE_LOCAL, 2),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            (locals[0] - 1.0).abs() < 1e-10,
            "AND(T,T)={}, want 1.0",
            locals[0]
        );
        assert!(
            (locals[1] - 1.0).abs() < 1e-10,
            "OR(F,T)={}, want 1.0",
            locals[1]
        );
        assert!(locals[2].abs() < 1e-10, "NOT(T)={}, want 0.0", locals[2]);
    }

    #[test]
    fn jit_nan_is_truthy_in_and_or() {
        // NaN produced at runtime (0/0) must be truthy in AND/OR, matching
        // the scalar interpreter (`v != 0.0`) and the AVX2/NEON batch
        // backends. FloatCC::NotEqual is unordered, so NaN != 0 is true.
        let code = vec![
            inst(opcode::PUSH_CONST, 0), // 0.0
            inst(opcode::PUSH_CONST, 0), // 0.0
            inst(opcode::DIV, 0),        // NaN at runtime
            inst(opcode::PUSH_CONST, 1), // 1.0
            inst(opcode::AND, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::DIV, 0), // NaN at runtime
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::OR, 0),
            inst(opcode::STORE_LOCAL, 1),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[0.0, 1.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        // Scalar semantics: f64_to_bool(NaN) is true, so both are 1.0.
        assert!(
            (locals[0] - 1.0).abs() < 1e-10,
            "AND(NaN, 1.0)={}, want 1.0 (scalar treats NaN as truthy)",
            locals[0]
        );
        assert!(
            (locals[1] - 1.0).abs() < 1e-10,
            "OR(NaN, 0.0)={}, want 1.0 (scalar treats NaN as truthy)",
            locals[1]
        );
    }

    #[test]
    fn jit_program_ending_with_jump_false_compiles() {
        // Hand-built bytecode may end with JUMP_FALSE whose fall-through
        // (pc+1) and target are the end of the program. The fall-through
        // block then *is* the end block; previously the epilogue terminated
        // it twice and panicked inside Cranelift's FunctionBuilder.
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
            inst(opcode::PUSH_TRUE, 0),
            inst(opcode::JUMP_FALSE, 4),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[25.0])
            .expect("bytecode ending with JUMP_FALSE must compile");

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(matches!(result, ObservationResult::Continue));
        assert!((pv - 25.0).abs() < 1e-10, "expected 25.0, got {pv}");

        // Same shape with a false condition: jumps straight to the end.
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
            inst(opcode::PUSH_FALSE, 0),
            inst(opcode::JUMP_FALSE, 4),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[25.0])
            .expect("bytecode ending with JUMP_FALSE must compile");
        let mut pv = 0.0;
        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(matches!(result, ObservationResult::Continue));
        assert!((pv - 25.0).abs() < 1e-10, "expected 25.0, got {pv}");
    }

    #[test]
    fn jit_neg_abs_min_max() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::NEG, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::ABS, 0),
            inst(opcode::STORE_LOCAL, 1),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 2),
            inst(opcode::MIN, 0),
            inst(opcode::STORE_LOCAL, 2),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 2),
            inst(opcode::MAX, 0),
            inst(opcode::STORE_LOCAL, 3),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[5.0, -3.0, 2.0]).unwrap();

        let mut locals = vec![0.0; 8];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((locals[0] - (-5.0)).abs() < 1e-10, "NEG(5)={}", locals[0]);
        assert!((locals[1] - 3.0).abs() < 1e-10, "ABS(-3)={}", locals[1]);
        assert!((locals[2] - 2.0).abs() < 1e-10, "MIN(5,2)={}", locals[2]);
        assert!((locals[3] - 5.0).abs() < 1e-10, "MAX(5,2)={}", locals[3]);
    }

    #[test]
    fn jit_min_max_propagate_nan() {
        // min/max must propagate NaN regardless of operand order, matching
        // the scalar interpreter and SIMD backends.
        let code = vec![
            inst(opcode::PUSH_CONST, 0), // NaN
            inst(opcode::PUSH_CONST, 1), // 5.0
            inst(opcode::MIN, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::MIN, 0),
            inst(opcode::STORE_LOCAL, 1),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::MAX, 0),
            inst(opcode::STORE_LOCAL, 2),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::MAX, 0),
            inst(opcode::STORE_LOCAL, 3),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[f64::NAN, 5.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(locals[0].is_nan(), "MIN(NaN, 5)={}", locals[0]);
        assert!(locals[1].is_nan(), "MIN(5, NaN)={}", locals[1]);
        assert!(locals[2].is_nan(), "MAX(NaN, 5)={}", locals[2]);
        assert!(locals[3].is_nan(), "MAX(5, NaN)={}", locals[3]);
    }

    #[test]
    fn jit_worst_of_propagates_nan() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1), // NaN
            inst(opcode::PUSH_CONST, 2),
            inst(opcode::WORST_OF, 3),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PUSH_CONST, 2),
            inst(opcode::BEST_OF, 3),
            inst(opcode::STORE_LOCAL, 1),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[5.0, f64::NAN, 8.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(locals[0].is_nan(), "WORST_OF with NaN = {}", locals[0]);
        assert!(locals[1].is_nan(), "BEST_OF with NaN = {}", locals[1]);
    }

    #[test]
    fn jit_perf_helpers_propagate_nan() {
        let code = vec![
            inst(opcode::WORST_OF_PERF, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::BEST_OF_PERF, 0),
            inst(opcode::STORE_LOCAL, 1),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[120.0, f64::NAN],
                &[100.0, 100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            locals[0].is_nan(),
            "WORST_OF_PERF with NaN spot = {}",
            locals[0]
        );
        assert!(
            locals[1].is_nan(),
            "BEST_OF_PERF with NaN spot = {}",
            locals[1]
        );
    }

    #[test]
    fn jit_is_final() {
        let code = vec![inst(opcode::PUSH_IS_FINAL, 0), inst(opcode::STORE_LOCAL, 0)];
        let compiled = JitCompiledProgram::compile(&code, &[]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                true,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            (locals[0] - 1.0).abs() < 1e-10,
            "is_final=true => {}",
            locals[0]
        );

        locals[0] = 0.0;
        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(locals[0].abs() < 1e-10, "is_final=false => {}", locals[0]);
    }

    #[test]
    fn jit_price_opcode() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PRICE, 0),
            inst(opcode::STORE_LOCAL, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[1.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0, 250.0, 300.0],
                &[100.0, 100.0, 100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((locals[0] - 250.0).abs() < 1e-10, "PRICE(1)={}", locals[0]);
    }

    #[test]
    fn jit_checked_execution_reports_all_invalid_price_indices() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PRICE, 0),
            inst(opcode::STORE_LOCAL, 0),
        ];

        for invalid in [-1.0, f64::NAN, 0.5, 3.0, 1e18, f64::INFINITY] {
            let compiled = JitCompiledProgram::compile(&code, &[invalid]).unwrap();
            let mut locals = [0.0];
            let mut state = [];
            let mut pv = 0.0;

            let error = unsafe {
                compiled.execute_checked(
                    &[100.0, 250.0, 300.0],
                    &[100.0, 100.0, 100.0],
                    1.0,
                    1.0,
                    false,
                    1.0,
                    &mut locals,
                    &mut state,
                    &mut pv,
                )
            }
            .expect_err("invalid PRICE index must be rejected");

            assert!(
                error.to_string().contains("asset index"),
                "unexpected error for PRICE({invalid}): {error}"
            );
            assert_eq!(locals[0], 0.0);
            assert_eq!(pv, 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "JIT execution failed")]
    fn jit_legacy_execution_does_not_silently_accept_invalid_price_index() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PRICE, 0),
            inst(opcode::STORE_LOCAL, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[-1.0]).unwrap();
        let mut locals = [0.0];
        let mut state = [];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0, 250.0, 300.0],
                &[100.0, 100.0, 100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
    }

    #[test]
    fn jit_in_bounds_price_unaffected_by_bounds_check() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PRICE, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PRICE, 0),
            inst(opcode::STORE_LOCAL, 1),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[0.0, 2.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0, 250.0, 300.0],
                &[100.0, 100.0, 100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((locals[0] - 100.0).abs() < 1e-10, "PRICE(0)={}", locals[0]);
        assert!((locals[1] - 300.0).abs() < 1e-10, "PRICE(2)={}", locals[1]);
    }

    #[test]
    fn jit_instructions_after_redeem_compile_and_are_unreachable() {
        // `redeem 50` followed by `pay 999`: must compile (previously
        // panicked with "you have to fill your block before switching") and
        // the trailing pay must never execute, matching the interpreter's
        // early return on REDEEM.
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::REDEEM, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[50.0, 999.0])
            .expect("bytecode with instructions after REDEEM must compile");

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(matches!(result, ObservationResult::Redeemed));
        assert!(
            (pv - 50.0).abs() < 1e-10,
            "post-redeem pay must not run; expected pv=50.0, got {pv}"
        );
    }

    #[test]
    fn jit_instructions_after_skip_compile_and_are_unreachable() {
        let code = vec![
            inst(opcode::SKIP, 0),
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[999.0])
            .expect("bytecode with instructions after SKIP must compile");

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(matches!(result, ObservationResult::Skipped));
        assert!(
            pv.abs() < 1e-10,
            "post-skip pay must not run; expected pv=0.0, got {pv}"
        );
    }

    #[test]
    fn jit_sub_mul() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::SUB, 0),
            inst(opcode::PUSH_CONST, 2),
            inst(opcode::MUL, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[10.0, 3.0, 2.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((pv - 14.0).abs() < 1e-10, "expected 14.0, got {pv}");
    }

    #[test]
    fn jit_state_persistence() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::STORE_STATE, 0),
            inst(opcode::PUSH_STATE, 0),
            inst(opcode::PAY, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[77.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!((state[0] - 77.0).abs() < 1e-10, "state[0]={}", state[0]);
        assert!((pv - 77.0).abs() < 1e-10, "pv={}", pv);
    }

    #[test]
    fn jit_best_of_perf_test() {
        let code = vec![inst(opcode::BEST_OF_PERF, 0), inst(opcode::STORE_LOCAL, 0)];
        let compiled = JitCompiledProgram::compile(&code, &[]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[120.0, 90.0],
                &[100.0, 100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            (locals[0] - 1.2).abs() < 1e-10,
            "BEST_OF_PERF={}",
            locals[0]
        );
    }

    #[test]
    fn jit_worst_of_multiple() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::PUSH_CONST, 2),
            inst(opcode::WORST_OF, 3),
            inst(opcode::STORE_LOCAL, 0),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[5.0, 2.0, 8.0]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            (locals[0] - 2.0).abs() < 1e-10,
            "WORST_OF(5,2,8)={}",
            locals[0]
        );
    }

    #[test]
    fn jit_exp_log() {
        let code = vec![
            inst(opcode::PUSH_CONST, 0),
            inst(opcode::EXP, 0),
            inst(opcode::STORE_LOCAL, 0),
            inst(opcode::PUSH_CONST, 1),
            inst(opcode::LOG, 0),
            inst(opcode::STORE_LOCAL, 1),
        ];
        let compiled = JitCompiledProgram::compile(&code, &[1.0, std::f64::consts::E]).unwrap();

        let mut locals = vec![0.0; 4];
        let mut state = vec![0.0; 4];
        let mut pv = 0.0;

        unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(
            (locals[0] - std::f64::consts::E).abs() < 1e-10,
            "exp(1)={}",
            locals[0]
        );
        assert!((locals[1] - 1.0).abs() < 1e-10, "log(e)={}", locals[1]);
    }

    #[test]
    fn jit_rejects_malformed_control_flow_and_stack() {
        let underflow = JitCompiledProgram::compile(&[inst(opcode::ADD, 0)], &[])
            .err()
            .expect("stack underflow must be rejected");
        assert!(underflow.contains("stack underflow"), "{underflow}");

        let out_of_range = JitCompiledProgram::compile(&[inst(opcode::JUMP, 2)], &[])
            .err()
            .expect("out-of-range jump must be rejected");
        assert!(out_of_range.contains("out of range"), "{out_of_range}");

        let backward =
            JitCompiledProgram::compile(&[inst(opcode::PUSH_TRUE, 0), inst(opcode::JUMP, 0)], &[])
                .err()
                .expect("backward jump must be rejected");
        assert!(backward.contains("backward/self jump"), "{backward}");

        let stack_across_branch = JitCompiledProgram::compile(
            &[
                inst(opcode::PUSH_CONST, 0),
                inst(opcode::PUSH_TRUE, 0),
                inst(opcode::JUMP_FALSE, 3),
            ],
            &[1.0],
        )
        .err()
        .expect("stack values across branches must be rejected");
        assert!(
            stack_across_branch.contains("only statement-level branches"),
            "{stack_across_branch}"
        );

        let flagged = JitCompiledProgram::compile(&[(opcode::SKIP, 1, 0)], &[])
            .err()
            .expect("unknown flags must be rejected");
        assert!(
            flagged.contains("unsupported instruction flags"),
            "{flagged}"
        );

        let unknown = JitCompiledProgram::compile(&[(0xff, 0, 0)], &[])
            .err()
            .expect("unknown opcode must be rejected");
        assert!(unknown.contains("unknown opcode"), "{unknown}");
    }

    #[test]
    fn jit_empty_program_continues() {
        let compiled = JitCompiledProgram::compile(&[], &[]).unwrap();
        let mut locals = [];
        let mut state = [];
        let mut pv = 0.0;
        let result = unsafe {
            compiled.execute(
                &[100.0],
                &[100.0],
                1.0,
                1.0,
                false,
                1.0,
                &mut locals,
                &mut state,
                &mut pv,
            )
        };
        assert!(matches!(result, ObservationResult::Continue));
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn real_product_jit_matches_interpreter_across_branches() {
        let source = "\
product \"JIT Differential\"
    notional: 100
    maturity: 1.0

    underlyings
        SPX = asset(0)
        SX5E = asset(1)

    state
        coupon_paid: bool = false

    schedule semi_annual from 0.5 to 1.0
        let wof = worst_of(performances())

        if wof >= 1.0 and not is_final then
            pay notional * 0.05
            set coupon_paid = true

        if is_final then
            if coupon_paid then
                redeem notional + 5
            else
                redeem notional * wof
";
        let product = parse_and_compile(source).unwrap();
        let evaluator = JitProductEvaluator::compile(&product, 2, 0.03).unwrap();
        let initial = [100.0, 100.0];

        let paths = [
            vec![vec![100.0, 100.0], vec![110.0, 105.0], vec![90.0, 80.0]],
            vec![vec![100.0, 100.0], vec![90.0, 95.0], vec![120.0, 110.0]],
        ];

        let mut scratch = evaluator.new_scratch();
        for path in paths {
            let interpreted = evaluate_product(&product, &path, &initial, 2, 0.03).unwrap();
            let jitted = evaluator
                .evaluate_path_with_scratch(&path, &initial, &mut scratch)
                .unwrap();
            assert!(
                (jitted - interpreted).abs() < 1e-10,
                "JIT {jitted} != interpreter {interpreted}"
            );
        }
    }

    #[test]
    fn prepared_jit_evaluator_validates_paths_and_scratch() {
        let source = "\
product \"Forward\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        redeem price(0)
";
        let product = parse_and_compile(source).unwrap();
        let evaluator = JitProductEvaluator::compile(&product, 1, 0.0).unwrap();
        let path = vec![vec![100.0], vec![105.0]];

        assert_eq!(evaluator.evaluate_path(&path, &[100.0]).unwrap(), 105.0);

        let short_path = vec![vec![100.0]];
        let error = evaluator.evaluate_path(&short_path, &[100.0]).unwrap_err();
        assert!(error.to_string().contains("expected 2"));

        let other_product = parse_and_compile(
            "\
product \"Two Assets\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
        SX5E = asset(1)
    schedule annual from 1.0 to 1.0
        redeem notional
",
        )
        .unwrap();
        let other_evaluator = JitProductEvaluator::compile(&other_product, 1, 0.0).unwrap();
        let mut wrong_scratch = other_evaluator.new_scratch();
        let error = evaluator
            .evaluate_path_with_scratch(&path, &[100.0], &mut wrong_scratch)
            .unwrap_err();
        assert!(error.to_string().contains("different product evaluator"));

        let invalid_price_product = parse_and_compile(
            "\
product \"Invalid Price Index\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        redeem price(1)
",
        )
        .unwrap();
        let invalid_evaluator =
            JitProductEvaluator::compile(&invalid_price_product, 1, 0.0).unwrap();
        let error = invalid_evaluator
            .evaluate_path(&path, &[100.0])
            .unwrap_err();
        assert!(error.to_string().contains("asset index"));
    }

    #[test]
    fn prepared_jit_evaluator_rejects_non_finite_spots_before_branching() {
        let product = parse_and_compile(
            "\
product \"Finite fallback\"
    notional: 1
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        if price(0) > 0 then
            redeem 1
        else
            redeem 2
",
        )
        .unwrap();
        let evaluator = JitProductEvaluator::compile(&product, 1, 0.0).unwrap();

        let nan_path = vec![vec![100.0], vec![f64::NAN]];
        let error = evaluator.evaluate_path(&nan_path, &[100.0]).unwrap_err();
        assert!(
            error.to_string().contains("path_spots[1][0]"),
            "unexpected error: {error}"
        );

        let finite_path = vec![vec![100.0], vec![100.0]];
        let error = evaluator
            .evaluate_path(&finite_path, &[f64::INFINITY])
            .unwrap_err();
        assert!(
            error.to_string().contains("initial_spots[0]"),
            "unexpected error: {error}"
        );
    }
}
