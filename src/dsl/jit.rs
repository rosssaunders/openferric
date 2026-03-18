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

use super::eval::ObservationResult;

// Re-use opcode constants from eval.
mod opcode {
    pub const JUMP: u8 = 0x01;
    pub const JUMP_FALSE: u8 = 0x02;
    pub const SKIP: u8 = 0x03;

    pub const PUSH_CONST: u8 = 0x10;
    pub const PUSH_LOCAL: u8 = 0x11;
    pub const PUSH_STATE: u8 = 0x12;
    pub const PUSH_NOTIONAL: u8 = 0x13;
    pub const PUSH_DATE: u8 = 0x14;
    pub const PUSH_IS_FINAL: u8 = 0x15;
    pub const PUSH_TRUE: u8 = 0x16;
    pub const PUSH_FALSE: u8 = 0x17;

    pub const STORE_LOCAL: u8 = 0x20;
    pub const STORE_STATE: u8 = 0x21;

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

    pub const EQ: u8 = 0x40;
    pub const NE: u8 = 0x41;
    pub const LT: u8 = 0x42;
    pub const LE: u8 = 0x43;
    pub const GT: u8 = 0x44;
    pub const GE: u8 = 0x45;
    pub const AND: u8 = 0x46;
    pub const OR: u8 = 0x47;
    pub const NOT: u8 = 0x48;

    pub const PAY: u8 = 0x50;
    pub const REDEEM: u8 = 0x51;
    pub const PRICE: u8 = 0x52;
    pub const WORST_OF: u8 = 0x53;
    pub const BEST_OF: u8 = 0x54;
    pub const WORST_OF_PERF: u8 = 0x55;
    pub const BEST_OF_PERF: u8 = 0x56;
}

/// Packed bytecode instruction. Must match the layout in `eval.rs`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Instruction {
    opcode: u8,
    flags: u8,
    operand: u16,
}

const _: () = assert!(std::mem::size_of::<Instruction>() == 4);

const RESULT_CONTINUE: u8 = 0;
const RESULT_REDEEMED: u8 = 1;
const RESULT_SKIPPED: u8 = 2;

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

    /// Execute the compiled program.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `spots`, `initial_spots`, `locals`,
    /// `state`, and `pv` are valid and appropriately sized.
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
            RESULT_REDEEMED => ObservationResult::Redeemed,
            RESULT_SKIPPED => ObservationResult::Skipped,
            _ => ObservationResult::Continue,
        }
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

        let mut stack = CompileStack::new();
        let mem = MemFlags::new();
        let mut block_terminated = false;

        let mut pc = 0usize;
        while pc < instructions.len() {
            if pc > 0 {
                if let Some(&target_block) = block_map.get(&pc) {
                    if !block_terminated {
                        builder.ins().jump(target_block, &[]);
                    }
                    builder.switch_to_block(target_block);
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
                    block_terminated = true;
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
                }
                opcode::SKIP => {
                    builder.ins().jump(return_skipped_block, &[]);
                    block_terminated = true;
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
                }

                opcode::PRICE => {
                    let idx_f64 = stack.pop();
                    let idx_i64 = builder.ins().fcvt_to_uint(I64, idx_f64);
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
                    block_terminated = true;
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
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
                    // The brif terminates the current block; we immediately
                    // switch to the fall-through block so block_terminated
                    // stays false.
                    builder.switch_to_block(true_block);
                    block_terminated = false;
                }

                _ => {
                    return Err(format!("unknown opcode 0x{:02x} at pc {pc}", inst.opcode));
                }
            }

            pc += 1;
        }

        if !block_terminated {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
