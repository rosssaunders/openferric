//! Intermediate representation for compiled DSL products.
//!
//! The IR is the core data model that the evaluator walks. Variable access uses
//! slot indices (array indexing) rather than hash maps for O(1) access on the
//! hot MC path.

use crate::dsl::error::DslError;
use std::collections::HashSet;

/// Runtime value in the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Value {
    F64(f64),
    Bool(bool),
}

impl Value {
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::F64(v) => v,
            Self::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    #[inline]
    pub fn as_bool(self) -> bool {
        match self {
            Self::Bool(b) => b,
            Self::F64(v) => v != 0.0,
        }
    }
}

/// The type of an underlying asset, determining its market data shape and dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum UnderlyingType {
    #[default]
    Equity,
    Fx,
    Commodity,
    Rate,
}

/// Definition of a named underlying asset.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnderlyingDef {
    pub name: String,
    /// Index into the multi-asset spot/vol arrays.
    pub asset_index: usize,
    /// The type of this underlying (equity, FX, commodity, rate).
    #[serde(default)]
    pub underlying_type: UnderlyingType,
}

/// Definition of a state variable with its initial value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StateVarDef {
    pub name: String,
    /// Slot index for O(1) access during evaluation.
    pub slot: usize,
    pub initial: Value,
}

/// Binary arithmetic/comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinOp {
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

/// Unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Built-in function calls available in the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BuiltinFn {
    /// `worst_of(vec)` — minimum of a vector of f64.
    WorstOf,
    /// `best_of(vec)` — maximum of a vector of f64.
    BestOf,
    /// `performances()` — returns S_i(t)/S_i(0) for each underlying.
    Performances,
    /// `price(asset_index)` — current spot price of an underlying.
    Price,
    /// `min(a, b)`.
    Min,
    /// `max(a, b)`.
    Max,
    /// `abs(x)`.
    Abs,
    /// `exp(x)`.
    Exp,
    /// `log(x)`.
    Log,
}

/// Expression in the IR.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Expr {
    /// Literal constant.
    Literal(Value),
    /// Reference to a local variable by slot index.
    LocalVar(usize),
    /// Reference to a state variable by slot index.
    StateVar(usize),
    /// Reference to product-level `notional`.
    Notional,
    /// Reference to the current observation date (year fraction).
    ObservationDate,
    /// Whether the current observation is the final one in the schedule.
    IsFinal,
    /// Binary operation.
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Unary operation.
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
    /// Built-in function call.
    Call { func: BuiltinFn, args: Vec<Expr> },
}

/// Statement in the IR (executed per observation date per path).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Statement {
    /// `let var = expr` — bind a local variable.
    Let { slot: usize, expr: Expr },
    /// Conditional execution.
    If {
        condition: Expr,
        then_body: Vec<Statement>,
        else_body: Vec<Statement>,
    },
    /// `pay amount` — record a cashflow at the current observation date.
    Pay { amount: Expr },
    /// `redeem amount` — record a final payment and terminate the product.
    Redeem { amount: Expr },
    /// `set state_var = expr` — mutate a state variable.
    SetState { slot: usize, expr: Expr },
    /// `skip` — skip remaining observation dates (early exit without payment).
    Skip,
}

/// A schedule of observation dates with associated logic.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Schedule {
    /// Pre-computed observation dates in year fractions.
    pub dates: Vec<f64>,
    /// Statements executed at each observation date.
    pub body: Vec<Statement>,
}

/// The fully compiled product ready for evaluation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompiledProduct {
    pub name: String,
    pub notional: f64,
    pub maturity: f64,
    pub num_underlyings: usize,
    pub underlyings: Vec<UnderlyingDef>,
    /// State variables with slot-indexed access.
    pub state_vars: Vec<StateVarDef>,
    /// Named constants defined at product level.
    pub constants: Vec<(String, Value)>,
    /// Observation schedules with their logic.
    pub schedules: Vec<Schedule>,
}

impl CompiledProduct {
    /// Validate a compiled product before constructing an evaluation backend.
    ///
    /// `CompiledProduct` is public and deserializable, so callers are not
    /// required to obtain it through the DSL compiler. Evaluation backends
    /// rely on dense slot-indexed storage and finite product data; this method
    /// establishes those invariants for hand-built and deserialized products.
    pub fn validate(&self) -> Result<(), DslError> {
        fn invalid(message: impl Into<String>) -> DslError {
            DslError::EvalError(format!("invalid compiled product: {}", message.into()))
        }

        fn validate_value(value: Value, context: &str) -> Result<(), DslError> {
            if let Value::F64(value) = value
                && !value.is_finite()
            {
                return Err(DslError::EvalError(format!(
                    "invalid compiled product: {context} must be finite"
                )));
            }
            Ok(())
        }

        fn collect_local_slots(
            statements: &[Statement],
            slots: &mut HashSet<usize>,
        ) -> Result<(), DslError> {
            for statement in statements {
                match statement {
                    Statement::Let { slot, .. } => {
                        if *slot > u16::MAX as usize {
                            return Err(DslError::CompileError {
                                message: format!(
                                    "local variable slot {slot} exceeds the bytecode operand limit of {}",
                                    u16::MAX
                                ),
                                span: None,
                            });
                        }
                        // A slot may be reused in mutually exclusive branches.
                        // The storage invariant is that every referenced slot
                        // belongs to the dense allocated range, not that each
                        // slot has exactly one syntactic declaration.
                        slots.insert(*slot);
                    }
                    Statement::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        collect_local_slots(then_body, slots)?;
                        collect_local_slots(else_body, slots)?;
                    }
                    _ => {}
                }
            }
            Ok(())
        }

        fn validate_expr(
            expr: &Expr,
            assigned_local_slots: &HashSet<usize>,
            num_state_vars: usize,
            context: &str,
        ) -> Result<(), DslError> {
            match expr {
                Expr::Literal(value) => validate_value(*value, context),
                Expr::LocalVar(slot) => {
                    if !assigned_local_slots.contains(slot) {
                        return Err(DslError::EvalError(format!(
                            "invalid compiled product: local variable slot {slot} is read before assignment or outside its declaring scope"
                        )));
                    }
                    Ok(())
                }
                Expr::StateVar(slot) => {
                    if *slot >= num_state_vars {
                        return Err(DslError::EvalError(format!(
                            "invalid compiled product: state variable slot {slot} is outside the declared range 0..{num_state_vars}"
                        )));
                    }
                    Ok(())
                }
                Expr::BinOp { lhs, rhs, .. } => {
                    validate_expr(lhs, assigned_local_slots, num_state_vars, context)?;
                    validate_expr(rhs, assigned_local_slots, num_state_vars, context)
                }
                Expr::UnaryOp { operand, .. } => {
                    validate_expr(operand, assigned_local_slots, num_state_vars, context)
                }
                Expr::Call { args, .. } => {
                    for argument in args {
                        validate_expr(argument, assigned_local_slots, num_state_vars, context)?;
                    }
                    Ok(())
                }
                Expr::Notional | Expr::ObservationDate | Expr::IsFinal => Ok(()),
            }
        }

        fn validate_statements(
            statements: &[Statement],
            assigned_local_slots: &mut HashSet<usize>,
            num_state_vars: usize,
            schedule_index: usize,
        ) -> Result<(), DslError> {
            for statement in statements {
                match statement {
                    Statement::Let { slot, expr } => {
                        validate_expr(
                            expr,
                            assigned_local_slots,
                            num_state_vars,
                            &format!("schedule {schedule_index} local expression"),
                        )?;
                        assigned_local_slots.insert(*slot);
                    }
                    Statement::If {
                        condition,
                        then_body,
                        else_body,
                    } => {
                        validate_expr(
                            condition,
                            assigned_local_slots,
                            num_state_vars,
                            &format!("schedule {schedule_index} condition"),
                        )?;
                        // Branch-local `let` bindings are lexical: outer
                        // assignments are visible inside a branch, but new
                        // bindings do not escape it.
                        let mut then_assignments = assigned_local_slots.clone();
                        validate_statements(
                            then_body,
                            &mut then_assignments,
                            num_state_vars,
                            schedule_index,
                        )?;
                        let mut else_assignments = assigned_local_slots.clone();
                        validate_statements(
                            else_body,
                            &mut else_assignments,
                            num_state_vars,
                            schedule_index,
                        )?;
                    }
                    Statement::Pay { amount } | Statement::Redeem { amount } => {
                        validate_expr(
                            amount,
                            assigned_local_slots,
                            num_state_vars,
                            &format!("schedule {schedule_index} payment expression"),
                        )?;
                    }
                    Statement::SetState { slot, expr } => {
                        if *slot >= num_state_vars {
                            return Err(DslError::EvalError(format!(
                                "invalid compiled product: state variable slot {slot} is outside the declared range 0..{num_state_vars}"
                            )));
                        }
                        validate_expr(
                            expr,
                            assigned_local_slots,
                            num_state_vars,
                            &format!("schedule {schedule_index} state expression"),
                        )?;
                    }
                    Statement::Skip => {}
                }
            }
            Ok(())
        }

        if !self.notional.is_finite() {
            return Err(invalid("notional must be finite"));
        }
        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err(invalid("maturity must be positive and finite"));
        }
        if self.num_underlyings == 0 {
            return Err(invalid("num_underlyings must be greater than zero"));
        }

        // Products without an underlying declaration (for example, a fixed
        // cashflow) retain one simulated market dimension. Otherwise every
        // declared underlying must map one-to-one to a product dimension.
        if self.underlyings.is_empty() && self.num_underlyings != 1 {
            return Err(invalid(format!(
                "products without underlying definitions require exactly one market dimension, not {}",
                self.num_underlyings
            )));
        }
        if !self.underlyings.is_empty() && self.underlyings.len() != self.num_underlyings {
            return Err(invalid(format!(
                "num_underlyings is {}, but {} underlying definitions were provided",
                self.num_underlyings,
                self.underlyings.len()
            )));
        }

        let mut underlying_names = HashSet::with_capacity(self.underlyings.len());
        let mut asset_indices = HashSet::with_capacity(self.underlyings.len());
        for underlying in &self.underlyings {
            if !underlying_names.insert(underlying.name.as_str()) {
                return Err(invalid(format!(
                    "underlying name '{}' is declared more than once",
                    underlying.name
                )));
            }
            if underlying.asset_index >= self.num_underlyings {
                return Err(invalid(format!(
                    "underlying '{}' has asset index {}, outside the declared range 0..{}",
                    underlying.name, underlying.asset_index, self.num_underlyings
                )));
            }
            if !asset_indices.insert(underlying.asset_index) {
                return Err(invalid(format!(
                    "asset index {} is assigned to more than one underlying",
                    underlying.asset_index
                )));
            }
        }

        let mut state_names = HashSet::with_capacity(self.state_vars.len());
        for (expected_slot, state_var) in self.state_vars.iter().enumerate() {
            if state_var.slot != expected_slot {
                return Err(invalid(format!(
                    "state variable '{}' has slot {}, expected dense slot {expected_slot}",
                    state_var.name, state_var.slot
                )));
            }
            if !state_names.insert(state_var.name.as_str()) {
                return Err(invalid(format!(
                    "state variable name '{}' is declared more than once",
                    state_var.name
                )));
            }
            validate_value(
                state_var.initial,
                &format!("initial value for state variable '{}'", state_var.name),
            )?;
        }

        let mut constant_names = HashSet::with_capacity(self.constants.len());
        for (name, value) in &self.constants {
            if !constant_names.insert(name.as_str()) {
                return Err(invalid(format!(
                    "constant '{name}' is declared more than once"
                )));
            }
            validate_value(*value, &format!("constant '{name}'"))?;
        }

        for (schedule_index, schedule) in self.schedules.iter().enumerate() {
            if schedule.dates.is_empty() {
                return Err(invalid(format!("schedule {schedule_index} has no dates")));
            }
            for (date_index, &date) in schedule.dates.iter().enumerate() {
                if !date.is_finite() {
                    return Err(invalid(format!(
                        "schedule {schedule_index} date {date_index} must be finite"
                    )));
                }
                if date < 0.0 || date > self.maturity {
                    return Err(invalid(format!(
                        "schedule {schedule_index} date {date} is outside [0, {}]",
                        self.maturity
                    )));
                }
                if date_index > 0 && date <= schedule.dates[date_index - 1] {
                    return Err(invalid(format!(
                        "schedule {schedule_index} dates must be strictly increasing"
                    )));
                }
            }

            let mut local_slots = HashSet::new();
            collect_local_slots(&schedule.body, &mut local_slots)?;
            if let Some(max_slot) = local_slots.iter().copied().max()
                && max_slot + 1 != local_slots.len()
            {
                return Err(invalid(format!(
                    "schedule {schedule_index} local variable slots must form a dense range starting at zero"
                )));
            }
            let mut assigned_local_slots = HashSet::new();
            validate_statements(
                &schedule.body,
                &mut assigned_local_slots,
                self.state_vars.len(),
                schedule_index,
            )?;
        }

        Ok(())
    }

    /// Returns the total number of local variable slots needed.
    /// This is computed by walking the statement tree and finding the max slot index.
    pub fn max_local_slots(&self) -> usize {
        fn max_slot_in_stmts(stmts: &[Statement]) -> usize {
            let mut max = 0;
            for stmt in stmts {
                match stmt {
                    Statement::Let { slot, .. } => max = max.max(slot.saturating_add(1)),
                    Statement::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        max = max.max(max_slot_in_stmts(then_body));
                        max = max.max(max_slot_in_stmts(else_body));
                    }
                    _ => {}
                }
            }
            max
        }

        self.schedules
            .iter()
            .map(|s| max_slot_in_stmts(&s.body))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_product() -> CompiledProduct {
        CompiledProduct {
            name: "Validated".to_string(),
            notional: 100.0,
            maturity: 1.0,
            num_underlyings: 1,
            underlyings: vec![UnderlyingDef {
                name: "SPX".to_string(),
                asset_index: 0,
                underlying_type: UnderlyingType::Equity,
            }],
            state_vars: vec![StateVarDef {
                name: "called".to_string(),
                slot: 0,
                initial: Value::Bool(false),
            }],
            constants: vec![("coupon".to_string(), Value::F64(0.05))],
            schedules: vec![Schedule {
                dates: vec![0.5, 1.0],
                body: vec![
                    Statement::Let {
                        slot: 0,
                        expr: Expr::Literal(Value::F64(1.0)),
                    },
                    Statement::Pay {
                        amount: Expr::LocalVar(0),
                    },
                    Statement::SetState {
                        slot: 0,
                        expr: Expr::Literal(Value::Bool(true)),
                    },
                ],
            }],
        }
    }

    #[test]
    fn value_conversions() {
        assert_eq!(Value::F64(3.125).as_f64(), 3.125);
        assert_eq!(Value::Bool(true).as_f64(), 1.0);
        assert_eq!(Value::Bool(false).as_f64(), 0.0);
        assert!(Value::Bool(true).as_bool());
        assert!(!Value::Bool(false).as_bool());
        assert!(Value::F64(1.0).as_bool());
        assert!(!Value::F64(0.0).as_bool());
    }

    #[test]
    fn max_local_slots_counts_correctly() {
        let product = CompiledProduct {
            name: "test".to_string(),
            notional: 1.0,
            maturity: 1.0,
            num_underlyings: 1,
            underlyings: vec![],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![0.5, 1.0],
                body: vec![
                    Statement::Let {
                        slot: 0,
                        expr: Expr::Literal(Value::F64(1.0)),
                    },
                    Statement::Let {
                        slot: 2,
                        expr: Expr::Literal(Value::F64(2.0)),
                    },
                ],
            }],
        };
        assert_eq!(product.max_local_slots(), 3);
    }

    #[test]
    fn compiled_product_validation_accepts_dense_slots_reused_across_branches() {
        let mut product = valid_product();
        product.schedules[0].body = vec![Statement::If {
            condition: Expr::Literal(Value::Bool(true)),
            then_body: vec![
                Statement::Let {
                    slot: 0,
                    expr: Expr::Literal(Value::F64(1.0)),
                },
                Statement::Pay {
                    amount: Expr::LocalVar(0),
                },
            ],
            else_body: vec![
                Statement::Let {
                    slot: 0,
                    expr: Expr::Literal(Value::F64(2.0)),
                },
                Statement::Pay {
                    amount: Expr::LocalVar(0),
                },
            ],
        }];

        product.validate().unwrap();
    }

    #[test]
    fn compiled_product_validation_rejects_non_finite_values() {
        let mut product = valid_product();
        product.notional = f64::NAN;
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("notional")
        );

        let mut product = valid_product();
        product.state_vars[0].initial = Value::F64(f64::INFINITY);
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("initial value")
        );

        let mut product = valid_product();
        product.schedules[0].body[0] = Statement::Let {
            slot: 0,
            expr: Expr::Literal(Value::F64(f64::NAN)),
        };
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("local expression")
        );
    }

    #[test]
    fn compiled_product_validation_rejects_invalid_schedule_dates() {
        for dates in [
            vec![f64::NAN],
            vec![-0.1, 0.5],
            vec![0.75, 0.5],
            vec![0.5, 1.5],
            Vec::new(),
        ] {
            let mut product = valid_product();
            product.schedules[0].dates = dates;
            assert!(product.validate().is_err());
        }
    }

    #[test]
    fn compiled_product_validation_rejects_inconsistent_dimensions_and_slots() {
        let mut product = valid_product();
        product.num_underlyings = 2;
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("underlying definitions")
        );

        let mut product = valid_product();
        product.underlyings.clear();
        product.num_underlyings = 2;
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("exactly one market dimension")
        );

        let mut product = valid_product();
        product.underlyings[0].asset_index = 1;
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("asset index")
        );

        let mut product = valid_product();
        product.state_vars[0].slot = 1;
        assert!(
            product
                .validate()
                .unwrap_err()
                .to_string()
                .contains("dense slot")
        );
    }

    #[test]
    fn deserialized_product_rejects_undeclared_local_and_state_references() {
        let mut product = valid_product();
        product.schedules[0].body = vec![Statement::Pay {
            amount: Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(Expr::LocalVar(1)),
                rhs: Box::new(Expr::StateVar(1)),
            },
        }];

        let json = serde_json::to_string(&product).unwrap();
        let deserialized: CompiledProduct = serde_json::from_str(&json).unwrap();
        let error = deserialized.validate().unwrap_err().to_string();
        assert!(
            error.contains("local variable slot 1"),
            "unexpected error: {error}"
        );

        let mut product = valid_product();
        product.schedules[0].body = vec![Statement::SetState {
            slot: 1,
            expr: Expr::Literal(Value::Bool(true)),
        }];
        let error = product.validate().unwrap_err().to_string();
        assert!(
            error.contains("state variable slot 1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn compiled_product_validation_enforces_local_assignment_and_branch_scope() {
        let mut read_before_assignment = valid_product();
        read_before_assignment.schedules[0].body = vec![
            Statement::Pay {
                amount: Expr::LocalVar(0),
            },
            Statement::Let {
                slot: 0,
                expr: Expr::Literal(Value::F64(1.0)),
            },
        ];
        let error = read_before_assignment.validate().unwrap_err().to_string();
        assert!(
            error.contains("before assignment"),
            "unexpected error: {error}"
        );

        let mut branch_escape = valid_product();
        branch_escape.schedules[0].body = vec![
            Statement::If {
                condition: Expr::Literal(Value::Bool(true)),
                then_body: vec![Statement::Let {
                    slot: 0,
                    expr: Expr::Literal(Value::F64(1.0)),
                }],
                else_body: vec![],
            },
            Statement::Pay {
                amount: Expr::LocalVar(0),
            },
        ];
        let error = branch_escape.validate().unwrap_err().to_string();
        assert!(
            error.contains("outside its declaring scope"),
            "unexpected error: {error}"
        );
    }
}
