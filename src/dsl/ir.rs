//! Intermediate representation for compiled DSL products.
//!
//! The IR is the core data model that the evaluator walks. Variable access uses
//! slot indices (array indexing) rather than hash maps for O(1) access on the
//! hot MC path.

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

/// Definition of a named underlying asset.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnderlyingDef {
    pub name: String,
    /// Index into the multi-asset spot/vol arrays.
    pub asset_index: usize,
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
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Built-in function call.
    Call {
        func: BuiltinFn,
        args: Vec<Expr>,
    },
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
    /// Returns the total number of local variable slots needed.
    /// This is computed by walking the statement tree and finding the max slot index.
    pub fn max_local_slots(&self) -> usize {
        fn max_slot_in_stmts(stmts: &[Statement]) -> usize {
            let mut max = 0;
            for stmt in stmts {
                match stmt {
                    Statement::Let { slot, .. } => max = max.max(*slot + 1),
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

    #[test]
    fn value_conversions() {
        assert_eq!(Value::F64(3.14).as_f64(), 3.14);
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
}
