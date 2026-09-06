use crate::data::data_type;
use crate::dsl::{ProductDef, SymbolTable};
use openferric_core::dsl::{analysis, ast, ir, lexer};
use pyo3::prelude::*;

data_type!(Span, openferric_core::dsl::error::Span, {});
data_type!(ProductItem, ast::ProductItem, {});
data_type!(UnderlyingDecl, ast::UnderlyingDecl, {});
data_type!(StateDecl, ast::StateDecl, {});
data_type!(ScheduleDef, ast::ScheduleDef, {});
data_type!(ScheduleFreq, ast::ScheduleFreq, {
    fn period(&self) -> f64 {
        self.inner.period()
    }
    fn generate_dates(&self, start: f64, end: f64) -> PyResult<Vec<f64>> {
        self.inner
            .generate_dates(start, end)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
});
data_type!(AstExpr, ast::AstExpr, {});
data_type!(AstExprKind, ast::AstExprKind, {});
data_type!(AstStatement, ast::AstStatement, {});
data_type!(AstStatementKind, ast::AstStatementKind, {});
data_type!(AstBinOp, ast::AstBinOp, {});
data_type!(AstUnaryOp, ast::AstUnaryOp, {});
data_type!(Token, lexer::Token, {});
data_type!(TokenKind, lexer::TokenKind, {});
data_type!(Value, ir::Value, {
    fn as_f64(&self) -> f64 {
        self.inner.as_f64()
    }
    fn as_bool(&self) -> bool {
        self.inner.as_bool()
    }
});
data_type!(UnderlyingType, ir::UnderlyingType, {});
data_type!(UnderlyingDef, ir::UnderlyingDef, {});
data_type!(StateVarDef, ir::StateVarDef, {});
data_type!(BinOp, ir::BinOp, {});
data_type!(UnaryOp, ir::UnaryOp, {});
data_type!(BuiltinFn, ir::BuiltinFn, {});
data_type!(Expr, ir::Expr, {});
data_type!(Statement, ir::Statement, {});
data_type!(Schedule, ir::Schedule, {});
data_type!(CompletionCandidate, analysis::CompletionCandidate, {});
data_type!(
    CompletionCandidateKind,
    analysis::CompletionCandidateKind,
    {}
);
data_type!(DiagnosticInfo, analysis::DiagnosticInfo, {});
data_type!(DiagnosticSeverity, analysis::DiagnosticSeverity, {});
data_type!(HoverInfo, analysis::HoverInfo, {});
data_type!(SemanticTokenData, analysis::SemanticTokenData, {});
data_type!(SymbolKind, analysis::SymbolKind, {});
data_type!(SymbolScope, analysis::SymbolScope, {});

macro_rules! symbol_record {
    ($name:ident) => {
        #[pyclass(module = "openferric", from_py_object)]
        #[derive(Clone)]
        pub struct $name {
            pub(crate) inner: analysis::$name,
        }
        #[pymethods]
        impl $name {
            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                crate::helpers::to_python(py, &self.inner)
            }
            fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
                self.to_dict(py)?
                    .bind(py)
                    .get_item(name)
                    .map(Bound::unbind)
                    .map_err(|_| pyo3::exceptions::PyAttributeError::new_err(name.to_owned()))
            }
        }
    };
}
symbol_record!(SymbolInfo);
symbol_record!(SymbolRef);

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ObservationResult {
    Continue,
    Redeemed,
    Skipped,
}

#[pyclass(module = "openferric", from_py_object, get_all, set_all)]
#[derive(Clone, Copy)]
pub struct Cashflow {
    time: f64,
    amount: f64,
}
#[pymethods]
impl Cashflow {
    #[new]
    fn new(time: f64, amount: f64) -> Self {
        Self { time, amount }
    }
}

#[pyfunction]
fn tokenize(source: &str) -> PyResult<Vec<Token>> {
    lexer::tokenize(source)
        .map(|tokens| tokens.into_iter().map(|inner| Token { inner }).collect())
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}
#[pyfunction]
fn parse(tokens: Vec<Token>) -> PyResult<ProductDef> {
    openferric_core::dsl::parser::parse(tokens.into_iter().map(|token| token.inner).collect())
        .map(|inner| ProductDef { inner })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}
#[pyfunction]
fn build_symbol_table(ast: &ProductDef, source: &str) -> SymbolTable {
    SymbolTable {
        inner: analysis::build_symbol_table(&ast.inner, source),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! classes { ($($name:ty),+) => { $(module.add_class::<$name>()?;)+ } }
    classes!(
        Span,
        ProductItem,
        UnderlyingDecl,
        StateDecl,
        ScheduleDef,
        ScheduleFreq,
        AstExpr,
        AstExprKind,
        AstStatement,
        AstStatementKind,
        AstBinOp,
        AstUnaryOp,
        Token,
        TokenKind,
        Value,
        UnderlyingType,
        UnderlyingDef,
        StateVarDef,
        BinOp,
        UnaryOp,
        BuiltinFn,
        Expr,
        Statement,
        Schedule,
        CompletionCandidate,
        CompletionCandidateKind,
        DiagnosticInfo,
        DiagnosticSeverity,
        HoverInfo,
        SemanticTokenData,
        SymbolKind,
        SymbolScope,
        SymbolInfo,
        SymbolRef,
        ObservationResult,
        Cashflow
    );
    module.add_function(wrap_pyfunction!(tokenize, module)?)?;
    module.add_function(wrap_pyfunction!(parse, module)?)?;
    module.add_function(wrap_pyfunction!(build_symbol_table, module)?)?;
    module.add("MAX_SCHEDULE_DATES", ast::MAX_SCHEDULE_DATES)?;
    macro_rules! constants { ($($name:ident),+) => { $(module.add(stringify!($name), analysis::$name)?;)+ } }
    constants!(
        TOKEN_COMMENT,
        TOKEN_ENUM_MEMBER,
        TOKEN_FUNCTION,
        TOKEN_KEYWORD,
        TOKEN_NUMBER,
        TOKEN_OPERATOR,
        TOKEN_STRING,
        TOKEN_VARIABLE
    );
    Ok(())
}
