use openferric_core::core::{ExecutionPolicy as CoreExecutionPolicy, PricingEngine};
use openferric_core::dsl::{self as core_dsl, analysis, ast, eval, ir, lexer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::{Greeks, PricingResult};
use crate::helpers::{from_python, to_python};
use crate::market::Market;

fn error(value: impl ToString) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn checked_offset(source: &str, offset: usize) -> PyResult<usize> {
    if offset > source.len() || !source.is_char_boundary(offset) {
        return Err(error(
            "offset must be a UTF-8 byte boundary within the source",
        ));
    }
    Ok(offset)
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPolicy {
    Auto,
    Scalar,
    Simd,
    Parallel,
    Gpu,
    Jit,
}

impl ExecutionPolicy {
    pub(crate) fn from_core(value: CoreExecutionPolicy) -> Self {
        match value {
            CoreExecutionPolicy::Auto => Self::Auto,
            CoreExecutionPolicy::Scalar => Self::Scalar,
            CoreExecutionPolicy::Simd => Self::Simd,
            CoreExecutionPolicy::Parallel => Self::Parallel,
            CoreExecutionPolicy::Gpu => Self::Gpu,
            CoreExecutionPolicy::Jit => Self::Jit,
        }
    }

    pub(crate) fn to_core(self) -> CoreExecutionPolicy {
        match self {
            Self::Auto => CoreExecutionPolicy::Auto,
            Self::Scalar => CoreExecutionPolicy::Scalar,
            Self::Simd => CoreExecutionPolicy::Simd,
            Self::Parallel => CoreExecutionPolicy::Parallel,
            Self::Gpu => CoreExecutionPolicy::Gpu,
            Self::Jit => CoreExecutionPolicy::Jit,
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    Scalar = 0,
    Simd = 1,
    Parallel = 2,
    Gpu = 3,
    Jit = 4,
}

impl From<openferric_core::core::ExecutionBackend> for ExecutionBackend {
    fn from(value: openferric_core::core::ExecutionBackend) -> Self {
        use openferric_core::core::ExecutionBackend as Backend;
        match value {
            Backend::Scalar => Self::Scalar,
            Backend::Simd => Self::Simd,
            Backend::Parallel => Self::Parallel,
            Backend::Gpu => Self::Gpu,
            Backend::Jit => Self::Jit,
        }
    }
}

#[pymethods]
impl ExecutionBackend {
    fn diagnostic_code(&self) -> f64 {
        *self as u8 as f64
    }

    #[staticmethod]
    fn from_diagnostic_code(code: f64) -> Option<Self> {
        openferric_core::core::ExecutionBackend::from_diagnostic_code(code).map(Self::from)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CompiledProduct {
    pub(crate) inner: ir::CompiledProduct,
}

#[pymethods]
impl CompiledProduct {
    #[new]
    fn new(source: &str) -> PyResult<Self> {
        parse_and_compile(source)
    }

    #[staticmethod]
    fn from_json(serialized: &str) -> PyResult<Self> {
        let inner: ir::CompiledProduct = serde_json::from_str(serialized).map_err(error)?;
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_dict(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: ir::CompiledProduct = from_python(value)?;
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(error)
    }

    fn max_local_slots(&self) -> usize {
        self.inner.max_local_slots()
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity
    }

    #[getter]
    fn num_underlyings(&self) -> usize {
        self.inner.num_underlyings
    }

    #[getter]
    fn underlyings(&self) -> Vec<crate::dsl_data::UnderlyingDef> {
        self.inner
            .underlyings
            .iter()
            .cloned()
            .map(|inner| crate::dsl_data::UnderlyingDef { inner })
            .collect()
    }

    #[getter]
    fn state_vars(&self) -> Vec<crate::dsl_data::StateVarDef> {
        self.inner
            .state_vars
            .iter()
            .cloned()
            .map(|inner| crate::dsl_data::StateVarDef { inner })
            .collect()
    }

    #[getter]
    fn constants(&self) -> Vec<(String, crate::dsl_data::Value)> {
        self.inner
            .constants
            .iter()
            .map(|(name, value)| (name.clone(), crate::dsl_data::Value { inner: *value }))
            .collect()
    }

    #[getter]
    fn schedules(&self) -> Vec<crate::dsl_data::Schedule> {
        self.inner
            .schedules
            .iter()
            .cloned()
            .map(|inner| crate::dsl_data::Schedule { inner })
            .collect()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DslProduct {
    pub(crate) inner: core_dsl::DslProduct,
}

#[pymethods]
impl DslProduct {
    #[new]
    fn new(product: &CompiledProduct) -> Self {
        Self {
            inner: core_dsl::DslProduct::new(product.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_source(source: &str) -> PyResult<Self> {
        Ok(Self::new(&parse_and_compile(source)?))
    }

    #[getter]
    fn product(&self) -> CompiledProduct {
        CompiledProduct {
            inner: self.inner.product.clone(),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AssetMarketData {
    inner: core_dsl::AssetMarketData,
}

#[pymethods]
impl AssetMarketData {
    #[staticmethod]
    #[pyo3(signature = (spot, vol, dividend_yield=0.0))]
    fn equity(spot: f64, vol: f64, dividend_yield: f64) -> Self {
        Self {
            inner: core_dsl::AssetMarketData::Equity {
                spot,
                vol,
                dividend_yield,
            },
        }
    }

    #[staticmethod]
    fn fx(spot: f64, vol: f64, domestic_rate: f64, foreign_rate: f64) -> Self {
        Self {
            inner: core_dsl::AssetMarketData::Fx {
                spot,
                vol,
                domestic_rate,
                foreign_rate,
            },
        }
    }

    #[staticmethod]
    fn commodity(spot: f64, vol: f64, convenience_yield: f64, kappa: f64, mu: f64) -> Self {
        Self {
            inner: core_dsl::AssetMarketData::Commodity {
                spot,
                vol,
                convenience_yield,
                kappa,
                mu,
            },
        }
    }

    #[staticmethod]
    fn rate(initial_rate: f64, vol: f64, mean_reversion: f64, long_run_mean: f64) -> Self {
        Self {
            inner: core_dsl::AssetMarketData::Rate {
                initial_rate,
                vol,
                mean_reversion,
                long_run_mean,
            },
        }
    }

    fn initial_value(&self) -> f64 {
        self.inner.initial_value()
    }

    fn vol(&self) -> f64 {
        self.inner.vol()
    }

    fn with_spot_bump(&self, amount: f64) -> Self {
        Self {
            inner: self.inner.with_spot_bump(amount),
        }
    }

    fn with_vol_bump(&self, amount: f64) -> Self {
        Self {
            inner: self.inner.with_vol_bump(amount),
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MultiAssetMarket {
    inner: core_dsl::MultiAssetMarket,
}

#[pymethods]
impl MultiAssetMarket {
    #[new]
    fn new(assets: Vec<AssetMarketData>, correlation: Vec<Vec<f64>>, rate: f64) -> PyResult<Self> {
        let inner = core_dsl::MultiAssetMarket {
            assets: assets.into_iter().map(|asset| asset.inner).collect(),
            correlation,
            rate,
        };
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (spot, vol, rate, dividend_yield=0.0))]
    fn single(spot: f64, vol: f64, rate: f64, dividend_yield: f64) -> PyResult<Self> {
        let inner = core_dsl::MultiAssetMarket::single(spot, vol, rate, dividend_yield);
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn assets(&self) -> Vec<AssetMarketData> {
        self.inner
            .assets
            .iter()
            .cloned()
            .map(|inner| AssetMarketData { inner })
            .collect()
    }

    #[getter]
    fn correlation(&self) -> Vec<Vec<f64>> {
        self.inner.correlation.clone()
    }

    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate
    }

    fn initial_spots(&self) -> Vec<f64> {
        self.inner.initial_spots()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(error)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    #[staticmethod]
    fn from_dict(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: core_dsl::MultiAssetMarket = from_python(value)?;
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }
}

#[pyclass(module = "openferric", get_all, from_py_object)]
#[derive(Clone, Copy)]
pub struct ExtendedGreeks {
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    rho: f64,
    vanna: f64,
    volga: f64,
}

#[pyclass(module = "openferric", get_all, from_py_object)]
#[derive(Clone, Copy)]
pub struct CrossGreeks {
    cross_gamma: f64,
    corr_sens: f64,
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DslMonteCarloEngine {
    inner: core_dsl::DslMonteCarloEngine,
}

#[pymethods]
impl DslMonteCarloEngine {
    #[getter]
    fn num_paths(&self) -> usize {
        self.inner.num_paths
    }

    #[getter]
    fn num_steps(&self) -> usize {
        self.inner.num_steps
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn rng_kind(&self) -> crate::math_bindings::FastRngKind {
        crate::math_bindings::FastRngKind {
            inner: self.inner.rng_kind,
        }
    }

    #[new]
    #[pyo3(signature = (num_paths=10_000, num_steps=100, seed=42, rng_kind="xoshiro"))]
    fn new(num_paths: usize, num_steps: usize, seed: u64, rng_kind: &str) -> PyResult<Self> {
        if num_paths == 0 || num_steps == 0 {
            return Err(error("num_paths and num_steps must be positive"));
        }
        use openferric_core::math::FastRngKind;
        let rng_kind = match rng_kind.to_ascii_lowercase().as_str() {
            "xoshiro" | "xoshiro256plusplus" => FastRngKind::Xoshiro256PlusPlus,
            "pcg" | "pcg64" => FastRngKind::Pcg64,
            "std" | "stdrng" => FastRngKind::StdRng,
            "thread" | "threadrng" => FastRngKind::ThreadRng,
            _ => return Err(error("unknown RNG kind")),
        };
        let mut inner = core_dsl::DslMonteCarloEngine::new(num_paths, num_steps, seed);
        inner.rng_kind = rng_kind;
        Ok(Self { inner })
    }

    fn resolve_execution_backend(&self, policy: ExecutionPolicy) -> PyResult<ExecutionBackend> {
        self.inner
            .resolve_execution_backend(policy.to_core())
            .map(Into::into)
            .map_err(error)
    }

    fn price(
        &self,
        py: Python<'_>,
        product: &DslProduct,
        market: &Market,
    ) -> PyResult<PricingResult> {
        let market = market.to_core()?;
        py.detach(|| self.inner.price(&product.inner, &market))
            .map(Into::into)
            .map_err(error)
    }

    #[pyo3(signature = (product, market, policy=ExecutionPolicy::Auto))]
    fn price_multi_asset(
        &self,
        py: Python<'_>,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        policy: ExecutionPolicy,
    ) -> PyResult<PricingResult> {
        py.detach(|| {
            self.inner.price_multi_asset_with_policy(
                &product.inner,
                &market.inner,
                policy.to_core(),
            )
        })
        .map(Into::into)
        .map_err(error)
    }

    fn greeks_multi_asset(
        &self,
        py: Python<'_>,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_index: usize,
    ) -> PyResult<Greeks> {
        py.detach(|| {
            self.inner
                .greeks_multi_asset(&product.inner, &market.inner, asset_index)
        })
        .map(Greeks::from_core)
        .map_err(error)
    }

    fn extended_greeks_multi_asset(
        &self,
        py: Python<'_>,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_index: usize,
        base_price: f64,
    ) -> PyResult<ExtendedGreeks> {
        let result = py
            .detach(|| {
                self.inner.extended_greeks_multi_asset(
                    &product.inner,
                    &market.inner,
                    asset_index,
                    base_price,
                )
            })
            .map_err(error)?;
        Ok(ExtendedGreeks {
            delta: result.delta,
            gamma: result.gamma,
            vega: result.vega,
            theta: result.theta,
            rho: result.rho,
            vanna: result.vanna,
            volga: result.volga,
        })
    }

    fn cross_greeks_multi_asset(
        &self,
        py: Python<'_>,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_i: usize,
        asset_j: usize,
        base_price: f64,
    ) -> PyResult<CrossGreeks> {
        let result = py
            .detach(|| {
                self.inner.cross_greeks_multi_asset(
                    &product.inner,
                    &market.inner,
                    asset_i,
                    asset_j,
                    base_price,
                )
            })
            .map_err(error)?;
        Ok(CrossGreeks {
            cross_gamma: result.cross_gamma,
            corr_sens: result.corr_sens,
        })
    }
}

#[pyclass(module = "openferric")]
pub struct ProductEvaluator {
    inner: eval::ProductEvaluator,
}

#[pymethods]
impl ProductEvaluator {
    #[new]
    fn new(product: &CompiledProduct, num_steps: usize, rate: f64) -> PyResult<Self> {
        Ok(Self {
            inner: eval::ProductEvaluator::new(&product.inner, num_steps, rate).map_err(error)?,
        })
    }

    fn evaluate(
        &mut self,
        py: Python<'_>,
        path_spots: Vec<Vec<f64>>,
        initial_spots: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate(&path_spots, &initial_spots))
            .map_err(error)
    }
}

#[cfg(feature = "jit")]
#[pyclass(module = "openferric")]
pub struct JitProductEvaluator {
    inner: core_dsl::JitProductEvaluator,
}

#[cfg(feature = "jit")]
#[pymethods]
impl JitProductEvaluator {
    #[new]
    fn new(
        py: Python<'_>,
        product: &CompiledProduct,
        num_steps: usize,
        rate: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: py
                .detach(|| core_dsl::JitProductEvaluator::compile(&product.inner, num_steps, rate))
                .map_err(error)?,
        })
    }

    fn evaluate_path(
        &self,
        py: Python<'_>,
        path_spots: Vec<Vec<f64>>,
        initial_spots: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate_path(&path_spots, &initial_spots))
            .map_err(error)
    }

    fn new_scratch(&self) -> JitEvaluationScratch {
        JitEvaluationScratch {
            inner: self.inner.new_scratch(),
        }
    }

    fn evaluate_path_with_scratch(
        &self,
        py: Python<'_>,
        path_spots: Vec<Vec<f64>>,
        initial_spots: Vec<f64>,
        scratch: &mut JitEvaluationScratch,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner
                .evaluate_path_with_scratch(&path_spots, &initial_spots, &mut scratch.inner)
        })
        .map_err(error)
    }
}

#[cfg(feature = "jit")]
#[pyclass(module = "openferric")]
pub struct JitEvaluationScratch {
    inner: core_dsl::JitEvaluationScratch,
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ProductDef {
    pub(crate) inner: ast::ProductDef,
}

#[pymethods]
impl ProductDef {
    #[new]
    fn new(source: &str) -> PyResult<Self> {
        Ok(Self {
            inner: core_dsl::parser::parse(lexer::tokenize(source).map_err(error)?)
                .map_err(error)?,
        })
    }

    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: from_python(data)?,
        })
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }
    #[getter]
    fn body(&self) -> Vec<crate::dsl_data::ProductItem> {
        self.inner
            .body
            .iter()
            .cloned()
            .map(|inner| crate::dsl_data::ProductItem { inner })
            .collect()
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn span(&self) -> (usize, usize) {
        (self.inner.span.start, self.inner.span.end)
    }

    fn compile(&self) -> PyResult<CompiledProduct> {
        Ok(CompiledProduct {
            inner: core_dsl::compile(&self.inner).map_err(error)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(module = "openferric")]
pub struct SymbolTable {
    pub(crate) inner: analysis::SymbolTable,
}

#[pymethods]
impl SymbolTable {
    #[getter]
    fn declarations(&self) -> Vec<crate::dsl_data::SymbolInfo> {
        self.inner
            .declarations
            .iter()
            .cloned()
            .map(|inner| crate::dsl_data::SymbolInfo { inner })
            .collect()
    }

    #[getter]
    fn references(&self) -> Vec<crate::dsl_data::SymbolRef> {
        self.inner
            .references
            .iter()
            .cloned()
            .map(|inner| crate::dsl_data::SymbolRef { inner })
            .collect()
    }

    #[new]
    fn new(product: &ProductDef, source: &str) -> Self {
        Self {
            inner: analysis::build_symbol_table(&product.inner, source),
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn declaration_at(&self, py: Python<'_>, offset: usize) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner.declaration_at(offset))
    }

    fn reference_at(&self, py: Python<'_>, offset: usize) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner.reference_at(offset))
    }
}

#[pyfunction]
fn parse_and_compile(source: &str) -> PyResult<CompiledProduct> {
    Ok(CompiledProduct {
        inner: core_dsl::parse_and_compile(source).map_err(error)?,
    })
}

#[pyfunction]
fn compile(product: &ProductDef) -> PyResult<CompiledProduct> {
    product.compile()
}

#[pyfunction]
fn parse_and_diagnose(
    py: Python<'_>,
    source: &str,
) -> PyResult<(Option<ProductDef>, Option<CompiledProduct>, Py<PyAny>)> {
    let (ast, compiled, diagnostics) = analysis::parse_and_diagnose(source);
    Ok((
        ast.map(|inner| ProductDef { inner }),
        compiled.map(|inner| CompiledProduct { inner }),
        to_python(py, &diagnostics)?,
    ))
}

#[pyfunction]
fn completions(
    py: Python<'_>,
    source: &str,
    symbols: &SymbolTable,
    offset: usize,
) -> PyResult<Py<PyAny>> {
    checked_offset(source, offset)?;
    to_python(py, &analysis::completions(source, &symbols.inner, offset))
}

#[pyfunction]
fn hover_info(
    py: Python<'_>,
    source: &str,
    symbols: &SymbolTable,
    offset: usize,
) -> PyResult<Py<PyAny>> {
    checked_offset(source, offset)?;
    to_python(py, &analysis::hover_info(source, &symbols.inner, offset))
}

#[pyfunction]
fn goto_definition(source: &str, symbols: &SymbolTable, offset: usize) -> Option<(usize, usize)> {
    analysis::goto_definition(source, &symbols.inner, offset).map(|span| (span.start, span.end))
}

#[pyfunction]
fn semantic_token_data(py: Python<'_>, source: &str, symbols: &SymbolTable) -> PyResult<Py<PyAny>> {
    to_python(py, &analysis::semantic_token_data(source, &symbols.inner))
}

#[pyfunction]
fn offset_to_line_col(source: &str, offset: usize) -> PyResult<(u32, u32)> {
    Ok(analysis::offset_to_line_col(
        source,
        checked_offset(source, offset)?,
    ))
}

#[pyfunction]
fn line_col_to_offset(source: &str, line: u32, column: u32) -> usize {
    analysis::line_col_to_offset(source, line, column)
}

#[pyfunction]
fn annotate_source(source: &str, start: usize, end: usize) -> String {
    core_dsl::error::annotate_source(source, core_dsl::error::Span::new(start, end))
}

#[pyfunction]
fn evaluate_product(
    py: Python<'_>,
    product: &CompiledProduct,
    path_spots: Vec<Vec<f64>>,
    initial_spots: Vec<f64>,
    num_steps: usize,
    rate: f64,
) -> PyResult<f64> {
    py.detach(|| {
        eval::evaluate_product(&product.inner, &path_spots, &initial_spots, num_steps, rate)
    })
    .map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ExecutionPolicy>()?;
    module.add_class::<ExecutionBackend>()?;
    module.add_class::<CompiledProduct>()?;
    module.add_class::<DslProduct>()?;
    module.add_class::<AssetMarketData>()?;
    module.add_class::<MultiAssetMarket>()?;
    module.add_class::<ExtendedGreeks>()?;
    module.add_class::<CrossGreeks>()?;
    module.add_class::<DslMonteCarloEngine>()?;
    module.add_class::<ProductEvaluator>()?;
    module.add_class::<ProductDef>()?;
    module.add_class::<SymbolTable>()?;
    #[cfg(feature = "jit")]
    {
        module.add_class::<JitProductEvaluator>()?;
        module.add_class::<JitEvaluationScratch>()?;
    }
    module.add_function(wrap_pyfunction!(parse_and_compile, module)?)?;
    module.add_function(wrap_pyfunction!(compile, module)?)?;
    module.add_function(wrap_pyfunction!(parse_and_diagnose, module)?)?;
    module.add_function(wrap_pyfunction!(completions, module)?)?;
    module.add_function(wrap_pyfunction!(hover_info, module)?)?;
    module.add_function(wrap_pyfunction!(goto_definition, module)?)?;
    module.add_function(wrap_pyfunction!(semantic_token_data, module)?)?;
    module.add_function(wrap_pyfunction!(offset_to_line_col, module)?)?;
    module.add_function(wrap_pyfunction!(line_col_to_offset, module)?)?;
    module.add_function(wrap_pyfunction!(annotate_source, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_product, module)?)?;
    Ok(())
}
