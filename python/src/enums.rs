use pyo3::prelude::*;
use pyo3::types::PyDict;

fn string_enum(
    module: &Bound<'_, PyModule>,
    name: &str,
    variants: &[(&str, &str)],
) -> PyResult<()> {
    let members = PyDict::new(module.py());
    for (name, value) in variants {
        members.set_item(name, value)?;
    }
    let keywords = PyDict::new(module.py());
    keywords.set_item("type", module.py().get_type::<pyo3::types::PyString>())?;
    keywords.set_item("module", "openferric")?;
    let enumeration = module
        .py()
        .import("enum")?
        .getattr("Enum")?
        .call((name, members), Some(&keywords))?;
    module.add(name, enumeration)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    string_enum(
        module,
        "ForwardInterpolation",
        &[
            ("PiecewiseFlat", "piecewise_flat"),
            ("Linear", "linear"),
            ("CubicSpline", "cubic_spline"),
        ],
    )?;
    string_enum(
        module,
        "CurveStructure",
        &[
            ("Contango", "contango"),
            ("Backwardation", "backwardation"),
            ("Flat", "flat"),
            ("Mixed", "mixed"),
        ],
    )?;
    string_enum(
        module,
        "SeasonalityMode",
        &[
            ("Additive", "additive"),
            ("Multiplicative", "multiplicative"),
        ],
    )?;
    string_enum(
        module,
        "FbmScheme",
        &[("Cholesky", "cholesky"), ("Hybrid", "hybrid")],
    )?;
    string_enum(
        module,
        "BinaryBarrierType",
        &[
            ("DownIn", "down_in"),
            ("UpIn", "up_in"),
            ("DownOut", "down_out"),
            ("UpOut", "up_out"),
        ],
    )?;
    string_enum(
        module,
        "TerminationReason",
        &[
            ("GradientTolerance", "gradient_tolerance"),
            ("StepTolerance", "step_tolerance"),
            ("ObjectiveTolerance", "objective_tolerance"),
            ("Stagnation", "stagnation"),
            ("MaxIterations", "max_iterations"),
            ("NumericalFailure", "numerical_failure"),
        ],
    )?;
    string_enum(
        module,
        "CalibrationWarningFlag",
        &[
            ("IllConditioned", "ill_conditioned"),
            ("HitBoundary", "hit_boundary"),
            ("PoorFit", "poor_fit"),
            ("NonConvergent", "non_convergent"),
            ("UnstableParameters", "unstable_parameters"),
        ],
    )?;
    for name in ["MathError", "InterpolationError", "DslError"] {
        module.add(
            name,
            module.py().get_type::<pyo3::exceptions::PyValueError>(),
        )?;
    }
    module.add(
        "MAX_QUOTED_NORMAL_VOL",
        openferric_core::models::hw_calibration::MAX_QUOTED_NORMAL_VOL,
    )?;
    module.add(
        "FUNDING_RATE_BUMP_BP",
        openferric_core::pricing::funding_rate_swap::FUNDING_RATE_BUMP_BP,
    )?;
    module.add(
        "FUNDING_RATE_VOL_BUMP",
        openferric_core::pricing::funding_rate_swap::FUNDING_RATE_VOL_BUMP,
    )?;
    Ok(())
}
