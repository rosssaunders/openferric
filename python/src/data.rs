macro_rules! data_type {
    ($name:ident, $core:ty, {$($methods:tt)*}) => {
        #[pyo3::pyclass(module = "openferric", from_py_object)]
        #[derive(Clone)]
        pub struct $name { pub(crate) inner: $core }
        #[pyo3::pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (data=None, **fields))]
            fn new(data: Option<&Bound<'_, PyAny>>, fields: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
                let value = match (data, fields) {
                    (Some(_), Some(fields)) if !fields.is_empty() => return Err(pyo3::exceptions::PyTypeError::new_err("supply data or keyword fields, not both")),
                    (Some(data), _) => data,
                    (None, Some(fields)) => fields.as_any(),
                    (None, None) => return Err(pyo3::exceptions::PyTypeError::new_err("data is required")),
                };
                Ok(Self { inner: crate::helpers::from_python(value)? })
            }
            #[staticmethod]
            fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<Self> { Ok(Self { inner: crate::helpers::from_python(data)? }) }
            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { crate::helpers::to_python(py, &self.inner) }
            fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
                self.to_dict(py)?.bind(py).get_item(name).map(Bound::unbind).map_err(|_| pyo3::exceptions::PyAttributeError::new_err(name.to_owned()))
            }
            fn __repr__(&self) -> String { format!("{}({:?})", stringify!($name), self.inner) }
            $($methods)*
        }
    }
}
pub(crate) use data_type;
