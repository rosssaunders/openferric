use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

#[derive(Clone)]
struct ComplexFftPlan {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

static COMPLEX_FFT_CACHE: OnceLock<Mutex<HashMap<usize, ComplexFftPlan>>> = OnceLock::new();
static REAL_FFT_CACHE: OnceLock<Mutex<HashMap<usize, Arc<dyn RealToComplex<f64>>>>> =
    OnceLock::new();

thread_local! {
    static COMPLEX_FFT_SCRATCH: RefCell<HashMap<(usize, bool), Vec<Complex<f64>>>> =
        RefCell::new(HashMap::new());
    static REAL_FFT_SCRATCH: RefCell<HashMap<usize, Vec<Complex<f64>>>> =
        RefCell::new(HashMap::new());
}

fn complex_plan(n: usize) -> ComplexFftPlan {
    let cache = COMPLEX_FFT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("complex FFT cache lock poisoned");
    if let Some(plan) = guard.get(&n) {
        return plan.clone();
    }

    let mut planner = FftPlanner::<f64>::new();
    let plan = ComplexFftPlan {
        forward: planner.plan_fft_forward(n),
        inverse: planner.plan_fft_inverse(n),
    };
    guard.insert(n, plan.clone());
    plan
}

fn real_plan(n: usize) -> Arc<dyn RealToComplex<f64>> {
    let cache = REAL_FFT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("real FFT cache lock poisoned");
    if let Some(plan) = guard.get(&n) {
        return Arc::clone(plan);
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(n);
    guard.insert(n, Arc::clone(&plan));
    plan
}

fn fft_inplace(values: &mut [Complex<f64>], inverse: bool) {
    let n = values.len();
    assert!(n.is_power_of_two(), "fft length must be power-of-two");
    if n == 0 {
        return;
    }

    let plan = complex_plan(n);
    let fft = if inverse {
        &plan.inverse
    } else {
        &plan.forward
    };
    let scratch_len = fft.get_inplace_scratch_len();

    COMPLEX_FFT_SCRATCH.with(|cache| {
        let mut cache = cache.borrow_mut();
        let scratch = cache.entry((n, inverse)).or_default();
        if scratch.len() < scratch_len {
            scratch.resize(scratch_len, Complex::new(0.0, 0.0));
        }
        fft.process_with_scratch(values, &mut scratch[..scratch_len]);
    });

    if inverse {
        let inv_n = 1.0 / n as f64;
        for x in values {
            *x *= inv_n;
        }
    }
}

pub fn fft_forward(values: &mut [Complex<f64>]) {
    fft_inplace(values, false);
}

pub fn fft_inverse(values: &mut [Complex<f64>]) {
    fft_inplace(values, true);
}

pub fn fft_forward_real(values: &[f64]) -> Result<Vec<Complex<f64>>, String> {
    let n = values.len();
    if n < 2 || !n.is_power_of_two() {
        return Err("real FFT length must be a power-of-two >= 2".to_string());
    }

    let plan = real_plan(n);
    let mut input = values.to_vec();
    let mut half_spectrum = plan.make_output_vec();
    let scratch_len = plan.get_scratch_len();

    REAL_FFT_SCRATCH.with(|cache| {
        let mut cache = cache.borrow_mut();
        let scratch = cache.entry(n).or_default();
        if scratch.len() < scratch_len {
            scratch.resize(scratch_len, Complex::new(0.0, 0.0));
        }
        plan.process_with_scratch(&mut input, &mut half_spectrum, &mut scratch[..scratch_len])
            .map_err(|err| err.to_string())
    })?;

    let mut out = vec![Complex::new(0.0, 0.0); n];
    for (k, value) in half_spectrum.iter().copied().enumerate() {
        out[k] = value;
    }
    for k in 1..(n / 2) {
        out[n - k] = half_spectrum[k].conj();
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_inverse_roundtrip() {
        let original = vec![
            Complex::new(1.0, 0.0),
            Complex::new(-2.0, 0.5),
            Complex::new(3.0, -1.5),
            Complex::new(0.2, 0.1),
            Complex::new(-0.7, 0.0),
            Complex::new(0.3, -0.9),
            Complex::new(0.0, 0.0),
            Complex::new(2.5, 1.1),
        ];

        let mut transformed = original.clone();
        fft_forward(&mut transformed);
        fft_inverse(&mut transformed);

        for (lhs, rhs) in transformed.iter().zip(original.iter()) {
            assert!((*lhs - *rhs).norm() < 1e-10);
        }
    }

    #[test]
    fn real_fft_matches_complex_fft_for_real_input() {
        let real = vec![0.5, -1.0, 3.0, 2.0, -0.25, 1.5, 0.0, 4.0];
        let mut complex: Vec<Complex<f64>> =
            real.iter().copied().map(|x| Complex::new(x, 0.0)).collect();

        fft_forward(&mut complex);
        let real_fft = fft_forward_real(&real).expect("real FFT should succeed");

        for (lhs, rhs) in real_fft.iter().zip(complex.iter()) {
            assert!((*lhs - *rhs).norm() < 1e-12);
        }
    }
}
