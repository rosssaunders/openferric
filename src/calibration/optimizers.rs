//! Constrained optimizers for calibration workloads.
//!
//! References:
//! - Levenberg (1944), Marquardt (1963).
//! - Storn and Price (1997), Differential Evolution.
//! - Nelder and Mead (1965), simplex direct search.

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;

use crate::calibration::core::{BoxConstraints, ConvergenceInfo, TerminationReason};

/// Shared optimization payload used by all solvers.
#[derive(Debug, Clone)]
pub struct OptimisationResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub residuals: Vec<f64>,
    pub jacobian: DMatrix<f64>,
    pub convergence: ConvergenceInfo,
}

#[derive(Debug, Clone, Copy)]
pub struct LmOptions {
    pub max_iterations: usize,
    pub initial_lambda: f64,
    pub lambda_up: f64,
    pub lambda_down: f64,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub objective_tolerance: f64,
    pub finite_diff_epsilon: f64,
    pub max_stagnation: usize,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            max_iterations: 80,
            initial_lambda: 1e-2,
            lambda_up: 3.0,
            lambda_down: 0.35,
            gradient_tolerance: 1e-6,
            step_tolerance: 1e-7,
            objective_tolerance: 1e-10,
            finite_diff_epsilon: 1e-4,
            max_stagnation: 20,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DifferentialEvolutionOptions {
    pub max_generations: usize,
    pub population_size: usize,
    pub mutation_factor: f64,
    pub crossover_probability: f64,
    pub seed: u64,
    pub max_stagnation: usize,
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            max_generations: 120,
            population_size: 32,
            mutation_factor: 0.8,
            crossover_probability: 0.9,
            seed: 7,
            max_stagnation: 24,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NelderMeadOptions {
    pub max_iterations: usize,
    pub initial_step: f64,
    pub reflection: f64,
    pub expansion: f64,
    pub contraction: f64,
    pub shrink: f64,
    pub tolerance: f64,
}

impl Default for NelderMeadOptions {
    fn default() -> Self {
        Self {
            max_iterations: 240,
            initial_step: 0.08,
            reflection: 1.0,
            expansion: 2.0,
            contraction: 0.5,
            shrink: 0.5,
            tolerance: 1e-7,
        }
    }
}

#[inline]
fn least_squares_objective(residuals: &[f64]) -> f64 {
    0.5 * residuals.iter().map(|r| r * r).sum::<f64>()
}

fn finite_difference_jacobian<F>(
    x: &[f64],
    base_residuals: &[f64],
    bounds: &BoxConstraints,
    eps_scale: f64,
    residual_fn: &mut F,
    objective_evaluations: &mut usize,
) -> DMatrix<f64>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let m = base_residuals.len();
    let n = x.len();
    let mut j = DMatrix::zeros(m, n);

    for c in 0..n {
        let mut xp = x.to_vec();
        let h = (x[c].abs() * eps_scale).max(1e-7);

        xp[c] = (x[c] + h).min(bounds.upper[c]);
        if (xp[c] - x[c]).abs() < 1e-14 {
            xp[c] = (x[c] - h).max(bounds.lower[c]);
        }

        let denom = xp[c] - x[c];
        if denom.abs() < 1e-14 {
            continue;
        }

        let rp = residual_fn(&xp);
        *objective_evaluations += 1;
        for r in 0..m {
            j[(r, c)] = (rp[r] - base_residuals[r]) / denom;
        }
    }

    j
}

pub fn levenberg_marquardt<F>(
    initial: &[f64],
    bounds: &BoxConstraints,
    options: LmOptions,
    mut residual_fn: F,
) -> Result<OptimisationResult, String>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    if initial.len() != bounds.dimension() {
        return Err("LM initial vector dimension does not match bounds".to_string());
    }

    let mut x = bounds.clamp(initial);
    let mut evals = 0usize;
    let mut residuals = residual_fn(&x);
    evals += 1;
    if residuals.is_empty() {
        return Err("LM residual function returned empty residual vector".to_string());
    }

    let mut objective = least_squares_objective(&residuals);
    if !objective.is_finite() {
        return Err("LM objective is not finite at initial point".to_string());
    }

    let mut lambda = options.initial_lambda.max(1e-12);
    let mut iterations = 0usize;
    let mut last_gradient_norm = f64::INFINITY;
    let mut last_step_norm = f64::INFINITY;
    let mut reason = TerminationReason::MaxIterations;
    let mut converged = false;
    let mut stagnation = 0usize;

    for iter in 0..options.max_iterations {
        iterations = iter + 1;

        let jacobian = finite_difference_jacobian(
            &x,
            &residuals,
            bounds,
            options.finite_diff_epsilon.max(1e-8),
            &mut residual_fn,
            &mut evals,
        );

        let r_vec = DVector::from_column_slice(&residuals);
        let jt = jacobian.transpose();
        let mut a = &jt * &jacobian;
        let g = &jt * r_vec;

        last_gradient_norm = g.norm();
        if !last_gradient_norm.is_finite() {
            reason = TerminationReason::NumericalFailure;
            break;
        }
        if last_gradient_norm <= options.gradient_tolerance {
            converged = true;
            reason = TerminationReason::GradientTolerance;
            break;
        }

        for i in 0..a.nrows() {
            a[(i, i)] += lambda * (a[(i, i)].abs() + 1.0);
        }

        let Some(delta) = a.lu().solve(&(-g)) else {
            lambda = (lambda * options.lambda_up).min(1e12);
            stagnation += 1;
            if stagnation >= options.max_stagnation {
                reason = TerminationReason::Stagnation;
                break;
            }
            continue;
        };

        last_step_norm = delta.norm();
        if last_step_norm <= options.step_tolerance {
            converged = true;
            reason = TerminationReason::StepTolerance;
            break;
        }

        let mut candidate = x.clone();
        for i in 0..candidate.len() {
            candidate[i] += delta[i];
        }
        candidate = bounds.clamp(&candidate);

        let candidate_residuals = residual_fn(&candidate);
        evals += 1;
        let candidate_obj = least_squares_objective(&candidate_residuals);

        if candidate_obj.is_finite() && candidate_obj + 1e-16 < objective {
            let improvement = (objective - candidate_obj).abs();
            x = candidate;
            residuals = candidate_residuals;
            objective = candidate_obj;
            lambda = (lambda * options.lambda_down).max(1e-12);
            stagnation = 0;

            if improvement <= options.objective_tolerance {
                converged = true;
                reason = TerminationReason::ObjectiveTolerance;
                break;
            }
        } else {
            lambda = (lambda * options.lambda_up).min(1e12);
            stagnation += 1;
            if stagnation >= options.max_stagnation {
                reason = TerminationReason::Stagnation;
                break;
            }
        }
    }

    let jacobian = finite_difference_jacobian(
        &x,
        &residuals,
        bounds,
        options.finite_diff_epsilon.max(1e-8),
        &mut residual_fn,
        &mut evals,
    );

    Ok(OptimisationResult {
        x,
        objective,
        residuals,
        jacobian,
        convergence: ConvergenceInfo {
            iterations,
            objective_evaluations: evals,
            gradient_norm: last_gradient_norm,
            step_norm: last_step_norm,
            converged,
            reason,
        },
    })
}

pub fn differential_evolution<F>(
    bounds: &BoxConstraints,
    options: DifferentialEvolutionOptions,
    mut objective_fn: F,
) -> Result<OptimisationResult, String>
where
    F: FnMut(&[f64]) -> f64,
{
    let dim = bounds.dimension();
    if dim == 0 {
        return Err("DE requires non-empty parameter bounds".to_string());
    }

    let pop_size = options.population_size.max(dim + 2);
    if pop_size < 4 {
        return Err("DE requires population_size >= 4".to_string());
    }

    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut population = Vec::with_capacity(pop_size);
    let mut values = Vec::with_capacity(pop_size);
    let mut evals = 0usize;

    for _ in 0..pop_size {
        let mut x = vec![0.0; dim];
        for (d, xd) in x.iter_mut().enumerate() {
            let u: f64 = rng.random();
            *xd = bounds.lower[d] + u * (bounds.upper[d] - bounds.lower[d]);
        }
        let v = objective_fn(&x);
        evals += 1;
        population.push(x);
        values.push(v);
    }

    let mut best_idx = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
        .ok_or_else(|| "DE failed to initialize population".to_string())?;

    let mut stagnation = 0usize;
    let mut iterations = 0usize;
    let mut reason = TerminationReason::MaxIterations;
    let mut converged = false;

    for g in 0..options.max_generations {
        iterations = g + 1;
        let prev_best = values[best_idx];

        for i in 0..pop_size {
            let mut idxs: Vec<usize> = (0..pop_size).filter(|&k| k != i).collect();
            idxs.shuffle(&mut rng);
            let (a, b, c) = (idxs[0], idxs[1], idxs[2]);

            let mut mutant = vec![0.0; dim];
            for (d, md) in mutant.iter_mut().enumerate() {
                *md = population[a][d]
                    + options.mutation_factor * (population[b][d] - population[c][d]);
            }
            mutant = bounds.clamp(&mutant);

            let j_rand = rng.random_range(0..dim);
            let mut trial = population[i].clone();
            for d in 0..dim {
                let p: f64 = rng.random();
                if p <= options.crossover_probability || d == j_rand {
                    trial[d] = mutant[d];
                }
            }
            trial = bounds.clamp(&trial);

            let trial_value = objective_fn(&trial);
            evals += 1;
            if trial_value.is_finite() && trial_value < values[i] {
                population[i] = trial;
                values[i] = trial_value;
                if trial_value < values[best_idx] {
                    best_idx = i;
                }
            }
        }

        if (prev_best - values[best_idx]).abs() <= 1e-12 {
            stagnation += 1;
            if stagnation >= options.max_stagnation {
                converged = true;
                reason = TerminationReason::Stagnation;
                break;
            }
        } else {
            stagnation = 0;
        }
    }

    let best = population[best_idx].clone();
    let best_val = values[best_idx];

    Ok(OptimisationResult {
        x: best,
        objective: best_val,
        residuals: vec![best_val.sqrt()],
        jacobian: DMatrix::zeros(1, dim),
        convergence: ConvergenceInfo {
            iterations,
            objective_evaluations: evals,
            gradient_norm: f64::NAN,
            step_norm: f64::NAN,
            converged,
            reason,
        },
    })
}

pub fn nelder_mead<F>(
    initial: &[f64],
    bounds: &BoxConstraints,
    options: NelderMeadOptions,
    mut objective_fn: F,
) -> Result<OptimisationResult, String>
where
    F: FnMut(&[f64]) -> f64,
{
    let dim = bounds.dimension();
    if initial.len() != dim {
        return Err("Nelder-Mead initial vector dimension does not match bounds".to_string());
    }

    let mut simplex = Vec::with_capacity(dim + 1);
    let mut values = Vec::with_capacity(dim + 1);
    let mut evals = 0usize;

    let x0 = bounds.clamp(initial);
    simplex.push(x0.clone());
    values.push(objective_fn(&x0));
    evals += 1;

    for d in 0..dim {
        let mut x = x0.clone();
        let step = (bounds.upper[d] - bounds.lower[d]).abs() * options.initial_step.max(1e-4);
        x[d] = (x[d] + step).min(bounds.upper[d]);
        if (x[d] - x0[d]).abs() < 1e-14 {
            x[d] = (x[d] - step).max(bounds.lower[d]);
        }
        x = bounds.clamp(&x);
        simplex.push(x.clone());
        values.push(objective_fn(&x));
        evals += 1;
    }

    let mut iterations = 0usize;
    let mut reason = TerminationReason::MaxIterations;
    let mut converged = false;

    for iter in 0..options.max_iterations {
        iterations = iter + 1;

        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));

        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        values = order.iter().map(|&i| values[i]).collect();

        let best = values[0];
        let worst = values[dim];
        let spread = (worst - best).abs();

        let centroid: Vec<f64> = (0..dim)
            .map(|d| simplex.iter().take(dim).map(|x| x[d]).sum::<f64>() / dim as f64)
            .collect();

        let max_vertex_dist = simplex
            .iter()
            .map(|x| {
                x.iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);

        if spread <= options.tolerance && max_vertex_dist <= options.tolerance {
            converged = true;
            reason = TerminationReason::ObjectiveTolerance;
            break;
        }

        let xr: Vec<f64> = (0..dim)
            .map(|d| centroid[d] + options.reflection * (centroid[d] - simplex[dim][d]))
            .collect();
        let xr = bounds.clamp(&xr);
        let fr = objective_fn(&xr);
        evals += 1;

        if fr < values[0] {
            let xe: Vec<f64> = (0..dim)
                .map(|d| centroid[d] + options.expansion * (xr[d] - centroid[d]))
                .collect();
            let xe = bounds.clamp(&xe);
            let fe = objective_fn(&xe);
            evals += 1;

            if fe < fr {
                simplex[dim] = xe;
                values[dim] = fe;
            } else {
                simplex[dim] = xr;
                values[dim] = fr;
            }
            continue;
        }

        if fr < values[dim - 1] {
            simplex[dim] = xr;
            values[dim] = fr;
            continue;
        }

        let xc: Vec<f64> = (0..dim)
            .map(|d| centroid[d] + options.contraction * (simplex[dim][d] - centroid[d]))
            .collect();
        let xc = bounds.clamp(&xc);
        let fc = objective_fn(&xc);
        evals += 1;

        if fc < values[dim] {
            simplex[dim] = xc;
            values[dim] = fc;
            continue;
        }

        for i in 1..=dim {
            for d in 0..dim {
                simplex[i][d] = simplex[0][d] + options.shrink * (simplex[i][d] - simplex[0][d]);
            }
            simplex[i] = bounds.clamp(&simplex[i]);
            values[i] = objective_fn(&simplex[i]);
            evals += 1;
        }
    }

    let mut order: Vec<usize> = (0..simplex.len()).collect();
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));

    let best = simplex[order[0]].clone();
    let best_val = values[order[0]];

    Ok(OptimisationResult {
        x: best,
        objective: best_val,
        residuals: vec![best_val.sqrt()],
        jacobian: DMatrix::zeros(1, dim),
        convergence: ConvergenceInfo {
            iterations,
            objective_evaluations: evals,
            gradient_norm: f64::NAN,
            step_norm: f64::NAN,
            converged,
            reason,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_respects_bounds_and_fits_quadratic_residual() {
        let bounds = BoxConstraints::new(vec![-5.0, -5.0], vec![5.0, 5.0]).unwrap();
        let initial = [4.0, -4.0];
        let out = levenberg_marquardt(&initial, &bounds, LmOptions::default(), |x| {
            vec![x[0] - 1.5, x[1] + 2.0]
        })
        .unwrap();

        assert!((out.x[0] - 1.5).abs() < 1e-6);
        assert!((out.x[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn de_handles_box_constraints() {
        let bounds = BoxConstraints::new(vec![-1.0, -1.0], vec![1.0, 1.0]).unwrap();
        let out = differential_evolution(
            &bounds,
            DifferentialEvolutionOptions {
                max_generations: 80,
                population_size: 28,
                ..DifferentialEvolutionOptions::default()
            },
            |x| (x[0] - 0.2).powi(2) + (x[1] + 0.3).powi(2),
        )
        .unwrap();

        assert!((out.x[0] - 0.2).abs() < 1e-2);
        assert!((out.x[1] + 0.3).abs() < 1e-2);
    }

    #[test]
    fn nelder_mead_handles_box_constraints() {
        let bounds = BoxConstraints::new(vec![-1.0, -1.0], vec![1.0, 1.0]).unwrap();
        let out = nelder_mead(&[0.9, 0.9], &bounds, NelderMeadOptions::default(), |x| {
            (x[0] - 0.25).powi(2) + (x[1] + 0.4).powi(2)
        })
        .unwrap();

        assert!((out.x[0] - 0.25).abs() < 1e-4);
        assert!((out.x[1] + 0.4).abs() < 1e-4);
    }
}
