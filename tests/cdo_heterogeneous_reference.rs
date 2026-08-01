//! Independent finite-pool Gaussian-copula and base-correlation references.
//!
//! The primary targets were generated with SciPy 1.17.1 using 128-node
//! `scipy.special.roots_hermitenorm`. At every factor node an independent
//! Python program constructs the conditional Bernoulli loss distribution by
//! direct convolution; the four-name case was also checked by enumerating all
//! 16 default states. The 64/128/256-node SciPy values agree at binary64
//! precision for the targets below.

use approx::assert_relative_eq;
use openferric::credit::{
    BaseCorrelationPair, CdoReferenceName, CdoTranche, HeterogeneousSyntheticCdo, SurvivalCurve,
};

fn common_recovery_model() -> HeterogeneousSyntheticCdo {
    let survival_probabilities = [0.99, 0.97, 0.94, 0.89, 0.82];
    let factor_loadings = [0.10, 0.25, 0.40, 0.55, 0.70];
    HeterogeneousSyntheticCdo {
        names: survival_probabilities
            .into_iter()
            .zip(factor_loadings)
            .map(|(survival, factor_loading)| CdoReferenceName {
                notional: 1.0,
                recovery_rate: 0.40,
                factor_loading,
                survival_curve: SurvivalCurve::new(vec![(5.0, survival)]),
            })
            .collect(),
        risk_free_rate: 0.03,
        maturity: 5.0,
        payment_freq: 4,
        // Each equal-weight name loses 0.6 / 5 = 0.12.
        loss_unit: 0.12,
        factor_integration_points: 64,
    }
}

fn fully_heterogeneous_model() -> HeterogeneousSyntheticCdo {
    let notionals = [1.0, 2.0, 3.0, 4.0];
    let recoveries = [0.40, 0.50, 0.60, 0.75];
    let hazards = [0.008, 0.015, 0.030, 0.060];
    let factor_loadings = [-0.15, 0.20, 0.45, 0.65];
    HeterogeneousSyntheticCdo {
        names: (0..4)
            .map(|index| CdoReferenceName {
                notional: notionals[index],
                recovery_rate: recoveries[index],
                factor_loading: factor_loadings[index],
                survival_curve: SurvivalCurve::from_piecewise_hazard(&[5.0], &[hazards[index]]),
            })
            .collect(),
        risk_free_rate: 0.03,
        maturity: 5.0,
        payment_freq: 4,
        // Normalized name losses are [0.06, 0.10, 0.12, 0.10].
        loss_unit: 0.02,
        factor_integration_points: 64,
    }
}

fn near_unit_loading_model(factor_loading: f64) -> HeterogeneousSyntheticCdo {
    HeterogeneousSyntheticCdo {
        names: (0..10)
            .map(|_| CdoReferenceName {
                notional: 1.0,
                recovery_rate: 0.40,
                factor_loading,
                survival_curve: SurvivalCurve::new(vec![(5.0, 0.98)]),
            })
            .collect(),
        risk_free_rate: 0.03,
        maturity: 5.0,
        payment_freq: 4,
        loss_unit: 0.06,
        factor_integration_points: 64,
    }
}

fn tranche(attachment: f64, detachment: f64) -> CdoTranche {
    CdoTranche {
        attachment,
        detachment,
        notional: 1.0,
        spread: 0.0,
    }
}

#[test]
fn heterogeneous_curves_and_loadings_match_scipy_and_financepy() {
    let model = common_recovery_model();
    let tranche = tranche(0.12, 0.36);
    let actual = model.expected_loss_fraction(&tranche, 5.0).unwrap();

    // SciPy roots_hermitenorm(64/128/256) + exact Poisson-binomial recursion:
    let scipy_reference = 0.042_885_742_684_940_12;
    assert_relative_eq!(actual, scipy_reference, epsilon = 5.0e-12);

    // FinancePy 1.0.1 `tranche_surv_prob_recursion`, 40_000 factor steps,
    // gives 0.042885733853562646. Its midpoint integration is truncated to
    // [-6, 6], accounting for the 8.9e-9 difference from the full-normal
    // SciPy target. This is a cross-package model check, not the precision
    // budget used for the assertion above.
    let financepy_reference = 0.042_885_733_853_562_646;
    assert!((actual - financepy_reference).abs() < 9.1e-9);

    // FinancePy's base-correlation convention prices the two equity pieces
    // separately. With rho_A=0.12 and rho_D=0.38 it gives
    // 0.030542677155636966 on the same 40,000-step truncated factor grid;
    // full-normal SciPy quadrature gives the target asserted here.
    let correlations = BaseCorrelationPair {
        attachment_correlation: 0.12,
        detachment_correlation: 0.38,
    };
    let base_actual = model
        .expected_loss_fraction_base_correlation(&tranche, 5.0, correlations)
        .unwrap();
    let base_scipy_reference = 0.030_542_659_223_730_4;
    assert_relative_eq!(base_actual, base_scipy_reference, epsilon = 5.0e-12);
    let base_financepy_reference = 0.030_542_677_155_636_966;
    assert!((base_actual - base_financepy_reference).abs() < 1.82e-8);
}

#[test]
fn differing_exposures_recoveries_and_loadings_match_exact_enumeration() {
    let model = fully_heterogeneous_model();
    let tranche = tranche(0.08, 0.24);

    // Independent SciPy 128-node standard-normal quadrature. Conditional
    // losses were evaluated both by convolution and explicit enumeration of
    // all 2^4 default states.
    let expected_loss_fraction = 0.122_256_978_757_031_7;
    assert_relative_eq!(
        model.expected_loss_fraction(&tranche, 5.0).unwrap(),
        expected_loss_fraction,
        epsilon = 5.0e-12
    );

    // The same independent program applies the engine's quarterly midpoint
    // protection discounting and trapezoidal outstanding-notional premium
    // accrual at every payment date.
    let protection_reference = 0.113_043_578_590_296_32;
    let unit_annuity_reference = 4.366_093_039_976_794;
    let fair_spread_reference = 0.025_891_243_625_650_533;
    assert_relative_eq!(
        model.protection_leg_pv(&tranche).unwrap(),
        protection_reference,
        epsilon = 5.0e-12
    );
    assert_relative_eq!(
        model.premium_leg_pv(&tranche, 1.0).unwrap(),
        unit_annuity_reference,
        epsilon = 2.0e-11
    );
    assert_relative_eq!(
        model.fair_spread(&tranche).unwrap(),
        fair_spread_reference,
        epsilon = 2.0e-12
    );
}

#[test]
fn base_correlation_matches_independent_equity_tranche_difference() {
    let model = fully_heterogeneous_model();
    let tranche = tranche(0.08, 0.24);
    let correlations = BaseCorrelationPair {
        attachment_correlation: 0.18,
        detachment_correlation: 0.52,
    };

    // Independent SciPy calculation prices the [0,8%] equity tranche with
    // beta=sqrt(0.18), the [0,24%] equity tranche with beta=sqrt(0.52), and
    // subtracts their pool expected losses.
    assert_relative_eq!(
        model
            .expected_loss_fraction_base_correlation(&tranche, 5.0, correlations)
            .unwrap(),
        0.109_151_058_477_937_27,
        epsilon = 5.0e-12
    );
    assert_relative_eq!(
        model
            .protection_leg_pv_base_correlation(&tranche, correlations)
            .unwrap(),
        0.101_001_911_776_760_04,
        epsilon = 5.0e-12
    );
    assert_relative_eq!(
        model
            .premium_leg_pv_base_correlation(&tranche, 1.0, correlations)
            .unwrap(),
        4.391_377_364_905_903,
        epsilon = 2.0e-11
    );
    assert_relative_eq!(
        model
            .fair_spread_base_correlation(&tranche, correlations)
            .unwrap(),
        0.023_000_052_918_231_564,
        epsilon = 2.0e-12
    );
}

#[test]
fn flat_base_correlation_reduces_to_the_one_factor_model() {
    let mut model = fully_heterogeneous_model();
    let asset_correlation: f64 = 0.36;
    for name in &mut model.names {
        name.factor_loading = asset_correlation.sqrt();
    }
    let tranche = tranche(0.08, 0.24);
    let correlations = BaseCorrelationPair {
        attachment_correlation: asset_correlation,
        detachment_correlation: asset_correlation,
    };

    assert_relative_eq!(
        model.expected_loss_fraction(&tranche, 5.0).unwrap(),
        model
            .expected_loss_fraction_base_correlation(&tranche, 5.0, correlations)
            .unwrap(),
        epsilon = 2.0e-13
    );
    assert_relative_eq!(
        model.fair_spread(&tranche).unwrap(),
        model
            .fair_spread_base_correlation(&tranche, correlations)
            .unwrap(),
        epsilon = 2.0e-13
    );
}

#[test]
fn tranche_partition_recovers_unconditional_portfolio_loss() {
    let model = fully_heterogeneous_model();
    let boundaries = [0.0, 0.08, 0.24, 1.0];
    let tranche_loss_sum = boundaries
        .windows(2)
        .map(|points| {
            let mut part = tranche(points[0], points[1]);
            // With tranche notional equal to width, the currency result is the
            // expected loss as a fraction of portfolio notional.
            part.notional = part.width();
            model.expected_tranche_loss(&part, 5.0).unwrap()
        })
        .sum::<f64>();
    assert_relative_eq!(
        tranche_loss_sum,
        model.portfolio_expected_loss(5.0).unwrap(),
        epsilon = 3.0e-12
    );
}

#[test]
fn npv_is_zero_at_the_computed_fair_spread() {
    let model = fully_heterogeneous_model();
    let mut tranche = tranche(0.08, 0.24);
    tranche.spread = model.fair_spread(&tranche).unwrap();
    assert_relative_eq!(model.npv(&tranche).unwrap(), 0.0, epsilon = 3.0e-15);

    let correlations = BaseCorrelationPair {
        attachment_correlation: 0.18,
        detachment_correlation: 0.52,
    };
    tranche.spread = model
        .fair_spread_base_correlation(&tranche, correlations)
        .unwrap();
    assert_relative_eq!(
        model.npv_base_correlation(&tranche, correlations).unwrap(),
        0.0,
        epsilon = 3.0e-15
    );
}

#[test]
fn common_factor_quadrature_is_converged_on_the_pricing_grid() {
    let model_64 = fully_heterogeneous_model();
    let mut model_128 = model_64.clone();
    model_128.factor_integration_points = 128;
    let tranche = tranche(0.08, 0.24);
    let correlations = BaseCorrelationPair {
        attachment_correlation: 0.18,
        detachment_correlation: 0.52,
    };

    // These are direct locked-grid refinement budgets, separate from the
    // independent-oracle residuals in the reference tests above.
    assert_relative_eq!(
        model_64.expected_loss_fraction(&tranche, 5.0).unwrap(),
        model_128.expected_loss_fraction(&tranche, 5.0).unwrap(),
        epsilon = 2.0e-12
    );
    assert_relative_eq!(
        model_64
            .expected_loss_fraction_base_correlation(&tranche, 5.0, correlations)
            .unwrap(),
        model_128
            .expected_loss_fraction_base_correlation(&tranche, 5.0, correlations)
            .unwrap(),
        epsilon = 2.0e-12
    );
    assert_relative_eq!(
        model_64.fair_spread(&tranche).unwrap(),
        model_128.fair_spread(&tranche).unwrap(),
        epsilon = 2.0e-12
    );
}

#[test]
fn near_unit_loading_is_automatically_refined_to_the_scipy_reference() {
    let model = near_unit_loading_model(0.99);
    let tranche = tranche(0.03, 0.06);

    // Independent SciPy roots_hermitenorm(256) plus exact conditional
    // binomial probabilities. A fixed 64-node Legendre rule gives
    // 0.0291361544178759 and is materially wrong; automatic refinement must
    // reach the converged value instead.
    let reference = 0.032_251_600_545_289_5;
    assert_relative_eq!(
        model.expected_loss_fraction(&tranche, 5.0).unwrap(),
        reference,
        epsilon = 5.0e-12
    );
}

#[test]
fn near_unit_base_correlation_is_automatically_refined() {
    let model = near_unit_loading_model(0.0);
    let tranche = tranche(0.03, 0.06);
    let correlations = BaseCorrelationPair {
        attachment_correlation: 0.99_f64.powi(2),
        detachment_correlation: 0.99_f64.powi(2),
    };

    // A flat base-correlation pair reduces to the same beta=0.99 one-factor
    // model, exercising both separately refined equity-tranche integrations.
    assert_relative_eq!(
        model
            .expected_loss_fraction_base_correlation(&tranche, 5.0, correlations)
            .unwrap(),
        0.032_251_600_545_289_5,
        epsilon = 5.0e-12
    );
}

#[test]
fn factor_integration_reports_nonconvergence_at_the_safe_cap() {
    let mut model = near_unit_loading_model(0.9999);
    model.factor_integration_points = 2_048;
    let error = model
        .expected_loss_fraction(&tranche(0.03, 0.06), 5.0)
        .unwrap_err();
    assert!(
        error.to_string().contains("did not converge"),
        "unexpected error: {error}"
    );
}

#[test]
fn implausibly_low_factor_order_is_rejected() {
    let mut model = fully_heterogeneous_model();
    model.factor_integration_points = 31;
    let error = model
        .expected_loss_fraction(&tranche(0.08, 0.24), 5.0)
        .unwrap_err();
    assert!(error.to_string().contains("must be in [32, 2048]"));
}

#[test]
fn directly_mutated_malformed_survival_curves_are_rejected() {
    let malformed_nodes = [
        vec![],
        vec![(f64::NAN, 0.9)],
        vec![(f64::INFINITY, 0.9)],
        vec![(0.0, 0.9)],
        vec![(-1.0, 0.9)],
        vec![(1.0, f64::NAN)],
        vec![(1.0, 0.0)],
        vec![(1.0, 1.01)],
        vec![(2.0, 0.9), (1.0, 0.8)],
        vec![(1.0, 0.9), (1.0, 0.8)],
        vec![(1.0, 0.8), (2.0, 0.9)],
    ];
    for nodes in malformed_nodes {
        let mut model = fully_heterogeneous_model();
        model.names[0].survival_curve.tenors = nodes;
        let error = model
            .expected_loss_fraction(&tranche(0.08, 0.24), 5.0)
            .unwrap_err();
        assert!(
            error.to_string().contains("survival"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn non_commensurate_loss_grid_is_rejected_instead_of_rounded() {
    let mut model = fully_heterogeneous_model();
    model.loss_unit = 0.03;
    let error = model
        .expected_loss_fraction(&tranche(0.08, 0.24), 5.0)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not a positive integer multiple")
    );
}
