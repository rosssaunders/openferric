use approx::assert_relative_eq;
use openferric::credit::{Cds, CdsIndex, GaussianCopula, NthToDefaultBasket, SurvivalCurve};
use openferric::rates::YieldCurve;

fn flat_curve(rate: f64) -> YieldCurve {
    YieldCurve::new(vec![(10.0, (-rate * 10.0).exp())])
}

#[test]
fn heterogeneous_index_quote_reprices_common_running_coupon() {
    let mut index = CdsIndex {
        constituents: vec![
            Cds {
                notional: 2_000_000.0,
                spread: 0.01,
                maturity: 5.0,
                recovery_rate: 0.4,
                payment_freq: 4,
            },
            Cds {
                notional: 1_000_000.0,
                spread: 0.01,
                maturity: 5.0,
                recovery_rate: 0.2,
                payment_freq: 4,
            },
        ],
        weights: vec![0.3, 0.7],
    };
    let curves = [
        SurvivalCurve::from_piecewise_hazard(&[5.0], &[0.01]),
        SurvivalCurve::from_piecewise_hazard(&[5.0], &[0.4]),
    ];
    let discount = flat_curve(0.03);
    let mut protection = 0.0;
    let mut annuity = 0.0;
    for (hazard, recovery, weighted_notional) in [(0.01_f64, 0.4, 600_000.0), (0.4, 0.2, 700_000.0)]
    {
        for period in 1..=20 {
            let end = period as f64 * 0.25;
            let start = end - 0.25;
            let default_probability = (-hazard * start).exp() * -(-hazard * 0.25).exp_m1();
            let midpoint_discount = (-0.03 * (end - 0.125)).exp();
            protection +=
                weighted_notional * (1.0 - recovery) * midpoint_discount * default_probability;
            annuity += weighted_notional
                * (0.25 * (-(0.03 + hazard) * end).exp()
                    + 0.125 * midpoint_discount * default_probability);
        }
    }
    let expected = protection / annuity;
    let spread = index.fair_spread(&discount, &curves);
    assert_relative_eq!(spread, expected, epsilon = 3.0e-15);
    for constituent in &mut index.constituents {
        constituent.spread = spread;
    }
    assert_relative_eq!(index.npv(&discount, &curves), 0.0, epsilon = 3.0e-9);
}

#[test]
fn default_free_nth_to_default_still_owes_premium() {
    let basket = NthToDefaultBasket {
        n: 2,
        notional: 1_000_000.0,
        maturity: 1.1,
        recovery_rate: 0.4,
        payment_freq: 4,
    };
    let curves = vec![SurvivalCurve::from_piecewise_hazard(&[5.0], &[0.0]); 3];
    let expected_annuity = [0.25_f64, 0.5, 0.75, 1.0]
        .iter()
        .map(|time| 0.25 * (-0.03 * time).exp())
        .sum::<f64>()
        + 0.1 * (-0.033_f64).exp();
    assert_eq!(basket.fair_spread(&flat_curve(0.03), &curves), 0.0);
    assert_relative_eq!(
        basket.npv(0.01, &flat_curve(0.03), &curves),
        -10_000.0 * expected_annuity,
        epsilon = 3.0e-10
    );
}

#[test]
fn survival_underflow_cannot_turn_distressed_name_default_free() {
    for curve in [
        SurvivalCurve::from_piecewise_hazard(&[1.0], &[1000.0]),
        SurvivalCurve::new(vec![(1.0, 0.0)]),
    ] {
        assert!(!curve.tenors.is_empty());
        assert_relative_eq!(
            curve.survival_prob(1.0),
            1.0e-12,
            max_relative = 8.0 * f64::EPSILON
        );
        let cds = Cds {
            notional: 1_000_000.0,
            spread: 0.0,
            maturity: 1.0,
            recovery_rate: 0.4,
            payment_freq: 4,
        };
        assert_relative_eq!(
            cds.protection_leg_pv(&flat_curve(0.0), &curve),
            600_000.0 * (1.0 - 1.0e-12),
            epsilon = 2.0e-9
        );
    }
}

#[test]
fn unit_factor_loading_is_exact_comonotonic_default() {
    use rand::SeedableRng;
    let survival = SurvivalCurve::from_piecewise_hazard(&[5.0], &[0.02]);
    for loading in [-1.0, 1.0] {
        let copula = GaussianCopula::new(loading);
        let mut random = rand::rngs::StdRng::seed_from_u64(42);
        let simulated = copula.simulate_homogeneous(3, &survival, &mut random);
        assert_eq!(simulated.default_times[0], simulated.default_times[1]);
        assert_eq!(simulated.default_times[1], simulated.default_times[2]);
    }
}

#[test]
fn lhp_credit_loss_preserves_degenerate_probability_masses() {
    use openferric::credit::{CdoTranche, vasicek_portfolio_loss_cdf};
    for correlation in [0.0, 0.5, 1.0] {
        assert_eq!(vasicek_portfolio_loss_cdf(0.0, 0.0, 0.4, correlation), 1.0);
        assert_eq!(vasicek_portfolio_loss_cdf(0.0, 0.2, 1.0, correlation), 1.0);
        assert_eq!(vasicek_portfolio_loss_cdf(-0.1, 0.0, 0.4, correlation), 0.0);
        assert_eq!(vasicek_portfolio_loss_cdf(0.3, 1.0, 0.4, correlation), 0.0);
    }
    for loss in [0.0, 0.1, 0.5] {
        assert_eq!(vasicek_portfolio_loss_cdf(loss, 0.2, 0.4, 1.0), 0.8);
    }
    let tranche = CdoTranche {
        attachment: 0.1,
        detachment: 0.3,
        notional: 100.0,
        spread: 0.01,
    };
    assert_relative_eq!(
        tranche.expected_loss_fraction(0.2, 0.4, 1.0),
        0.2,
        epsilon = 1.0e-15
    );
    assert_eq!(tranche.expected_loss_fraction(1.0, 0.4, 0.5), 1.0);
}
