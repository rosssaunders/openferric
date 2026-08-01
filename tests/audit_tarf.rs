//! Audit regression tests for TARF knock-out semantics.
//!
//! Guards against a bug where the decumulator TARF applied the *upside* KO
//! check `s >= ko_barrier` (the standard-TARF direction), so a downside
//! barrier knocked out every path at the first fixing and an upside barrier
//! cancelled the product exactly in the states where the holder was losing,
//! overstating decumulator value.
//!
//! Reference values are external anchors: analytic Black-Scholes prices from
//! `openferric::engines::analytic::black_scholes::bs_price` combined into
//! closed-form strips of leveraged forwards (each TARF fixing without KO and
//! target is a leveraged synthetic forward, Wystup "FX Options and Structured
//! Products" 2nd ed.). No MC self-references.

use openferric::core::OptionType;
use openferric::engines::analytic::black_scholes::{bs_price, norm_cdf};
use openferric::instruments::tarf::Tarf;
use openferric::pricing::tarf::tarf_mc_price;

/// Effectively unreachable target profit (must stay finite for `validate()`).
const NO_TARGET: f64 = 1e15;
/// Universal "no KO barrier" sentinel for both TARF types.
const NO_BARRIER: f64 = f64::INFINITY;

fn weekly_fixings() -> Vec<f64> {
    (1..=52).map(|w| w as f64 / 52.0).collect()
}

/// Closed-form value of a TARF with no KO barrier and unreachable target: a
/// strip of leveraged forwards. `gain`/`lose` are the option payoffs
/// replicating the winning and (leveraged) losing legs of each fixing:
/// Standard = Call/Put, Decumulator = Put/Call. `bs_price` handles
/// discounting, matching the `exp(-rate * t_i)` factor in the MC.
#[allow(clippy::too_many_arguments)]
fn strip_closed_form(
    gain: OptionType,
    lose: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    notional: f64,
    leverage: f64,
    fixings: &[f64],
) -> f64 {
    fixings
        .iter()
        .map(|&t| {
            notional
                * (bs_price(gain, spot, strike, rate, q, vol, t)
                    - leverage * bs_price(lose, spot, strike, rate, q, vol, t))
        })
        .sum()
}

/// Discounted Black-Scholes value of `(S_T - strike)` restricted to
/// `strike < S_T < upper`.  This is the difference of the two truncated
/// lognormal zeroth and first moments, not an approximation by a vanilla.
#[allow(clippy::too_many_arguments)]
fn truncated_call_band(
    spot: f64,
    strike: f64,
    upper: f64,
    rate: f64,
    q: f64,
    vol: f64,
    t: f64,
) -> f64 {
    let root_t = t.sqrt();
    let d = |level: f64| {
        let d1 = ((spot / level).ln() + (rate - q + 0.5 * vol * vol) * t) / (vol * root_t);
        (d1, d1 - vol * root_t)
    };
    let (d1_k, d2_k) = d(strike);
    let (d1_b, d2_b) = d(upper);
    spot * (-q * t).exp() * (norm_cdf(d1_k) - norm_cdf(d1_b))
        - strike * (-rate * t).exp() * (norm_cdf(d2_k) - norm_cdf(d2_b))
}

/// Discounted Black-Scholes value of `(strike - S_T)` restricted to
/// `lower < S_T < strike`.
#[allow(clippy::too_many_arguments)]
fn truncated_put_band(
    spot: f64,
    lower: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    t: f64,
) -> f64 {
    let root_t = t.sqrt();
    let d = |level: f64| {
        let d1 = ((spot / level).ln() + (rate - q + 0.5 * vol * vol) * t) / (vol * root_t);
        (d1, d1 - vol * root_t)
    };
    let (d1_b, d2_b) = d(lower);
    let (d1_k, d2_k) = d(strike);
    strike * (-rate * t).exp() * (norm_cdf(-d2_k) - norm_cdf(-d2_b))
        - spot * (-q * t).exp() * (norm_cdf(-d1_k) - norm_cdf(-d1_b))
}

/// The decumulator KO barrier is a downside barrier: it must trigger when
/// spot falls through it and must NOT trigger when spot rallies away from it.
///
/// Under the inverted implementation (`s >= 80` with spot 100) every path
/// knocked out at fixing 1 regardless of direction: prob_ko = 1.0,
/// avg_fixings = 1.0, price = 0.0 in both scenarios below.
#[test]
fn decumulator_downside_ko_triggers_on_falling_spot_not_on_rally() {
    let tarf = Tarf::decumulator(100.0, 1000.0, 80.0, NO_TARGET, 2.0, weekly_fixings());

    // Strong downward risk-neutral drift (r = 0, q = 30%): log-drift is
    // -0.311/yr, so P(S_1y <= 80) = Phi(0.59) ~ 0.72 before even counting
    // intermediate fixings. The downside KO must fire on most paths.
    let falling = tarf_mc_price(&tarf, 100.0, 0.0, 0.30, 0.15, 20_000, 42).unwrap();

    // Strong upward drift (r = 30%, q = 0): the static lower-barrier hit
    // probability (b/S0)^(2 nu / sigma^2) = 0.8^25.7 ~ 0.3% bounds the
    // continuously-monitored case from above; discrete weekly monitoring is
    // rarer still.
    let rallying = tarf_mc_price(&tarf, 100.0, 0.30, 0.0, 0.15, 20_000, 42).unwrap();

    assert!(
        falling.prob_ko_hit > 0.5,
        "downside KO must fire on falling paths, got prob_ko = {}",
        falling.prob_ko_hit
    );
    assert!(
        rallying.prob_ko_hit < 0.05,
        "downside KO must not fire on rallying paths, got prob_ko = {}",
        rallying.prob_ko_hit
    );
    // Not an instant fixing-1 knock-out in either scenario.
    assert!(
        falling.avg_fixings > 1.5,
        "avg_fixings = {} suggests instant KO at fixing 1",
        falling.avg_fixings
    );
    assert!(
        rallying.avg_fixings > 10.0,
        "avg_fixings = {} suggests spurious early KO",
        rallying.avg_fixings
    );
}

/// Degenerate limit: KO barrier unreachable (INFINITY sentinel) + target
/// unreachable reduces both TARF types to closed-form strips of leveraged
/// forwards. Anchors are analytic Black-Scholes strips (see module docs); the
/// GBM stepping is exact-in-distribution, so the MC estimate is unbiased and
/// a 4-standard-error band at a fixed seed is a sound acceptance region.
#[test]
fn tarf_without_ko_and_target_prices_as_forward_strip() {
    let (spot, rate, q, vol) = (100.0, 0.03, 0.0, 0.15);
    let (strike, notional, leverage) = (100.0, 1000.0, 2.0);
    let fixings = weekly_fixings();

    // Standard: each fixing pays N * [(S-K)+ - L*(K-S)+]  => N * [C - L*P].
    let std_closed = strip_closed_form(
        OptionType::Call,
        OptionType::Put,
        spot,
        strike,
        rate,
        q,
        vol,
        notional,
        leverage,
        &fixings,
    );
    // Decumulator: each fixing pays N * [(K-S)+ - L*(S-K)+] => N * [P - L*C].
    let dec_closed = strip_closed_form(
        OptionType::Put,
        OptionType::Call,
        spot,
        strike,
        rate,
        q,
        vol,
        notional,
        leverage,
        &fixings,
    );

    let std_tarf = Tarf::standard(
        strike,
        notional,
        NO_BARRIER,
        NO_TARGET,
        leverage,
        fixings.clone(),
    );
    let dec_tarf = Tarf::decumulator(strike, notional, NO_BARRIER, NO_TARGET, leverage, fixings);

    let std_mc = tarf_mc_price(&std_tarf, spot, rate, q, vol, 100_000, 42).unwrap();
    let dec_mc = tarf_mc_price(&dec_tarf, spot, rate, q, vol, 100_000, 42).unwrap();

    assert!(
        (std_mc.price - std_closed).abs() < 4.0 * std_mc.std_error,
        "standard strip: mc = {} +/- {}, closed form = {}",
        std_mc.price,
        std_mc.std_error,
        std_closed
    );
    assert!(
        (dec_mc.price - dec_closed).abs() < 4.0 * dec_mc.std_error,
        "decumulator strip: mc = {} +/- {}, closed form = {}",
        dec_mc.price,
        dec_mc.std_error,
        dec_closed
    );
    // The INFINITY sentinel must never register a KO for either type (for the
    // decumulator this exercises the is_finite() guard on `s <= barrier`).
    assert_eq!(std_mc.prob_ko_hit, 0.0);
    assert_eq!(dec_mc.prob_ko_hit, 0.0);
    assert_eq!(std_mc.avg_fixings, 52.0);
    assert_eq!(dec_mc.avg_fixings, 52.0);
}

/// A KO barrier cancels the remaining fixings precisely in the states where
/// the holder is accumulating gains, so enabling a finite barrier must
/// REDUCE holder value for BOTH types. Under the inverted implementation an
/// upside barrier RAISED decumulator value (by truncating losing rally
/// states) instead of reducing it.
#[test]
fn finite_ko_barrier_reduces_holder_value_for_both_types() {
    let (spot, rate, q, vol) = (100.0, 0.03, 0.0, 0.15);
    let fixings = weekly_fixings();
    let price = |t: &Tarf| tarf_mc_price(t, spot, rate, q, vol, 50_000, 42).unwrap();

    let std_no_ko = price(&Tarf::standard(
        100.0,
        1000.0,
        NO_BARRIER,
        NO_TARGET,
        2.0,
        fixings.clone(),
    ));
    let std_ko = price(&Tarf::standard(
        100.0,
        1000.0,
        120.0,
        NO_TARGET,
        2.0,
        fixings.clone(),
    ));
    let dec_no_ko = price(&Tarf::decumulator(
        100.0,
        1000.0,
        NO_BARRIER,
        NO_TARGET,
        2.0,
        fixings.clone(),
    ));
    let dec_ko = price(&Tarf::decumulator(
        100.0, 1000.0, 80.0, NO_TARGET, 2.0, fixings,
    ));

    // 3 * (sum of standard errors) exceeds the standard error of the
    // difference of independent estimates, so this is a conservative bound.
    assert!(
        std_ko.price < std_no_ko.price - 3.0 * (std_ko.std_error + std_no_ko.std_error),
        "upside KO must reduce standard TARF value: with KO {} vs without {}",
        std_ko.price,
        std_no_ko.price
    );
    assert!(
        dec_ko.price < dec_no_ko.price - 3.0 * (dec_ko.std_error + dec_no_ko.std_error),
        "downside KO must reduce decumulator value: with KO {} vs without {}",
        dec_ko.price,
        dec_no_ko.price
    );
    // The KO genuinely fires and removes remaining fixings.
    assert!(std_ko.prob_ko_hit > 0.0);
    assert!(dec_ko.prob_ko_hit > 0.0);
    assert!(std_ko.avg_fixings < 52.0);
    assert!(dec_ko.avg_fixings < 52.0);
}

/// The KO check precedes the fixing P&L: the breaching fixing pays nothing.
/// With a single fixing and the barrier epsilon-close to the strike, the
/// surviving payoff has an exact truncated-lognormal decomposition:
///   Standard   = N * [Call restricted to K<S<B - L*Put]
///   Decumulator = N * [Put restricted to B<S<K - L*Call]
/// If the breaching fixing were paid (wrong ordering), the winning leg would
/// be included and both prices would be strongly positive instead.
#[test]
fn ko_breaching_fixing_pays_nothing_single_fixing_closed_form() {
    let (spot, rate, q, vol, t) = (100.0, 0.03, 0.0, 0.15, 1.0);
    let (strike, notional, leverage) = (100.0, 1000.0, 1.0);
    let eps = 1e-4;

    let std_tarf = Tarf::standard(strike, notional, strike + eps, NO_TARGET, leverage, vec![t]);
    let std_mc = tarf_mc_price(&std_tarf, spot, rate, q, vol, 200_000, 42).unwrap();
    let std_expected = notional
        * (truncated_call_band(spot, strike, strike + eps, rate, q, vol, t)
            - leverage * bs_price(OptionType::Put, spot, strike, rate, q, vol, t));
    assert!(
        (std_mc.price - std_expected).abs() < 4.0 * std_mc.std_error,
        "standard KO ordering: mc = {} +/- {}, exact truncated-moment value = {}",
        std_mc.price,
        std_mc.std_error,
        std_expected
    );

    let dec_tarf = Tarf::decumulator(strike, notional, strike - eps, NO_TARGET, leverage, vec![t]);
    let dec_mc = tarf_mc_price(&dec_tarf, spot, rate, q, vol, 200_000, 42).unwrap();
    let dec_expected = notional
        * (truncated_put_band(spot, strike - eps, strike, rate, q, vol, t)
            - leverage * bs_price(OptionType::Call, spot, strike, rate, q, vol, t));
    assert!(
        (dec_mc.price - dec_expected).abs() < 4.0 * dec_mc.std_error,
        "decumulator KO ordering: mc = {} +/- {}, exact truncated-moment value = {}",
        dec_mc.price,
        dec_mc.std_error,
        dec_expected
    );
}

/// A finite KO barrier must sit strictly on the KO side of the strike for the
/// product type; INFINITY (no barrier) is accepted for both types.
#[test]
fn validation_enforces_type_aware_barrier_side() {
    let fixings = weekly_fixings();
    assert!(
        Tarf::standard(100.0, 1000.0, 80.0, 50_000.0, 2.0, fixings.clone())
            .validate()
            .is_err(),
        "standard TARF with downside barrier must be rejected"
    );
    assert!(
        Tarf::decumulator(100.0, 1000.0, 120.0, 50_000.0, 2.0, fixings.clone())
            .validate()
            .is_err(),
        "decumulator TARF with upside barrier must be rejected"
    );
    assert!(
        Tarf::standard(100.0, 1000.0, NO_BARRIER, 50_000.0, 2.0, fixings.clone())
            .validate()
            .is_ok()
    );
    assert!(
        Tarf::decumulator(100.0, 1000.0, NO_BARRIER, 50_000.0, 2.0, fixings)
            .validate()
            .is_ok()
    );
}
