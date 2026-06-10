//! Cross-engine consistency tests.
//!
//! Every test in this file prices the SAME instrument through several
//! independent engines and asserts agreement. This catches sign/convention
//! bugs, drift/discounting mistakes, and payoff wiring errors that no
//! single-engine reference test can see.
//!
//! Engine pairs covered:
//! 1. European vanillas: Black-Scholes analytic vs CRR binomial tree vs
//!    trinomial tree vs Crank-Nicolson PDE vs Monte Carlo vs Carr-Madan FFT
//!    (exact Black-Scholes characteristic function, puts via put-call parity).
//! 2. American puts: CRR binomial vs `AmericanBinomialEngine` vs trinomial vs
//!    Crank-Nicolson PDE vs Longstaff-Schwartz LSM. American calls with q = 0
//!    and r >= 0 must equal European calls (no early exercise).
//! 3. Barriers: Reiner-Rubinstein analytic vs discretely-monitored MC using
//!    the Broadie-Glasserman-Kou barrier shift, one-sided discrete-vs-
//!    continuous ordering, and in + out = vanilla parity inside each engine.
//! 4. Asians: geometric discrete closed form vs MC geometric payoff.
//! 5. Digitals: cash-or-nothing analytic vs an exact-terminal-GBM MC.
//!
//! Tolerances:
//! - Deterministic engines (tree/PDE/FFT): documented per test, generally
//!   relative 2e-3 plus a small absolute floor for deep out-of-the-money
//!   points whose prices are ~1e-12 (relative error is meaningless there).
//! - Monte Carlo: |mc - reference| <= 4 * stderr + 5e-7. The absolute floor
//!   covers deep-OTM points below Monte Carlo resolution: when the exercise
//!   probability is ~1e-7, 60k paths produce zero payoffs (stderr exactly 0)
//!   while the analytic price is a positive number of that magnitude.
//!
//! All Monte Carlo engines use fixed seeds; the suite is fully deterministic
//! and uses no external data.

use openferric::core::types::AsianSpec;
use openferric::core::{Averaging, OptionType, PricingEngine, StrikeType};
use openferric::engines::analytic::{
    BarrierAnalyticEngine, BlackScholesEngine, DigitalAnalyticEngine, GeometricAsianEngine,
};
use openferric::engines::fft::{BlackScholesCharFn, CarrMadanParams, carr_madan_price_at_strikes};
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::engines::monte_carlo::MonteCarloPricingEngine;
use openferric::engines::numerical::AmericanBinomialEngine;
use openferric::engines::pde::CrankNicolsonEngine;
use openferric::engines::tree::{BinomialTreeEngine, TrinomialTreeEngine};
use openferric::instruments::{AsianOption, BarrierOption, CashOrNothingOption, VanillaOption};
use openferric::market::Market;
use openferric::math::fast_norm::beasley_springer_moro_inv_cdf;
use openferric::math::fast_rng::{Xoshiro256PlusPlus, uniform_open01};

/// Common strike for the moneyness grid (S/K is varied through the spot).
const STRIKE: f64 = 100.0;

/// One point of the European consistency grid.
#[derive(Debug, Clone, Copy)]
struct GridPoint {
    option_type: OptionType,
    spot: f64,
    vol: f64,
    rate: f64,
    div: f64,
    expiry: f64,
}

impl GridPoint {
    fn market(&self) -> Market {
        Market::builder()
            .spot(self.spot)
            .rate(self.rate)
            .dividend_yield(self.div)
            .flat_vol(self.vol)
            .build()
            .expect("valid market")
    }

    fn european(&self) -> VanillaOption {
        match self.option_type {
            OptionType::Call => VanillaOption::european_call(STRIKE, self.expiry),
            OptionType::Put => VanillaOption::european_put(STRIKE, self.expiry),
        }
    }

    fn label(&self) -> String {
        format!(
            "{:?} S={} K={} vol={} r={} q={} T={}",
            self.option_type, self.spot, STRIKE, self.vol, self.rate, self.div, self.expiry
        )
    }
}

/// Full 96-point European grid:
/// moneyness S/K in {0.7, 1.0, 1.3} x vol in {0.1, 0.4} x r in {-0.01, 0.05}
/// x q in {0.0, 0.03} x T in {0.25, 2.0} x {Call, Put}.
fn european_grid() -> Vec<GridPoint> {
    let mut points = Vec::with_capacity(96);
    for option_type in [OptionType::Call, OptionType::Put] {
        for spot in [70.0, 100.0, 130.0] {
            for vol in [0.1, 0.4] {
                for rate in [-0.01, 0.05] {
                    for div in [0.0, 0.03] {
                        for expiry in [0.25, 2.0] {
                            points.push(GridPoint {
                                option_type,
                                spot,
                                vol,
                                rate,
                                div,
                                expiry,
                            });
                        }
                    }
                }
            }
        }
    }
    points
}

/// Runs `check(point, black_scholes_reference_price)` over the whole grid.
fn for_each_european_point(check: impl Fn(&GridPoint, f64)) {
    let analytic = BlackScholesEngine::new();
    for point in european_grid() {
        let reference = analytic
            .price(&point.european(), &point.market())
            .expect("analytic pricing succeeds")
            .price;
        check(&point, reference);
    }
}

/// Asserts `|value - reference| <= abs_tol + rel_tol * |reference|` with a
/// failure message carrying every grid parameter.
fn assert_close(engine: &str, label: &str, value: f64, reference: f64, rel_tol: f64, abs_tol: f64) {
    let err = (value - reference).abs();
    let tol = abs_tol + rel_tol * reference.abs();
    assert!(
        value.is_finite() && err <= tol,
        "{engine} disagrees at [{label}]: value={value} reference={reference} err={err:.3e} tol={tol:.3e}"
    );
}

/// Asserts an MC estimate agrees with a reference within 4 standard errors
/// (plus a tiny absolute floor for zero-variance deep-OTM points whose
/// analytic price is below Monte Carlo resolution; see module docs).
fn assert_mc_close(engine: &str, label: &str, value: f64, stderr: f64, reference: f64) {
    let err = (value - reference).abs();
    let tol = 4.0 * stderr + 5e-7;
    assert!(
        value.is_finite() && err <= tol,
        "{engine} disagrees at [{label}]: value={value} reference={reference} stderr={stderr} err={err:.3e} tol=4*stderr+5e-7={tol:.3e}"
    );
}

// ---------------------------------------------------------------------------
// 1. European vanillas across analytic / tree / PDE / MC / FFT engines.
// ---------------------------------------------------------------------------

#[test]
fn european_binomial_tree_matches_black_scholes() {
    // CRR converges ~O(1/n) with odd/even oscillation; 1000 steps puts the
    // discretization error comfortably inside relative 2e-3 (+ small absolute
    // floor for prices ~1e-12 where relative error is meaningless).
    let engine = BinomialTreeEngine::new(1000);
    for_each_european_point(|point, reference| {
        let price = engine
            .price(&point.european(), &point.market())
            .expect("binomial pricing succeeds")
            .price;
        assert_close(
            "CRR binomial(1000)",
            &point.label(),
            price,
            reference,
            2e-3,
            2e-3,
        );
    });
}

#[test]
fn european_trinomial_tree_matches_black_scholes() {
    let engine = TrinomialTreeEngine::new(800);
    for_each_european_point(|point, reference| {
        let price = engine
            .price(&point.european(), &point.market())
            .expect("trinomial pricing succeeds")
            .price;
        assert_close(
            "trinomial(800)",
            &point.label(),
            price,
            reference,
            2e-3,
            2e-3,
        );
    });
}

/// Vol-adaptive Crank-Nicolson grid. A fixed S_max = 5K wastes most of the
/// 400 space nodes when vol*sqrt(T) is small (dS becomes coarse relative to
/// the narrow gamma region around the strike and the ATM error blows past
/// 2e-3), while high-vol long-dated points need S_max several stddevs above
/// spot to control truncation. Put S_max ~3.5 lognormal stddevs above
/// max(spot, K); the call upper boundary is the exact asymptote, so modest
/// truncation distances are safe.
fn crank_nicolson_for(point: &GridPoint) -> CrankNicolsonEngine {
    let sigma_sqrt_t = point.vol * point.expiry.sqrt();
    let multiplier = ((point.spot / STRIKE).max(1.0) * (3.5 * sigma_sqrt_t).exp()).clamp(1.5, 8.0);
    CrankNicolsonEngine::new(400, 400).with_s_max_multiplier(multiplier)
}

#[test]
fn european_crank_nicolson_matches_black_scholes() {
    for_each_european_point(|point, reference| {
        let price = crank_nicolson_for(point)
            .price(&point.european(), &point.market())
            .expect("crank-nicolson pricing succeeds")
            .price;
        assert_close(
            "Crank-Nicolson(400x400)",
            &point.label(),
            price,
            reference,
            2e-3,
            2e-3,
        );
    });
}

#[test]
fn european_monte_carlo_matches_black_scholes_within_4_stderr() {
    // GBM path generation uses exact log-Euler increments, so one time step
    // samples the terminal distribution exactly; only sampling error remains.
    let engine = MonteCarloPricingEngine::new(60_000, 1, 20_240_607);
    for_each_european_point(|point, reference| {
        let result = engine
            .price(&point.european(), &point.market())
            .expect("monte carlo pricing succeeds");
        let stderr = result.stderr.expect("monte carlo reports stderr");
        assert_mc_close("MC(60k,1)", &point.label(), result.price, stderr, reference);
    });
}

#[test]
fn european_carr_madan_fft_matches_black_scholes() {
    // Exact Black-Scholes characteristic function through the Carr-Madan
    // pricer (no Heston proxy needed). FFT gives calls; puts via put-call
    // parity: P = C - S e^{-qT} + K e^{-rT}. Default grid (n=4096, eta=0.25,
    // alpha=1.5) resolves vanillas to ~1e-4 absolute; tolerance 1e-3 + 1e-4.
    let params = CarrMadanParams::default();
    for_each_european_point(|point, reference| {
        let cf =
            BlackScholesCharFn::new(point.spot, point.rate, point.div, point.vol, point.expiry);
        let prices = carr_madan_price_at_strikes(
            &cf,
            point.rate,
            point.expiry,
            point.spot,
            &[STRIKE],
            params,
        )
        .expect("carr-madan pricing succeeds");
        let call = prices[0].1;
        let price = match point.option_type {
            OptionType::Call => call,
            OptionType::Put => {
                call - point.spot * (-point.div * point.expiry).exp()
                    + STRIKE * (-point.rate * point.expiry).exp()
            }
        };
        assert_close(
            "Carr-Madan FFT",
            &point.label(),
            price,
            reference,
            1e-4,
            1e-3,
        );
    });
}

// ---------------------------------------------------------------------------
// 2. American exercise: trees vs PDE vs LSM, and the q=0 call no-early-
//    exercise identity.
// ---------------------------------------------------------------------------

/// Smaller American grid: S in {90, 100, 110} x vol in {0.2, 0.4} x
/// q in {0.0, 0.03}, with r = 0.05 and T = 1.
fn american_put_grid() -> Vec<GridPoint> {
    let mut points = Vec::new();
    for spot in [90.0, 100.0, 110.0] {
        for vol in [0.2, 0.4] {
            for div in [0.0, 0.03] {
                points.push(GridPoint {
                    option_type: OptionType::Put,
                    spot,
                    vol,
                    rate: 0.05,
                    div,
                    expiry: 1.0,
                });
            }
        }
    }
    points
}

#[test]
fn american_put_tree_pde_lsm_agree() {
    let reference_tree = BinomialTreeEngine::new(2000);
    let american_binomial = AmericanBinomialEngine::new(2000);
    let trinomial = TrinomialTreeEngine::new(800);
    let pde = CrankNicolsonEngine::new(400, 400).with_s_max_multiplier(5.0);
    let lsm = LongstaffSchwartzEngine::new(40_000, 50, 7_771);

    for point in american_put_grid() {
        let option = VanillaOption::american_put(STRIKE, point.expiry);
        let market = point.market();
        let label = format!("American {}", point.label());

        let reference = reference_tree
            .price(&option, &market)
            .expect("binomial american pricing succeeds")
            .price;

        // Independent CRR implementation: same lattice parameterization, so
        // agreement should be near machine accuracy.
        let alt = american_binomial
            .price(&option, &market)
            .expect("american binomial pricing succeeds")
            .price;
        assert_close(
            "AmericanBinomial(2000) vs CRR(2000)",
            &label,
            alt,
            reference,
            1e-9,
            1e-9,
        );

        let tri = trinomial
            .price(&option, &market)
            .expect("trinomial american pricing succeeds")
            .price;
        assert_close("trinomial(800)", &label, tri, reference, 2e-3, 2e-3);

        let cn = pde
            .price(&option, &market)
            .expect("crank-nicolson american pricing succeeds")
            .price;
        assert_close("Crank-Nicolson(400x400)", &label, cn, reference, 2e-3, 2e-3);

        // LSM carries both Monte Carlo noise and a small low-side regression
        // bias (quadratic basis, 50 exercise dates), so allow 4*stderr plus
        // 0.5% relative for the bias.
        let lsm_result = lsm
            .price(&option, &market)
            .expect("lsm american pricing succeeds");
        let lsm_stderr = lsm_result.stderr.expect("lsm reports stderr");
        let err = (lsm_result.price - reference).abs();
        let tol = 4.0 * lsm_stderr + 5e-3 * reference;
        assert!(
            err <= tol,
            "LSM(40k,50) disagrees at [{label}]: value={} reference={reference} stderr={lsm_stderr} err={err:.3e} tol={tol:.3e}",
            lsm_result.price
        );
    }
}

#[test]
fn american_call_without_dividends_equals_european_call() {
    // With q = 0 and r >= 0 early exercise of a call is never optimal, so the
    // American price must collapse to the European one in every engine.
    // (Deliberately excludes r < 0, where early exercise can be optimal.)
    let analytic = BlackScholesEngine::new();
    let tree = BinomialTreeEngine::new(2000);
    let pde = CrankNicolsonEngine::new(400, 400).with_s_max_multiplier(5.0);
    let lsm = LongstaffSchwartzEngine::new(40_000, 50, 7_772);

    for spot in [80.0, 100.0, 125.0] {
        for vol in [0.2, 0.4] {
            let point = GridPoint {
                option_type: OptionType::Call,
                spot,
                vol,
                rate: 0.05,
                div: 0.0,
                expiry: 1.0,
            };
            let market = point.market();
            let american = VanillaOption::american_call(STRIKE, point.expiry);
            let european = VanillaOption::european_call(STRIKE, point.expiry);
            let label = format!("American-call-q0 {}", point.label());

            let euro_reference = analytic
                .price(&european, &market)
                .expect("analytic pricing succeeds")
                .price;

            // Within one engine the American and European backward inductions
            // must coincide exactly when intrinsic never beats continuation.
            let tree_am = tree.price(&american, &market).expect("tree am").price;
            let tree_eu = tree.price(&european, &market).expect("tree eu").price;
            assert_close(
                "CRR American==European",
                &label,
                tree_am,
                tree_eu,
                0.0,
                1e-9,
            );
            assert_close(
                "CRR(2000) vs analytic",
                &label,
                tree_am,
                euro_reference,
                2e-3,
                2e-3,
            );

            let cn_am = pde.price(&american, &market).expect("pde am").price;
            assert_close(
                "Crank-Nicolson American",
                &label,
                cn_am,
                euro_reference,
                2e-3,
                2e-3,
            );

            let lsm_result = lsm.price(&american, &market).expect("lsm am");
            let lsm_stderr = lsm_result.stderr.expect("lsm reports stderr");
            let err = (lsm_result.price - euro_reference).abs();
            // 4*stderr plus 0.5% relative headroom for LSM regression bias
            // (spurious early exercise adds a small high-side bias here).
            let tol = 4.0 * lsm_stderr + 5e-3 * euro_reference;
            assert!(
                err <= tol,
                "LSM American call disagrees at [{label}]: value={} reference={euro_reference} stderr={lsm_stderr} err={err:.3e} tol={tol:.3e}",
                lsm_result.price
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Barrier options: analytic (continuous monitoring) vs MC (discrete).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum BarrierKind {
    UpOut,
    UpIn,
    DownOut,
    DownIn,
}

fn barrier_option(
    kind: BarrierKind,
    level: f64,
    option_type: OptionType,
    expiry: f64,
) -> BarrierOption {
    let builder = match option_type {
        OptionType::Call => BarrierOption::builder().call(),
        OptionType::Put => BarrierOption::builder().put(),
    };
    let builder = builder.strike(STRIKE).expiry(expiry).rebate(0.0);
    let builder = match kind {
        BarrierKind::UpOut => builder.up_and_out(level),
        BarrierKind::UpIn => builder.up_and_in(level),
        BarrierKind::DownOut => builder.down_and_out(level),
        BarrierKind::DownIn => builder.down_and_in(level),
    };
    builder.build().expect("valid barrier option")
}

#[test]
fn barrier_in_out_parity_analytic_and_mc() {
    // With zero rebate, in + out = vanilla must hold exactly within each
    // engine. For MC the same seed/steps reproduce identical paths across the
    // three pricings, so the identity holds per path and the only slack is
    // floating-point summation order (tolerance 1e-9).
    let spot = 100.0;
    let market = Market::builder()
        .spot(spot)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.25)
        .build()
        .expect("valid market");
    let expiry = 1.0;

    let analytic = BarrierAnalyticEngine::new();
    let bs = BlackScholesEngine::new();
    let mc = MonteCarloPricingEngine::new(30_000, 400, 4_242);

    for (in_kind, out_kind, option_type, level) in [
        (
            BarrierKind::UpIn,
            BarrierKind::UpOut,
            OptionType::Call,
            130.0,
        ),
        (
            BarrierKind::UpIn,
            BarrierKind::UpOut,
            OptionType::Put,
            130.0,
        ),
        (
            BarrierKind::DownIn,
            BarrierKind::DownOut,
            OptionType::Call,
            70.0,
        ),
        (
            BarrierKind::DownIn,
            BarrierKind::DownOut,
            OptionType::Put,
            70.0,
        ),
    ] {
        let label = format!(
            "{option_type:?} barrier={level} S={spot} K={STRIKE} vol=0.25 r=0.03 q=0.01 T={expiry}"
        );
        let knock_in = barrier_option(in_kind, level, option_type, expiry);
        let knock_out = barrier_option(out_kind, level, option_type, expiry);
        let vanilla = match option_type {
            OptionType::Call => VanillaOption::european_call(STRIKE, expiry),
            OptionType::Put => VanillaOption::european_put(STRIKE, expiry),
        };

        // Analytic parity: exact identity of the Reiner-Rubinstein formulas.
        let a_in = analytic
            .price(&knock_in, &market)
            .expect("analytic in")
            .price;
        let a_out = analytic
            .price(&knock_out, &market)
            .expect("analytic out")
            .price;
        let a_vanilla = bs.price(&vanilla, &market).expect("analytic vanilla").price;
        assert_close(
            "analytic in+out parity",
            &label,
            a_in + a_out,
            a_vanilla,
            1e-9,
            1e-9,
        );

        // MC parity with common paths (same seed, steps, maturity, model).
        let m_in = mc.price(&knock_in, &market).expect("mc in").price;
        let m_out = mc.price(&knock_out, &market).expect("mc out").price;
        let m_vanilla = mc.price(&vanilla, &market).expect("mc vanilla").price;
        assert_close(
            "mc in+out parity (common seed)",
            &label,
            m_in + m_out,
            m_vanilla,
            1e-9,
            1e-9,
        );
    }
}

#[test]
fn barrier_mc_matches_analytic_with_bgk_barrier_shift() {
    // The MC engine monitors the barrier discretely on the path grid while
    // the Reiner-Rubinstein engine assumes continuous monitoring. Comparing
    // them directly is biased, so we use the Broadie-Glasserman-Kou
    // continuity correction: a discretely monitored barrier at H prices like
    // a continuous barrier at H * exp(+/- beta * sigma * sqrt(dt)) with
    // beta = 0.5826 (+ for up barriers, - for down barriers).
    //
    // Tolerances: 4 * stderr for MC noise plus 0.5% relative for the O(1/n)
    // residual of the BGK approximation at 400 monitoring steps.
    //
    // We also assert the one-sided ordering: discrete monitoring makes a
    // knock-out strictly MORE valuable (fewer knock-outs) and a knock-in
    // LESS valuable than continuous monitoring.
    const BGK_BETA: f64 = 0.5826;
    let spot = 100.0;
    let vol = 0.25;
    let expiry = 1.0;
    let steps = 400usize;
    let market = Market::builder()
        .spot(spot)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(vol)
        .build()
        .expect("valid market");

    let analytic = BarrierAnalyticEngine::new();
    let mc = MonteCarloPricingEngine::new(30_000, steps, 9_001);
    let dt = expiry / steps as f64;
    let shift = (BGK_BETA * vol * dt.sqrt()).exp();

    for (kind, option_type, level, is_up, is_out) in [
        (BarrierKind::UpOut, OptionType::Call, 130.0, true, true),
        (BarrierKind::UpIn, OptionType::Call, 130.0, true, false),
        (BarrierKind::DownOut, OptionType::Put, 70.0, false, true),
        (BarrierKind::DownIn, OptionType::Put, 70.0, false, false),
    ] {
        let label = format!(
            "{kind:?} {option_type:?} barrier={level} S={spot} K={STRIKE} vol={vol} r=0.03 q=0.01 T={expiry} steps={steps}"
        );
        let option = barrier_option(kind, level, option_type, expiry);

        let continuous = analytic
            .price(&option, &market)
            .expect("analytic barrier")
            .price;

        let mc_result = mc.price(&option, &market).expect("mc barrier");
        let mc_stderr = mc_result.stderr.expect("mc reports stderr");

        // BGK-adjusted continuous price approximating discrete monitoring:
        // move the barrier AWAY from the spot.
        let adjusted_level = if is_up { level * shift } else { level / shift };
        let adjusted = barrier_option(kind, adjusted_level, option_type, expiry);
        let adjusted_price = analytic
            .price(&adjusted, &market)
            .expect("analytic adjusted")
            .price;

        let err = (mc_result.price - adjusted_price).abs();
        let tol = 4.0 * mc_stderr + 5e-3 * adjusted_price.abs();
        assert!(
            err <= tol,
            "MC vs BGK-adjusted analytic disagrees at [{label}]: mc={} adjusted_analytic={adjusted_price} continuous_analytic={continuous} stderr={mc_stderr} err={err:.3e} tol={tol:.3e}",
            mc_result.price
        );

        // One-sided discrete-vs-continuous ordering (within MC noise):
        // discrete KO >= continuous KO, discrete KI <= continuous KI.
        let slack = 4.0 * mc_stderr;
        if is_out {
            assert!(
                mc_result.price >= continuous - slack,
                "discrete knock-out should not be cheaper than continuous at [{label}]: mc={} continuous={continuous} slack={slack:.3e}",
                mc_result.price
            );
        } else {
            assert!(
                mc_result.price <= continuous + slack,
                "discrete knock-in should not be dearer than continuous at [{label}]: mc={} continuous={continuous} slack={slack:.3e}",
                mc_result.price
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Asian options: geometric discrete closed form vs MC geometric payoff.
// ---------------------------------------------------------------------------

#[test]
fn geometric_asian_closed_form_matches_monte_carlo() {
    // 12 monthly observations on a 12-step path: every observation time maps
    // exactly onto a path grid point, so the MC payoff prices precisely the
    // contract the discrete closed form values; remaining error is sampling.
    let expiry = 1.0;
    let observations: Vec<f64> = (1..=12).map(|i| expiry * i as f64 / 12.0).collect();
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.25)
        .build()
        .expect("valid market");

    let analytic = GeometricAsianEngine::new();
    let mc = MonteCarloPricingEngine::new(100_000, 12, 5_555);

    for option_type in [OptionType::Call, OptionType::Put] {
        for strike in [90.0, 100.0, 110.0] {
            let option = AsianOption::new(
                option_type,
                strike,
                expiry,
                AsianSpec {
                    averaging: Averaging::Geometric,
                    strike_type: StrikeType::Fixed,
                    observation_times: observations.clone(),
                },
            );
            let label = format!(
                "geometric Asian {option_type:?} S=100 K={strike} vol=0.25 r=0.03 q=0.01 T={expiry} obs=12"
            );

            let reference = analytic.price(&option, &market).expect("closed form").price;
            let mc_result = mc.price(&option, &market).expect("mc asian");
            let stderr = mc_result.stderr.expect("mc reports stderr");
            assert_mc_close(
                "Asian MC(100k,12)",
                &label,
                mc_result.price,
                stderr,
                reference,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Digital cash-or-nothing: analytic vs Monte Carlo.
// ---------------------------------------------------------------------------

/// Exact-terminal-distribution GBM Monte Carlo for a cash-or-nothing digital.
/// The library's generic MC engine has no digital instrument, so this uses
/// the crate's own deterministic RNG and inverse-CDF sampler directly:
/// S_T = S0 * exp((r - q - sigma^2/2) T + sigma sqrt(T) Z). Returns
/// (discounted price, standard error).
fn mc_cash_or_nothing(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    cash: f64,
    rate: f64,
    div: f64,
    vol: f64,
    expiry: f64,
    paths: usize,
    seed: u64,
) -> (f64, f64) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let drift = (rate - div - 0.5 * vol * vol) * expiry;
    let diffusion = vol * expiry.sqrt();
    let discount = (-rate * expiry).exp();

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for _ in 0..paths {
        let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
        let terminal = spot * (drift + diffusion * z).exp();
        let pays = match option_type {
            OptionType::Call => terminal > strike,
            OptionType::Put => terminal < strike,
        };
        let payoff = if pays { cash } else { 0.0 };
        sum += payoff;
        sum_sq += payoff * payoff;
    }
    let n = paths as f64;
    let mean = sum / n;
    let var = (sum_sq - sum * sum / n).max(0.0) / (n - 1.0);
    (discount * mean, discount * (var / n).sqrt())
}

#[test]
fn digital_cash_or_nothing_analytic_matches_monte_carlo() {
    let spot = 100.0;
    let rate = 0.03;
    let div = 0.01;
    let vol = 0.25;
    let expiry = 1.0;
    let cash = 10.0;
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(div)
        .flat_vol(vol)
        .build()
        .expect("valid market");
    let analytic = DigitalAnalyticEngine::new();

    for option_type in [OptionType::Call, OptionType::Put] {
        for strike in [90.0, 100.0, 110.0] {
            let option = CashOrNothingOption::new(option_type, strike, cash, expiry);
            let label = format!(
                "cash-or-nothing {option_type:?} S={spot} K={strike} cash={cash} vol={vol} r={rate} q={div} T={expiry}"
            );

            let reference = analytic
                .price(&option, &market)
                .expect("digital analytic")
                .price;
            let (mc_price, mc_stderr) = mc_cash_or_nothing(
                option_type,
                spot,
                strike,
                cash,
                rate,
                div,
                vol,
                expiry,
                200_000,
                31_337,
            );
            assert_mc_close("digital MC(200k)", &label, mc_price, mc_stderr, reference);
        }
    }
}
