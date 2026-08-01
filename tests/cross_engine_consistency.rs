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
//! 3. Barriers: Reiner-Rubinstein continuous-monitoring analytic parity,
//!    discretely-monitored MC vs independent Brownian-bridge Sobol references,
//!    one-sided discrete-vs-continuous ordering, and in + out = vanilla parity.
//! 4. Asians: geometric discrete closed form vs MC geometric payoff.
//! 5. Digitals: cash-or-nothing analytic vs an exact-terminal-GBM MC.
//!
//! Tolerances:
//! - Deterministic engines (tree/PDE/FFT): documented per test, generally
//!   relative 2e-3 plus a small absolute floor for deep out-of-the-money
//!   points whose prices are ~1e-12 (relative error is meaningless there).
//! - Monte Carlo: |mc - reference| <= 4 combined standard errors.  For the
//!   exact-terminal vanilla grid the lognormal payoff variance is analytic,
//!   so even a zero-hit deep-OTM sample has a non-zero, contract-derived error
//!   budget rather than an arbitrary absolute price floor.
//!
//! All Monte Carlo engines use fixed seeds; the suite is fully deterministic
//! and uses no external data.

use openferric::core::types::AsianSpec;
use openferric::core::{Averaging, Instrument, OptionType, PricingEngine, StrikeType};
use openferric::engines::analytic::{
    BarrierAnalyticEngine, BlackScholesEngine, DigitalAnalyticEngine, GeometricAsianEngine,
};
use openferric::engines::fft::{BlackScholesCharFn, CarrMadanParams, carr_madan_price_at_strikes};
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::engines::monte_carlo::{MonteCarloInstrument, MonteCarloPricingEngine};
use openferric::engines::numerical::AmericanBinomialEngine;
use openferric::engines::pde::CrankNicolsonEngine;
use openferric::engines::tree::{BinomialTreeEngine, TrinomialTreeEngine};
use openferric::instruments::{AsianOption, BarrierOption, CashOrNothingOption, VanillaOption};
use openferric::market::Market;
use openferric::math::fast_norm::beasley_springer_moro_inv_cdf;
use openferric::math::fast_rng::{Xoshiro256PlusPlus, uniform_open01};
use openferric::math::normal_cdf;

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

/// Asserts an MC estimate agrees with an independent reference within four
/// combined standard errors.  `reference_stderr` is zero for analytic refs.
fn assert_mc_close(
    engine: &str,
    label: &str,
    value: f64,
    stderr: f64,
    reference: f64,
    reference_stderr: f64,
) {
    let err = (value - reference).abs();
    let combined_stderr = stderr.hypot(reference_stderr);
    let roundoff = 32.0 * f64::EPSILON * reference.abs().max(1.0);
    let tol = 4.0 * combined_stderr + roundoff;
    assert!(
        value.is_finite() && err <= tol,
        "{engine} disagrees at [{label}]: value={value} reference={reference} stderr={stderr} reference_stderr={reference_stderr} err={err:.3e} tol={tol:.3e}"
    );
}

/// Exact standard error of an undiscounted-iid exact-terminal GBM payoff
/// estimator, derived from truncated lognormal moments through order two.
fn exact_terminal_vanilla_stderr(point: &GridPoint, reference: f64, paths: usize) -> f64 {
    let sqrt_variance = point.vol * point.expiry.sqrt();
    let d2 = ((point.spot / STRIKE).ln()
        + (point.rate - point.div - 0.5 * point.vol * point.vol) * point.expiry)
        / sqrt_variance;
    let upper_moment = |power: i32| {
        let p = power as f64;
        point.spot.powi(power)
            * (p * (point.rate - point.div - 0.5 * point.vol * point.vol) * point.expiry
                + 0.5 * p * p * point.vol * point.vol * point.expiry)
                .exp()
            * normal_cdf(d2 + p * sqrt_variance)
    };
    let lower_moment = |power: i32| {
        let p = power as f64;
        point.spot.powi(power)
            * (p * (point.rate - point.div - 0.5 * point.vol * point.vol) * point.expiry
                + 0.5 * p * p * point.vol * point.vol * point.expiry)
                .exp()
            * normal_cdf(-(d2 + p * sqrt_variance))
    };
    let raw_second = match point.option_type {
        OptionType::Call => {
            upper_moment(2) - 2.0 * STRIKE * upper_moment(1) + STRIKE * STRIKE * upper_moment(0)
        }
        OptionType::Put => {
            STRIKE * STRIKE * lower_moment(0) - 2.0 * STRIKE * lower_moment(1) + lower_moment(2)
        }
    };
    let discount = (-point.rate * point.expiry).exp();
    let variance = (discount * discount * raw_second - reference * reference).max(0.0);
    (variance / paths as f64).sqrt()
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
        let reported_stderr = result.stderr.expect("monte carlo reports stderr");
        assert!(reported_stderr.is_finite() && reported_stderr >= 0.0);
        let exact_sampling_stderr = exact_terminal_vanilla_stderr(point, reference, 60_000);
        assert_mc_close(
            "MC(60k,1)",
            &point.label(),
            result.price,
            exact_sampling_stderr,
            reference,
            0.0,
        );
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

        // The CRR target values the same exercise schedule to a much smaller
        // discretisation error than the LSM sampling error.  Do not conceal a
        // regression miss with a percentage-of-price cushion.
        let lsm_result = lsm
            .price(&option, &market)
            .expect("lsm american pricing succeeds");
        let lsm_stderr = lsm_result.stderr.expect("lsm reports stderr");
        let err = (lsm_result.price - reference).abs();
        let tol = 4.0 * lsm_stderr;
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
            let tol = 4.0 * lsm_stderr;
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

/// Test-only vanilla adapter that deliberately exercises the generic
/// path-by-path engine instead of the exact-terminal vanilla fast path.
///
/// Barrier in/out parity needs all three prices to consume the same
/// stepwise random stream. The optimized `VanillaOption` dispatch samples
/// the mathematically equivalent terminal distribution directly, so it
/// intentionally has a different seeded stream.
#[derive(Debug)]
struct PathwiseVanilla(VanillaOption);

impl Instrument for PathwiseVanilla {
    fn instrument_type(&self) -> &str {
        "PathwiseVanilla"
    }
}

impl MonteCarloInstrument for PathwiseVanilla {
    fn validate_for_mc(&self) -> Result<(), openferric::core::PricingError> {
        self.0.validate()
    }

    fn maturity(&self) -> f64 {
        self.0.expiry
    }

    fn reference_strike(&self, _spot: f64) -> f64 {
        self.0.strike
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        let terminal = path[path.len() - 1];
        match self.0.option_type {
            OptionType::Call => (terminal - self.0.strike).max(0.0),
            OptionType::Put => (self.0.strike - terminal).max(0.0),
        }
    }
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
        let pathwise_vanilla = PathwiseVanilla(vanilla);
        let m_vanilla = mc
            .price(&pathwise_vanilla, &market)
            .expect("pathwise mc vanilla")
            .price;
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
fn barrier_mc_matches_independent_discrete_sobol_reference() {
    // Contract-matched references were generated with SciPy using 64
    // independently Owen-scrambled, Brownian-bridge Sobol replicates of 2^15
    // paths.  Monitoring occurs at the same 400 dates as the library MC.  The
    // knock-in refs use exact vanilla parity as a control variate.  Each tuple
    // carries the replicate standard error, which is combined with MC stderr.
    //
    // We retain the supplemental one-sided ordering: discrete monitoring makes a
    // knock-out strictly MORE valuable (fewer knock-outs) and a knock-in
    // LESS valuable than continuous monitoring.
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

    for (kind, option_type, level, is_out, reference, reference_stderr) in [
        (
            BarrierKind::UpOut,
            OptionType::Call,
            130.0,
            true,
            2.294_523_586_005_971,
            1.451_140_866_445_940_2e-3,
        ),
        (
            BarrierKind::UpIn,
            OptionType::Call,
            130.0,
            false,
            8.467_871_040_331_177,
            1.451_140_866_445_940_2e-3,
        ),
        (
            BarrierKind::DownOut,
            OptionType::Put,
            70.0,
            true,
            4.266_014_703_753_552,
            1.642_005_939_624_996_5e-3,
        ),
        (
            BarrierKind::DownIn,
            OptionType::Put,
            70.0,
            false,
            4.535_949_902_517_597,
            1.642_005_939_624_996_5e-3,
        ),
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

        assert_mc_close(
            "barrier MC(30k,400)",
            &label,
            mc_result.price,
            mc_stderr,
            reference,
            reference_stderr,
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
                0.0,
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
            assert_mc_close(
                "digital MC(200k)",
                &label,
                mc_price,
                mc_stderr,
                reference,
                0.0,
            );
        }
    }
}
