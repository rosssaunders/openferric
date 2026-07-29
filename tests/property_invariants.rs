//! Property-based no-arbitrage and structural invariants.
//!
//! Deterministic parameter sweeps (seeded `Xoshiro256PlusPlus`, no proptest
//! dependency) asserting model-free identities and structural properties:
//!
//! 1. Put-call parity (Black-Scholes, Black-76, Garman-Kohlhagen, digitals).
//! 2. Monotonicity and bounds of Black-Scholes prices and Greeks.
//! 3. Convexity of the call price in strike (butterfly spreads).
//! 4. Barrier in/out parity and knock-out structural bounds.
//! 5. Vol surface no-arbitrage (calendar monotonicity, Fengler butterfly check).
//! 6. Yield curve bootstrap invariants and par-swap repricing.
//! 7. CDS survival-curve invariants and par-spread repricing.
//! 8. Analytic Greeks vs central finite differences of the analytic price.
//!
//! Every assertion message carries the full parameter set so a failure is
//! immediately reproducible.

use openferric::core::{
    BarrierDirection as CoreBarrierDirection, BarrierStyle, OptionType, PricingEngine,
    PricingResult,
};
use openferric::credit::{Cds, bootstrap_survival_curve_from_cds_spreads};
use openferric::engines::analytic::digital::DigitalAnalyticEngine;
use openferric::engines::analytic::{
    BarrierAnalyticEngine, Black76Engine, BlackScholesEngine, GarmanKohlhagenEngine,
};
use openferric::instruments::digital::{AssetOrNothingOption, CashOrNothingOption};
use openferric::instruments::{BarrierOption, FuturesOption, FxOption, VanillaOption};
use openferric::market::Market;
use openferric::math::fast_rng::Xoshiro256PlusPlus;
use openferric::pricing::barrier::barrier_price_closed_form;
use openferric::pricing::european::{black_76_price, black_scholes_price};
use openferric::rates::{YieldCurve, YieldCurveBuilder};
use openferric::vol::fengler::FenglerSurface;
use openferric::vol::surface::{SviParams, VolSurface};

// ============================================================================
// Helpers
// ============================================================================

/// Deterministic uniform sample in `[lo, hi)`.
fn uniform(rng: &mut Xoshiro256PlusPlus, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.next_f64()
}

/// Flat-vol equity market.
fn make_market(spot: f64, rate: f64, dividend_yield: f64, vol: f64) -> Market {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend_yield)
        .flat_vol(vol)
        .build()
        .expect("valid market")
}

/// Black-Scholes analytic result for a European option.
fn bs_result(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PricingResult {
    let option = match option_type {
        OptionType::Call => VanillaOption::european_call(strike, expiry),
        OptionType::Put => VanillaOption::european_put(strike, expiry),
    };
    let market = make_market(spot, rate, dividend_yield, vol);
    BlackScholesEngine::new()
        .price(&option, &market)
        .expect("BS pricing succeeds")
}

fn bs_price(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    bs_result(option_type, spot, strike, rate, dividend_yield, vol, expiry).price
}

/// Random Black-Scholes scenario over a wide but well-posed domain.
#[derive(Debug, Clone, Copy)]
struct BsScenario {
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
}

fn sample_bs(rng: &mut Xoshiro256PlusPlus) -> BsScenario {
    BsScenario {
        spot: uniform(rng, 20.0, 300.0),
        strike: uniform(rng, 20.0, 300.0),
        rate: uniform(rng, -0.02, 0.10),
        dividend_yield: uniform(rng, 0.0, 0.06),
        vol: uniform(rng, 0.05, 0.80),
        expiry: uniform(rng, 0.05, 3.0),
    }
}

/// Flat continuously-compounded discount curve with quarterly nodes.
fn flat_discount_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let n = (max_tenor * 4.0).ceil() as usize;
    YieldCurve::new(
        (1..=n)
            .map(|i| {
                let t = i as f64 * 0.25;
                (t, (-rate * t).exp())
            })
            .collect(),
    )
}

// ============================================================================
// 1. Put-call parity
// ============================================================================

#[test]
fn put_call_parity_black_scholes() {
    // C - P = S e^{-qT} - K e^{-rT}, model-free for European options.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA11C_E001);
    for i in 0..200 {
        let s = sample_bs(&mut rng);
        let call = bs_price(
            OptionType::Call,
            s.spot,
            s.strike,
            s.rate,
            s.dividend_yield,
            s.vol,
            s.expiry,
        );
        let put = bs_price(
            OptionType::Put,
            s.spot,
            s.strike,
            s.rate,
            s.dividend_yield,
            s.vol,
            s.expiry,
        );
        let forward_value =
            s.spot * (-s.dividend_yield * s.expiry).exp() - s.strike * (-s.rate * s.expiry).exp();
        let err = (call - put - forward_value).abs();
        assert!(
            err <= 1e-10,
            "BS put-call parity violated at point {i}: {s:?} call={call} put={put} \
             S e^-qT - K e^-rT = {forward_value} err={err}"
        );
    }
}

#[test]
fn put_call_parity_black76() {
    // Forward parity: C - P = e^{-rT} (F - K).
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA11C_E002);
    let engine = Black76Engine::new();
    let dummy = make_market(100.0, 0.0, 0.0, 0.2);
    for i in 0..200 {
        let forward = uniform(&mut rng, 20.0, 300.0);
        let strike = uniform(&mut rng, 20.0, 300.0);
        let rate = uniform(&mut rng, -0.02, 0.10);
        let vol = uniform(&mut rng, 0.05, 0.80);
        let expiry = uniform(&mut rng, 0.05, 3.0);

        let call = engine
            .price(
                &FuturesOption::call(forward, strike, vol, rate, expiry),
                &dummy,
            )
            .expect("Black76 call prices")
            .price;
        let put = engine
            .price(
                &FuturesOption::put(forward, strike, vol, rate, expiry),
                &dummy,
            )
            .expect("Black76 put prices")
            .price;
        let parity = (-rate * expiry).exp() * (forward - strike);
        let err = (call - put - parity).abs();
        assert!(
            err <= 1e-10,
            "Black76 parity violated at point {i}: F={forward} K={strike} r={rate} \
             vol={vol} T={expiry} call={call} put={put} e^-rT(F-K)={parity} err={err}"
        );
    }
}

#[test]
fn black_scholes_and_black76_are_homogeneous() {
    // Scaling every cash amount by λ must scale the option value by λ.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA11C_E005);
    for i in 0..200 {
        let spot = uniform(&mut rng, 0.1, 500.0);
        let strike = uniform(&mut rng, 0.1, 500.0);
        let scale = uniform(&mut rng, 0.01, 100.0);
        let rate = uniform(&mut rng, -0.10, 0.25);
        let vol = uniform(&mut rng, 0.01, 2.0);
        let expiry = uniform(&mut rng, 0.001, 10.0);

        for option_type in [OptionType::Call, OptionType::Put] {
            let base_bs = black_scholes_price(option_type, spot, strike, rate, vol, expiry);
            let scaled_bs =
                black_scholes_price(option_type, scale * spot, scale * strike, rate, vol, expiry);
            let bs_tolerance = 8e-10 * (1.0 + scaled_bs.abs());
            assert!(
                (scaled_bs - scale * base_bs).abs() <= bs_tolerance,
                "Black-Scholes homogeneity failed at point {i}: type={option_type:?} \
                 S={spot} K={strike} scale={scale} r={rate} vol={vol} T={expiry} \
                 base={base_bs} scaled={scaled_bs}"
            );

            let base_black76 = black_76_price(option_type, spot, strike, rate, vol, expiry);
            let scaled_black76 =
                black_76_price(option_type, scale * spot, scale * strike, rate, vol, expiry);
            let black76_tolerance = 8e-10 * (1.0 + scaled_black76.abs());
            assert!(
                (scaled_black76 - scale * base_black76).abs() <= black76_tolerance,
                "Black-76 homogeneity failed at point {i}: type={option_type:?} \
                 F={spot} K={strike} scale={scale} r={rate} vol={vol} T={expiry} \
                 base={base_black76} scaled={scaled_black76}"
            );
        }
    }
}

#[test]
fn expiry_and_zero_volatility_follow_intrinsic_value_contract() {
    let discount = (-0.05_f64).exp();
    for option_type in [OptionType::Call, OptionType::Put] {
        for (spot, strike) in [(80.0, 100.0), (100.0, 100.0), (120.0, 100.0)] {
            let intrinsic = match option_type {
                OptionType::Call => f64::max(spot - strike, 0.0),
                OptionType::Put => f64::max(strike - spot, 0.0),
            };
            assert_eq!(
                black_scholes_price(option_type, spot, strike, 0.05, 0.20, 0.0),
                intrinsic
            );
            let deterministic_spot_value = match option_type {
                OptionType::Call => f64::max(spot - strike * discount, 0.0),
                OptionType::Put => f64::max(strike * discount - spot, 0.0),
            };
            assert_eq!(
                black_scholes_price(option_type, spot, strike, 0.05, 0.0, 1.0),
                deterministic_spot_value
            );
            assert_eq!(
                black_76_price(option_type, spot, strike, 0.05, 0.20, 0.0),
                intrinsic
            );
            assert_eq!(
                black_76_price(option_type, spot, strike, 0.05, 0.0, 1.0),
                discount * intrinsic
            );
        }
    }
}

#[test]
fn already_breached_barrier_has_deterministic_style_semantics() {
    for (direction, barrier) in [
        (CoreBarrierDirection::Down, 105.0),
        (CoreBarrierDirection::Up, 95.0),
    ] {
        for option_type in [OptionType::Call, OptionType::Put] {
            let vanilla = black_scholes_price(option_type, 100.0, 100.0, 0.03, 0.20, 1.0);
            let knock_in = barrier_price_closed_form(
                option_type,
                BarrierStyle::In,
                direction,
                100.0,
                100.0,
                barrier,
                0.03,
                0.20,
                1.0,
            );
            let knock_out = barrier_price_closed_form(
                option_type,
                BarrierStyle::Out,
                direction,
                100.0,
                100.0,
                barrier,
                0.03,
                0.20,
                1.0,
            );
            assert_eq!(knock_in, vanilla);
            assert_eq!(knock_out, 0.0);
        }
    }
}

#[test]
fn put_call_parity_garman_kohlhagen() {
    // FX parity: C - P = S e^{-r_f T} - K e^{-r_d T}.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA11C_E003);
    let engine = GarmanKohlhagenEngine::new();
    let dummy = make_market(100.0, 0.0, 0.0, 0.2);
    for i in 0..200 {
        let spot = uniform(&mut rng, 0.5, 2.5);
        let strike = uniform(&mut rng, 0.5, 2.5);
        let rd = uniform(&mut rng, -0.01, 0.08);
        let rf = uniform(&mut rng, -0.01, 0.08);
        let vol = uniform(&mut rng, 0.05, 0.50);
        let expiry = uniform(&mut rng, 0.05, 3.0);

        let call = engine
            .price(
                &FxOption::new(OptionType::Call, rd, rf, spot, strike, vol, expiry),
                &dummy,
            )
            .expect("GK call prices")
            .price;
        let put = engine
            .price(
                &FxOption::new(OptionType::Put, rd, rf, spot, strike, vol, expiry),
                &dummy,
            )
            .expect("GK put prices")
            .price;
        let parity = spot * (-rf * expiry).exp() - strike * (-rd * expiry).exp();
        let err = (call - put - parity).abs();
        assert!(
            err <= 1e-10,
            "Garman-Kohlhagen parity violated at point {i}: S={spot} K={strike} rd={rd} \
             rf={rf} vol={vol} T={expiry} call={call} put={put} parity={parity} err={err}"
        );
    }
}

#[test]
fn put_call_parity_digitals() {
    // cash_call + cash_put = cash e^{-rT}; asset_call + asset_put = S e^{-qT}.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA11C_E004);
    let engine = DigitalAnalyticEngine::new();
    for i in 0..200 {
        let s = sample_bs(&mut rng);
        let cash = uniform(&mut rng, 0.5, 10.0);
        let market = make_market(s.spot, s.rate, s.dividend_yield, s.vol);

        let cash_call = engine
            .price(
                &CashOrNothingOption::new(OptionType::Call, s.strike, cash, s.expiry),
                &market,
            )
            .expect("cash-or-nothing call prices")
            .price;
        let cash_put = engine
            .price(
                &CashOrNothingOption::new(OptionType::Put, s.strike, cash, s.expiry),
                &market,
            )
            .expect("cash-or-nothing put prices")
            .price;
        let cash_parity = cash * (-s.rate * s.expiry).exp();
        let cash_err = (cash_call + cash_put - cash_parity).abs();
        assert!(
            cash_err <= 1e-10,
            "cash-or-nothing parity violated at point {i}: {s:?} cash={cash} \
             call={cash_call} put={cash_put} cash e^-rT={cash_parity} err={cash_err}"
        );

        let asset_call = engine
            .price(
                &AssetOrNothingOption::new(OptionType::Call, s.strike, s.expiry),
                &market,
            )
            .expect("asset-or-nothing call prices")
            .price;
        let asset_put = engine
            .price(
                &AssetOrNothingOption::new(OptionType::Put, s.strike, s.expiry),
                &market,
            )
            .expect("asset-or-nothing put prices")
            .price;
        let asset_parity = s.spot * (-s.dividend_yield * s.expiry).exp();
        let asset_err = (asset_call + asset_put - asset_parity).abs();
        assert!(
            asset_err <= 1e-10,
            "asset-or-nothing parity violated at point {i}: {s:?} call={asset_call} \
             put={asset_put} S e^-qT={asset_parity} err={asset_err}"
        );
    }
}

// ============================================================================
// 2. Monotonicity and bounds (Black-Scholes analytic)
// ============================================================================

#[test]
fn call_price_non_increasing_in_strike() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0B0B_0001);
    for i in 0..200 {
        let s = sample_bs(&mut rng);
        let mut prev = f64::INFINITY;
        for j in 0..21 {
            let strike = s.spot * (0.4 + 0.08 * j as f64);
            let price = bs_price(
                OptionType::Call,
                s.spot,
                strike,
                s.rate,
                s.dividend_yield,
                s.vol,
                s.expiry,
            );
            assert!(
                price <= prev + 1e-12 * (1.0 + prev.min(1e300)),
                "call not non-increasing in K at scenario {i} step {j}: {s:?} \
                 K={strike} price={price} prev={prev}"
            );
            prev = price;
        }
    }
}

#[test]
fn call_price_non_decreasing_in_spot_and_vol() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0B0B_0002);
    for i in 0..200 {
        let s = sample_bs(&mut rng);

        let mut prev = f64::NEG_INFINITY;
        for j in 0..21 {
            let spot = s.strike * (0.4 + 0.08 * j as f64);
            let price = bs_price(
                OptionType::Call,
                spot,
                s.strike,
                s.rate,
                s.dividend_yield,
                s.vol,
                s.expiry,
            );
            assert!(
                price + 1e-12 * (1.0 + price) >= prev,
                "call not non-decreasing in S at scenario {i} step {j}: {s:?} \
                 S={spot} price={price} prev={prev}"
            );
            prev = price;
        }

        let mut prev = f64::NEG_INFINITY;
        for j in 0..21 {
            let vol = 0.05 + 0.05 * j as f64;
            let price = bs_price(
                OptionType::Call,
                s.spot,
                s.strike,
                s.rate,
                s.dividend_yield,
                vol,
                s.expiry,
            );
            assert!(
                price + 1e-12 * (1.0 + price) >= prev,
                "call not non-decreasing in vol at scenario {i} step {j}: {s:?} \
                 vol={vol} price={price} prev={prev}"
            );
            prev = price;
        }
    }
}

#[test]
fn call_price_non_decreasing_in_expiry_without_dividends() {
    // NOTE: monotonicity of the European call price in T only holds model-free
    // for q = 0 and r >= 0 (the call is then worth its American counterpart and
    // longer-dated dominates). With q > 0 (or r < 0) the European call can
    // legitimately *decrease* in T, so the sweep is restricted to q = 0, r >= 0.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0B0B_0003);
    for i in 0..200 {
        let spot = uniform(&mut rng, 20.0, 300.0);
        let strike = uniform(&mut rng, 20.0, 300.0);
        let rate = uniform(&mut rng, 0.0, 0.10);
        let vol = uniform(&mut rng, 0.05, 0.80);
        let mut prev = f64::NEG_INFINITY;
        for j in 0..21 {
            let expiry = 0.05 + 0.15 * j as f64;
            let price = bs_price(OptionType::Call, spot, strike, rate, 0.0, vol, expiry);
            assert!(
                price + 1e-12 * (1.0 + price) >= prev,
                "call not non-decreasing in T (q=0) at scenario {i} step {j}: \
                 S={spot} K={strike} r={rate} vol={vol} T={expiry} price={price} prev={prev}"
            );
            prev = price;
        }
    }
}

#[test]
fn call_price_within_no_arbitrage_bounds() {
    // max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0B0B_0004);
    for i in 0..200 {
        let s = sample_bs(&mut rng);
        let price = bs_price(
            OptionType::Call,
            s.spot,
            s.strike,
            s.rate,
            s.dividend_yield,
            s.vol,
            s.expiry,
        );
        let lower = (s.spot * (-s.dividend_yield * s.expiry).exp()
            - s.strike * (-s.rate * s.expiry).exp())
        .max(0.0);
        let upper = s.spot * (-s.dividend_yield * s.expiry).exp();
        assert!(
            price >= lower - 1e-10 && price <= upper + 1e-10,
            "call outside no-arbitrage bounds at point {i}: {s:?} price={price} \
             lower={lower} upper={upper}"
        );
    }
}

#[test]
fn greeks_within_structural_bounds() {
    // Call delta in [0, e^{-qT}], put delta in [-e^{-qT}, 0], gamma >= 0, vega >= 0.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0B0B_0005);
    for i in 0..200 {
        let s = sample_bs(&mut rng);
        let df_q = (-s.dividend_yield * s.expiry).exp();

        for option_type in [OptionType::Call, OptionType::Put] {
            let result = bs_result(
                option_type,
                s.spot,
                s.strike,
                s.rate,
                s.dividend_yield,
                s.vol,
                s.expiry,
            );
            let greeks = result.greeks.expect("BS engine returns greeks");
            match option_type {
                OptionType::Call => assert!(
                    (-1e-12..=df_q + 1e-12).contains(&greeks.delta),
                    "call delta out of [0, e^-qT] at point {i}: {s:?} \
                     delta={} e^-qT={df_q}",
                    greeks.delta
                ),
                OptionType::Put => assert!(
                    (-df_q - 1e-12..=1e-12).contains(&greeks.delta),
                    "put delta out of [-e^-qT, 0] at point {i}: {s:?} \
                     delta={} e^-qT={df_q}",
                    greeks.delta
                ),
            }
            assert!(
                greeks.gamma >= -1e-12,
                "negative gamma at point {i} ({option_type:?}): {s:?} gamma={}",
                greeks.gamma
            );
            assert!(
                greeks.vega >= -1e-12,
                "negative vega at point {i} ({option_type:?}): {s:?} vega={}",
                greeks.vega
            );
        }
    }
}

// ============================================================================
// 3. Convexity in strike (butterfly)
// ============================================================================

#[test]
fn call_price_convex_in_strike() {
    // Butterfly: C(K-h) - 2 C(K) + C(K+h) >= 0 for all K, h > 0.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x00C0_FFEE);
    for i in 0..200 {
        let s = sample_bs(&mut rng);
        let h = s.spot * uniform(&mut rng, 0.01, 0.10);
        for j in 0..15 {
            let strike = s.spot * (0.5 + 0.07 * j as f64);
            let price = |k: f64| {
                bs_price(
                    OptionType::Call,
                    s.spot,
                    k,
                    s.rate,
                    s.dividend_yield,
                    s.vol,
                    s.expiry,
                )
            };
            let butterfly = price(strike - h) - 2.0 * price(strike) + price(strike + h);
            assert!(
                butterfly >= -1e-12,
                "butterfly arbitrage at scenario {i} step {j}: {s:?} K={strike} h={h} \
                 butterfly={butterfly}"
            );
        }
    }
}

// ============================================================================
// 4. Barrier parity and bounds
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BarrierDirection {
    Down,
    Up,
}

fn barrier_pair(
    direction: BarrierDirection,
    option_type: OptionType,
    strike: f64,
    expiry: f64,
    barrier: f64,
) -> (BarrierOption, BarrierOption) {
    let base = || {
        let b = BarrierOption::builder()
            .strike(strike)
            .expiry(expiry)
            .rebate(0.0);
        match option_type {
            OptionType::Call => b.call(),
            OptionType::Put => b.put(),
        }
    };
    let (knock_in, knock_out) = match direction {
        BarrierDirection::Down => (
            base().down_and_in(barrier).build(),
            base().down_and_out(barrier).build(),
        ),
        BarrierDirection::Up => (
            base().up_and_in(barrier).build(),
            base().up_and_out(barrier).build(),
        ),
    };
    (
        knock_in.expect("valid knock-in"),
        knock_out.expect("valid knock-out"),
    )
}

#[test]
fn barrier_in_out_parity_all_eight_types() {
    // With zero rebate: knock-in + knock-out = vanilla, for all four barrier
    // directions x call/put (8 combinations), and knock-out <= vanilla.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBA22_0001);
    let barrier_engine = BarrierAnalyticEngine::new();
    for direction in [BarrierDirection::Down, BarrierDirection::Up] {
        for option_type in [OptionType::Call, OptionType::Put] {
            for i in 0..50 {
                let spot = uniform(&mut rng, 50.0, 200.0);
                let strike = spot * uniform(&mut rng, 0.6, 1.4);
                let rate = uniform(&mut rng, 0.0, 0.08);
                let q = uniform(&mut rng, 0.0, 0.04);
                let vol = uniform(&mut rng, 0.10, 0.60);
                let expiry = uniform(&mut rng, 0.1, 2.0);
                let barrier = match direction {
                    BarrierDirection::Down => spot * uniform(&mut rng, 0.50, 0.95),
                    BarrierDirection::Up => spot * uniform(&mut rng, 1.05, 2.00),
                };

                let market = make_market(spot, rate, q, vol);
                let (knock_in, knock_out) =
                    barrier_pair(direction, option_type, strike, expiry, barrier);

                let in_price = barrier_engine
                    .price(&knock_in, &market)
                    .expect("knock-in prices")
                    .price;
                let out_price = barrier_engine
                    .price(&knock_out, &market)
                    .expect("knock-out prices")
                    .price;
                let vanilla = bs_price(option_type, spot, strike, rate, q, vol, expiry);

                let err = (in_price + out_price - vanilla).abs();
                assert!(
                    err <= 1e-8,
                    "barrier in/out parity violated ({direction:?} {option_type:?}, point {i}): \
                     S={spot} K={strike} B={barrier} r={rate} q={q} vol={vol} T={expiry} \
                     in={in_price} out={out_price} vanilla={vanilla} err={err}"
                );
                assert!(
                    out_price <= vanilla + 1e-10,
                    "knock-out above vanilla ({direction:?} {option_type:?}, point {i}): \
                     S={spot} K={strike} B={barrier} r={rate} q={q} vol={vol} T={expiry} \
                     out={out_price} vanilla={vanilla}"
                );
                assert!(
                    in_price >= -1e-10 && out_price >= -1e-10,
                    "negative barrier price ({direction:?} {option_type:?}, point {i}): \
                     S={spot} K={strike} B={barrier} r={rate} q={q} vol={vol} T={expiry} \
                     in={in_price} out={out_price}"
                );
            }
        }
    }
}

#[test]
fn knock_out_non_increasing_as_barrier_tightens_toward_spot() {
    // A tighter barrier (closer to spot) removes payoff paths, so the
    // knock-out value must not increase.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBA22_0002);
    let barrier_engine = BarrierAnalyticEngine::new();
    for direction in [BarrierDirection::Down, BarrierDirection::Up] {
        for option_type in [OptionType::Call, OptionType::Put] {
            for i in 0..25 {
                let spot = uniform(&mut rng, 50.0, 200.0);
                let strike = spot * uniform(&mut rng, 0.7, 1.3);
                let rate = uniform(&mut rng, 0.0, 0.08);
                let q = uniform(&mut rng, 0.0, 0.04);
                let vol = uniform(&mut rng, 0.10, 0.60);
                let expiry = uniform(&mut rng, 0.1, 2.0);

                let mut prev = f64::INFINITY;
                for j in 0..16 {
                    let frac = j as f64 / 15.0;
                    // Barrier marches toward spot as j grows.
                    let barrier = match direction {
                        BarrierDirection::Down => spot * (0.50 + (0.97 - 0.50) * frac),
                        BarrierDirection::Up => spot * (1.95 - (1.95 - 1.03) * frac),
                    };
                    let market = make_market(spot, rate, q, vol);
                    let (_, knock_out) =
                        barrier_pair(direction, option_type, strike, expiry, barrier);
                    let out_price = barrier_engine
                        .price(&knock_out, &market)
                        .expect("knock-out prices")
                        .price;
                    assert!(
                        out_price <= prev + 1e-10,
                        "knock-out increased as barrier tightened \
                         ({direction:?} {option_type:?}, scenario {i} step {j}): \
                         S={spot} K={strike} B={barrier} r={rate} q={q} vol={vol} \
                         T={expiry} price={out_price} prev={prev}"
                    );
                    prev = out_price;
                }
            }
        }
    }
}

// ============================================================================
// 5. Vol surface no-arbitrage
// ============================================================================

#[test]
fn svi_surface_total_variance_non_decreasing_in_expiry() {
    // Slices share (b, rho, m, sigma) and have strictly increasing `a`, so the
    // surface is calendar-arbitrage-free by construction; the query layer
    // (interpolation in T at fixed log-moneyness) must preserve that.
    let forward = 100.0;
    let common = |t: f64| SviParams {
        a: 0.005 + 0.02 * t,
        b: 0.08,
        rho: -0.3,
        m: 0.0,
        sigma: 0.25,
    };
    let expiries = [0.25, 0.5, 1.0, 2.0, 3.0];
    let surface = VolSurface::new(expiries.iter().map(|&t| (t, common(t))).collect(), forward)
        .expect("valid SVI surface");

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5141_0001);
    for i in 0..50 {
        let k = uniform(&mut rng, -1.2, 1.2);
        let strike = forward * k.exp();
        let a = uniform(&mut rng, 0.25, 3.0);
        let b = uniform(&mut rng, 0.25, 3.0);
        let (t1, t2) = if a <= b { (a, b) } else { (b, a) };
        let w1 = surface.total_variance(strike, t1);
        let w2 = surface.total_variance(strike, t2);
        assert!(
            w2 >= w1 - 1e-12,
            "calendar arbitrage in SVI surface at pair {i}: k={k} K={strike} \
             t1={t1} t2={t2} w1={w1} w2={w2}"
        );
    }
}

#[test]
fn fengler_surface_from_clean_quotes_has_no_arbitrage() {
    // Clean synthetic quotes: strike-independent vol with an increasing total
    // variance term structure. The fitted surface must report no butterfly
    // (g(k) >= 0) or calendar violations, and stay calendar-monotone when
    // queried off-grid.
    let expiries: [f64; 4] = [0.25, 0.5, 1.0, 2.0];
    let spot_rate = 0.02_f64;
    let forward_curve: Vec<(f64, f64)> = expiries
        .iter()
        .map(|&t| (t, 100.0 * (spot_rate * t).exp()))
        .collect();
    let mut quotes = Vec::new();
    for &t in &expiries {
        let vol = 0.18 + 0.03 * t.sqrt();
        for j in 0..11 {
            let strike = 60.0 + 10.0 * j as f64;
            quotes.push((strike, t, vol));
        }
    }
    let surface = FenglerSurface::new(&quotes, &forward_curve);

    let violations = surface.check_arbitrage();
    assert!(
        violations.is_empty(),
        "Fengler surface built from clean quotes reports arbitrage: {violations:?}"
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5141_0002);
    for i in 0..50 {
        let k = uniform(&mut rng, -0.4, 0.4);
        let a = uniform(&mut rng, 0.25, 2.0);
        let b = uniform(&mut rng, 0.25, 2.0);
        let (t1, t2) = if a <= b { (a, b) } else { (b, a) };
        let w1 = surface.total_variance(k, t1);
        let w2 = surface.total_variance(k, t2);
        assert!(
            w2 >= w1 - 1e-12,
            "calendar arbitrage in Fengler surface at pair {i}: k={k} t1={t1} t2={t2} \
             w1={w1} w2={w2}"
        );
    }
}

// ============================================================================
// 6. Yield curve invariants
// ============================================================================

#[test]
fn deposit_curve_discount_factor_invariants() {
    let deposits: Vec<(f64, f64)> = [
        (0.25, 0.020),
        (0.5, 0.022),
        (1.0, 0.025),
        (2.0, 0.028),
        (3.0, 0.030),
        (5.0, 0.032),
        (7.0, 0.033),
        (10.0, 0.034),
    ]
    .to_vec();
    let curve = YieldCurveBuilder::from_deposits(&deposits);

    let df0 = curve.discount_factor(0.0);
    assert!((df0 - 1.0).abs() <= 1e-14, "DF(0) must equal 1, got {df0}");

    let mut prev = 1.0_f64;
    for j in 1..=100 {
        let t = 0.1 * j as f64;
        let df = curve.discount_factor(t);
        assert!(
            df.is_finite() && df > 0.0 && df < 1.0,
            "DF({t}) out of (0, 1): {df}"
        );
        assert!(
            df < prev,
            "discount factors not strictly decreasing for positive rates: \
             DF({t})={df} >= DF({})={prev}",
            t - 0.1
        );
        prev = df;
    }

    for j in 0..40 {
        let t1 = 0.25 * j as f64;
        let t2 = t1 + 0.25;
        let fwd = curve.forward_rate(t1, t2);
        assert!(
            fwd.is_finite(),
            "forward rate over [{t1}, {t2}] is not finite: {fwd}"
        );
    }
}

#[test]
fn swap_bootstrap_reprices_input_par_rates() {
    // Bootstrap contract (see `YieldCurveBuilder::from_swap_rates_with_settings`):
    // with the default interpolating method each input swap reprices exactly,
    // i.e. coupon * sum DF(t_i) + DF(T) = 1, equivalently
    // par = freq * (1 - DF(T)) / sum DF(t_i) equals the input quote.
    let frequency = 2usize;
    let quotes: Vec<(f64, f64)> = vec![
        (1.0, 0.020),
        (2.0, 0.022),
        (3.0, 0.024),
        (5.0, 0.026),
        (7.0, 0.027),
        (10.0, 0.028),
    ];
    let curve = YieldCurveBuilder::from_swap_rates(&quotes, frequency);

    let df0 = curve.discount_factor(0.0);
    assert!((df0 - 1.0).abs() <= 1e-14, "DF(0) must equal 1, got {df0}");

    let dt = 1.0 / frequency as f64;
    for &(tenor, quote) in &quotes {
        let periods = (tenor * frequency as f64).round() as usize;
        let annuity: f64 = (1..=periods)
            .map(|i| curve.discount_factor(i as f64 * dt))
            .sum();
        let par = (1.0 - curve.discount_factor(tenor)) / (dt * annuity);
        let err = (par - quote).abs();
        assert!(
            err <= 1e-8,
            "par swap repricing failed at tenor {tenor}: input={quote} implied par={par} \
             err={err}"
        );
    }

    let mut prev = 1.0_f64;
    for j in 1..=100 {
        let t = 0.1 * j as f64;
        let df = curve.discount_factor(t);
        assert!(
            df.is_finite() && df > 0.0 && df < prev,
            "swap curve DF not strictly decreasing/finite at t={t}: df={df} prev={prev}"
        );
        prev = df;
    }
}

// ============================================================================
// 7. CDS invariants
// ============================================================================

#[test]
fn cds_bootstrap_survival_and_par_spread_invariants() {
    let discount_curve = flat_discount_curve(0.03, 12.0);
    let recovery = 0.4;
    let payment_freq = 4usize;
    let quotes: Vec<(f64, f64)> = vec![
        (1.0, 0.0060),
        (3.0, 0.0080),
        (5.0, 0.0100),
        (7.0, 0.0115),
        (10.0, 0.0130),
    ];

    let survival =
        bootstrap_survival_curve_from_cds_spreads(&quotes, recovery, payment_freq, &discount_curve);

    // Survival probabilities in (0, 1], non-increasing.
    let mut prev = 1.0_f64;
    for j in 0..=120 {
        let t = 0.1 * j as f64;
        let p = survival.survival_prob(t);
        assert!(
            p > 0.0 && p <= 1.0,
            "survival probability out of (0, 1] at t={t}: {p}"
        );
        assert!(
            p <= prev + 1e-12,
            "survival probability increased at t={t}: S(t)={p} prev={prev}"
        );
        prev = p;
    }

    // Par-spread repricing of the bootstrap inputs.
    for &(tenor, spread) in &quotes {
        let cds = Cds {
            notional: 1.0,
            spread,
            maturity: tenor,
            recovery_rate: recovery,
            payment_freq,
        };
        let fair = cds.fair_spread(&discount_curve, &survival);
        let err = (fair - spread).abs();
        assert!(
            err <= 1e-8,
            "CDS par-spread repricing failed at tenor {tenor}: input={spread} fair={fair} \
             err={err}"
        );
        let npv = cds.npv(&discount_curve, &survival);
        assert!(
            npv.abs() <= 1e-8,
            "pillar CDS NPV not ~0 at tenor {tenor}: npv={npv}"
        );
    }
}

// ============================================================================
// 8. Greeks vs finite differences
// ============================================================================

#[test]
fn bs_greeks_match_central_finite_differences() {
    // Independent of external references: the analytic Greeks must agree with
    // central finite differences of the engine's own analytic price.
    // Conventions (from the engine): vega = dP/dvol, rho = dP/dr,
    // theta = -dP/dT (calendar decay).
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD1FF_0001);
    // Moderate ranges keep the Greeks well away from machine noise so a pure
    // relative tolerance is meaningful.
    for i in 0..50 {
        let spot = uniform(&mut rng, 50.0, 200.0);
        let strike = spot * uniform(&mut rng, 0.75, 1.30);
        let rate = uniform(&mut rng, 0.0, 0.08);
        let q = uniform(&mut rng, 0.0, 0.04);
        let vol = uniform(&mut rng, 0.15, 0.50);
        let expiry = uniform(&mut rng, 0.30, 2.0);

        for option_type in [OptionType::Call, OptionType::Put] {
            let greeks = bs_result(option_type, spot, strike, rate, q, vol, expiry)
                .greeks
                .expect("BS engine returns greeks");
            let price_at = |s: f64, r: f64, v: f64, t: f64| -> f64 {
                bs_price(option_type, s, strike, r, q, v, t)
            };

            let hs = spot * 1e-4;
            let fd_delta = (price_at(spot + hs, rate, vol, expiry)
                - price_at(spot - hs, rate, vol, expiry))
                / (2.0 * hs);

            let hg = spot * 1e-3;
            let fd_gamma = (price_at(spot + hg, rate, vol, expiry)
                - 2.0 * price_at(spot, rate, vol, expiry)
                + price_at(spot - hg, rate, vol, expiry))
                / (hg * hg);

            let hv = 1e-4;
            let fd_vega = (price_at(spot, rate, vol + hv, expiry)
                - price_at(spot, rate, vol - hv, expiry))
                / (2.0 * hv);

            let hr = 1e-5;
            let fd_rho = (price_at(spot, rate + hr, vol, expiry)
                - price_at(spot, rate - hr, vol, expiry))
                / (2.0 * hr);

            let ht = 1e-5;
            let fd_theta = -(price_at(spot, rate, vol, expiry + ht)
                - price_at(spot, rate, vol, expiry - ht))
                / (2.0 * ht);

            let checks = [
                ("delta", greeks.delta, fd_delta),
                ("gamma", greeks.gamma, fd_gamma),
                ("vega", greeks.vega, fd_vega),
                ("theta", greeks.theta, fd_theta),
                ("rho", greeks.rho, fd_rho),
            ];
            for (name, analytic, fd) in checks {
                // Relative 1e-5 with a small absolute floor for near-zero Greeks.
                let tol = 1e-5 * analytic.abs().max(1e-2);
                let err = (analytic - fd).abs();
                assert!(
                    err <= tol,
                    "{name} mismatch vs FD at point {i} ({option_type:?}): S={spot} \
                     K={strike} r={rate} q={q} vol={vol} T={expiry} analytic={analytic} \
                     fd={fd} err={err} tol={tol}"
                );
            }
        }
    }
}
