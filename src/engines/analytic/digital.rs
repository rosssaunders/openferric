//! Module `engines::analytic::digital`.
//!
//! Implements digital abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `DigitalAnalyticEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use crate::core::{Greeks, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::digital::{AssetOrNothingOption, CashOrNothingOption, GapOption};
use crate::market::Market;
use crate::math::{normal_cdf, normal_pdf};

/// Analytic Black-Scholes style engine for digital/binary options.
#[derive(Debug, Clone, Default)]
pub struct DigitalAnalyticEngine;

impl DigitalAnalyticEngine {
    /// Creates a digital analytic engine.
    pub fn new() -> Self {
        Self
    }
}

/// Zero Greeks returned at expiry for all digital types.
const ZERO_GREEKS: Greeks = Greeks {
    delta: 0.0,
    gamma: 0.0,
    vega: 0.0,
    theta: 0.0,
    rho: 0.0,
};

#[inline]
fn d1_d2(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, f64) {
    super::bs_inline::stable_d1_d2(spot, strike, rate, dividend_yield, vol, expiry)
}

#[inline]
fn cash_or_nothing_expiry(option_type: OptionType, spot: f64, strike: f64, cash: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > strike => cash,
        OptionType::Put if spot < strike => cash,
        _ => 0.0,
    }
}

#[inline]
fn asset_or_nothing_expiry(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > strike => spot,
        OptionType::Put if spot < strike => spot,
        _ => 0.0,
    }
}

#[inline]
fn gap_expiry(option_type: OptionType, spot: f64, payoff_strike: f64, trigger_strike: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > trigger_strike => spot - payoff_strike,
        OptionType::Put if spot < trigger_strike => payoff_strike - spot,
        _ => 0.0,
    }
}

/// Compute closed-form Greeks for a cash-or-nothing option.
///
/// Formulas derived from P_call = C·e^{-rT}·N(d2), with put via N(-d2) = 1 - N(d2).
/// Reference: Haug, "The Complete Guide to Option Pricing Formulas" (2nd ed.), §2.10.
#[inline]
#[allow(clippy::too_many_arguments)]
fn cash_or_nothing_greeks(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    expiry: f64,
    cash: f64,
) -> Greeks {
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let (d1, d2) = d1_d2(spot, strike, rate, q, vol, expiry);
    let df_r = (-rate * expiry).exp();
    let npd2 = normal_pdf(d2);

    // sign: +1 for call, -1 for put
    let (sign, nd2_signed) = match option_type {
        OptionType::Call => (1.0, normal_cdf(d2)),
        OptionType::Put => (-1.0, normal_cdf(-d2)),
    };

    let delta = (sign * cash * df_r * npd2 / spot) / sig_sqrt_t;
    let gamma = (((-sign * cash * df_r * npd2 * d1) / spot) / spot) / (sig_sqrt_t * sig_sqrt_t);

    // ∂d2/∂σ = -d1/σ, so vega = sign * C·df_r·n(d2)·(-d1/σ) (raw, per unit vol)
    let vega = -sign * cash * df_r * npd2 * d1 / vol;

    // theta = -∂P/∂T, with ∂d2/∂T = (r - q - σ²/2)/(σ√T) - d2/(2T)
    let dd2_dt = (rate - q) / sig_sqrt_t - 0.5 * vol / sqrt_t - d2 / (2.0 * expiry);
    let theta = cash * df_r * (rate * nd2_signed - sign * npd2 * dd2_dt);

    // rho = ∂P/∂r = C·df_r·(-T·Nd2_signed + sign·n(d2)·√T/σ) (raw, per unit rate)
    let rho = cash * df_r * (-expiry * nd2_signed + sign * npd2 * sqrt_t / vol);

    Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Compute closed-form Greeks for an asset-or-nothing option.
///
/// Formulas derived from P_call = S·e^{-qT}·N(d1), with put via N(-d1) = 1 - N(d1).
/// Reference: Haug, "The Complete Guide to Option Pricing Formulas" (2nd ed.), §2.10.
#[inline]
#[allow(clippy::too_many_arguments)]
fn asset_or_nothing_greeks(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    expiry: f64,
) -> Greeks {
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let (d1, d2) = d1_d2(spot, strike, rate, q, vol, expiry);
    let df_q = (-q * expiry).exp();
    let npd1 = normal_pdf(d1);

    let (sign, nd1_signed) = match option_type {
        OptionType::Call => (1.0, normal_cdf(d1)),
        OptionType::Put => (-1.0, normal_cdf(-d1)),
    };

    // delta = df_q·(Nd1_signed + sign·n(d1)/(σ√T))
    let delta = df_q * (nd1_signed + sign * npd1 / sig_sqrt_t);

    // gamma = -sign·df_q·n(d1)·d2/(S·σ²·T), since ∂d1/∂S = 1/(S·σ·√T)
    let gamma = ((-sign * df_q * npd1 * d2) / spot) / (sig_sqrt_t * sig_sqrt_t);

    // vega = -sign·S·df_q·n(d1)·d2/σ (since ∂d1/∂σ = -d2/σ; raw, per unit vol)
    let vega = -sign * spot * df_q * npd1 * d2 / vol;

    // theta = -∂P/∂T, with ∂d1/∂T = (r - q + σ²/2)/(σ√T) - d1/(2T)
    let dd1_dt = (rate - q) / sig_sqrt_t + 0.5 * vol / sqrt_t - d1 / (2.0 * expiry);
    let theta = spot * df_q * (q * nd1_signed - sign * npd1 * dd1_dt);

    // rho = sign·S·df_q·n(d1)·√T/σ (since ∂d1/∂r = √T/σ; raw, per unit rate)
    let rho = sign * spot * df_q * npd1 * sqrt_t / vol;

    Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

impl PricingEngine<CashOrNothingOption> for DigitalAnalyticEngine {
    fn price(
        &self,
        instrument: &CashOrNothingOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: cash_or_nothing_expiry(
                    instrument.option_type,
                    market.spot,
                    instrument.strike,
                    instrument.cash,
                ),
                stderr: None,
                greeks: Some(ZERO_GREEKS),
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;
        let q = market.effective_dividend_yield(instrument.expiry);

        let (_, d2) = d1_d2(
            market.spot,
            instrument.strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );
        let df_r = (-market.rate * instrument.expiry).exp();

        let price = match instrument.option_type {
            OptionType::Call => instrument.cash * df_r * normal_cdf(d2),
            OptionType::Put => instrument.cash * df_r * normal_cdf(-d2),
        };

        let greeks = cash_or_nothing_greeks(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
            instrument.cash,
        );

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d2", d2);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: Some(greeks),
            diagnostics,
        })
    }
}

impl PricingEngine<AssetOrNothingOption> for DigitalAnalyticEngine {
    fn price(
        &self,
        instrument: &AssetOrNothingOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: asset_or_nothing_expiry(
                    instrument.option_type,
                    market.spot,
                    instrument.strike,
                ),
                stderr: None,
                greeks: Some(ZERO_GREEKS),
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;
        let q = market.effective_dividend_yield(instrument.expiry);

        let (d1, _) = d1_d2(
            market.spot,
            instrument.strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );
        let df_q = (-q * instrument.expiry).exp();

        let price = match instrument.option_type {
            OptionType::Call => market.spot * df_q * normal_cdf(d1),
            OptionType::Put => market.spot * df_q * normal_cdf(-d1),
        };

        let greeks = asset_or_nothing_greeks(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d1", d1);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: Some(greeks),
            diagnostics,
        })
    }
}

impl PricingEngine<GapOption> for DigitalAnalyticEngine {
    fn price(
        &self,
        instrument: &GapOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: gap_expiry(
                    instrument.option_type,
                    market.spot,
                    instrument.payoff_strike,
                    instrument.trigger_strike,
                ),
                stderr: None,
                greeks: Some(ZERO_GREEKS),
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.checked_vol_for(instrument.trigger_strike, instrument.expiry)?;
        let q = market.effective_dividend_yield(instrument.expiry);

        let (d1, d2) = d1_d2(
            market.spot,
            instrument.trigger_strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );
        let df_r = (-market.rate * instrument.expiry).exp();

        // Decompose the gap into a cancellation-safe vanilla at the trigger
        // plus a cash-digital adjustment:
        // call = C(K2) + (K2-K1) df N(d2)
        // put  = P(K2) + (K1-K2) df N(-d2).
        let vanilla = super::black_scholes::bs_price(
            instrument.option_type,
            market.spot,
            instrument.trigger_strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );
        let price = match instrument.option_type {
            OptionType::Call => {
                vanilla
                    + (instrument.trigger_strike - instrument.payoff_strike) * df_r * normal_cdf(d2)
            }
            OptionType::Put => {
                vanilla
                    + (instrument.payoff_strike - instrument.trigger_strike)
                        * df_r
                        * normal_cdf(-d2)
            }
        };
        if !price.is_finite() {
            return Err(PricingError::NumericalError(format!(
                "gap option price is non-finite: {price}"
            )));
        }

        // Gap = asset-or-nothing(K2) - K1 * cash-or-nothing(K2, cash=1).
        // Greeks are the linear combination of the two building blocks.
        let asset_g = asset_or_nothing_greeks(
            instrument.option_type,
            market.spot,
            instrument.trigger_strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );
        let cash_g = cash_or_nothing_greeks(
            instrument.option_type,
            market.spot,
            instrument.trigger_strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
            1.0,
        );
        let k1 = instrument.payoff_strike;
        let greeks = match instrument.option_type {
            OptionType::Call => Greeks {
                delta: asset_g.delta - k1 * cash_g.delta,
                gamma: asset_g.gamma - k1 * cash_g.gamma,
                vega: asset_g.vega - k1 * cash_g.vega,
                theta: asset_g.theta - k1 * cash_g.theta,
                rho: asset_g.rho - k1 * cash_g.rho,
            },
            OptionType::Put => {
                // Put = K1·df_r·N(-d2) - S·df_q·N(-d1)
                //     = -(asset_put) + K1·(cash_put)
                // asset_put pays S when S < K2, cash_put pays 1 when S < K2
                // gap_put = K1·cash_put - asset_put
                Greeks {
                    delta: k1 * cash_g.delta - asset_g.delta,
                    gamma: k1 * cash_g.gamma - asset_g.gamma,
                    vega: k1 * cash_g.vega - asset_g.vega,
                    theta: k1 * cash_g.theta - asset_g.theta,
                    rho: k1 * cash_g.rho - asset_g.rho,
                }
            }
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d1", d1);
        diagnostics.insert("d2", d2);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: Some(greeks),
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn cash_or_nothing_matches_haug_reference() {
        let instrument = CashOrNothingOption::new(OptionType::Call, 80.0, 10.0, 0.75);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.06)
            .dividend_yield(0.06)
            .flat_vol(0.35)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        // Independently evaluated from cash*exp(-rT)*N(d2) with SciPy
        // 1.17.1's normal CDF.  The old 6.9358 target used a different
        // convention and only passed because the tolerance was five cents.
        assert_relative_eq!(price, 6.888_929_133_869_653, epsilon = 2e-12);
    }

    #[test]
    fn asset_or_nothing_put_matches_haug_reference_value() {
        let instrument = AssetOrNothingOption::new(OptionType::Put, 65.0, 0.50);
        let market = Market::builder()
            .spot(70.0)
            .rate(0.07)
            .dividend_yield(0.05)
            .flat_vol(0.27)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        assert_relative_eq!(price, 20.206_947_298_368_55, epsilon = 2e-12);
    }

    #[test]
    fn gap_call_matches_haug_reference() {
        let instrument = GapOption::new(OptionType::Call, 57.0, 50.0, 0.50);
        let market = Market::builder()
            .spot(50.0)
            .rate(0.09)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        assert_relative_eq!(price, -0.005_252_489_258_786_852, epsilon = 2e-12);
    }

    #[test]
    fn cash_put_preserves_deep_tail_probability() {
        // d2 is exactly 9. The reference is 1e20 * Phi(-9), computed with
        // 100-decimal erfc arithmetic.
        const EXPECTED: f64 = 11.285_884_059_538_406;
        let instrument = CashOrNothingOption::new(OptionType::Put, 1.0, 1.0e20, 1.0);
        let market = Market::builder()
            .spot(1.82_f64.exp())
            .rate(0.0)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let result = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap();
        assert!(
            ((result.price - EXPECTED) / EXPECTED).abs() <= 2.0e-14,
            "price={:.17e}",
            result.price
        );
        let greeks = result.greeks.unwrap();
        assert!(greeks.delta < 0.0);
        assert!(greeks.rho.is_finite() && greeks.rho != 0.0);
    }

    #[test]
    fn vanilla_equivalent_gap_put_preserves_deep_tail_value() {
        const EXPECTED: f64 = 22.752_884_600_977_636;
        let instrument = GapOption::new(OptionType::Put, 1.0e18, 1.0e18, 1.0);
        let market = Market::builder()
            .spot(5.0e18)
            .rate(0.0)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;
        assert!(price >= 0.0);
        assert!(
            ((price - EXPECTED) / EXPECTED).abs() <= 2.0e-12,
            "price={price:.17e}"
        );
    }

    fn test_market() -> Market {
        Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.25)
            .build()
            .unwrap()
    }

    fn assert_scipy_greeks(actual: Greeks, expected: [f64; 5]) {
        let actual = [
            actual.delta,
            actual.gamma,
            actual.vega,
            actual.theta,
            actual.rho,
        ];
        for (name, (got, want)) in ["delta", "gamma", "vega", "theta", "rho"]
            .into_iter()
            .zip(actual.into_iter().zip(expected))
        {
            let error = (got - want).abs();
            let tolerance = 3.0e-12 * want.abs().max(1.0);
            assert!(
                error <= tolerance,
                "{name}: got={got} reference={want} error={error} tolerance={tolerance}"
            );
        }
    }

    // Prices and analytic derivatives below were independently evaluated with
    // SciPy 1.17.1's norm.cdf/PDF formulas. Unlike the former finite-difference
    // checks, these targets do not carry arbitrary bump-size error.

    #[test]
    fn cash_or_nothing_call_matches_scipy_price_and_greeks() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = CashOrNothingOption::new(OptionType::Call, 105.0, 10.0, 0.50);

        let result = engine.price(&inst, &market).unwrap();
        assert_relative_eq!(result.price, 3.802_902_839_262_12, epsilon = 3.0e-12);
        assert_scipy_greeks(
            result.greeks.unwrap(),
            [
                0.211_670_298_635_450_84,
                0.001_230_408_231_841_815_8,
                1.538_010_289_802_269_8,
                -0.829_368_326_393_814_1,
                8.682_063_512_141_482,
            ],
        );
    }

    #[test]
    fn cash_or_nothing_put_matches_scipy_price_and_greeks() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = CashOrNothingOption::new(OptionType::Put, 95.0, 5.0, 1.0);

        let result = engine.price(&inst, &market).unwrap();
        assert_relative_eq!(result.price, 2.000_780_642_444_971_2, epsilon = 3.0e-12);
        assert_scipy_greeks(
            result.greeks.unwrap(),
            [
                -0.074_391_685_594_942_44,
                0.001_339_565_659_503_633,
                3.348_914_148_759_082,
                -0.095_400_179_687_809_38,
                -9.439_949_201_939_216,
            ],
        );
    }

    #[test]
    fn asset_or_nothing_call_matches_scipy_price_and_greeks() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = AssetOrNothingOption::new(OptionType::Call, 105.0, 0.50);

        let result = engine.price(&inst, &market).unwrap();
        assert_relative_eq!(result.price, 45.450_974_561_703_276, epsilon = 3.0e-12);
        assert_scipy_greeks(
            result.greeks.unwrap(),
            [
                2.677_047_881_289_266_4,
                0.035_144_667_791_061_4,
                43.930_834_738_826_75,
                -16.741_303_600_489_32,
                111.126_906_783_611_69,
            ],
        );
    }

    #[test]
    fn asset_or_nothing_put_matches_scipy_price_and_greeks() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = AssetOrNothingOption::new(OptionType::Put, 95.0, 1.0);

        let result = engine.price(&inst, &market).unwrap();
        assert_relative_eq!(result.price, 31.983_175_746_098_72, epsilon = 3.0e-12);
        assert_scipy_greeks(
            result.greeks.unwrap(),
            [
                -1.093_610_268_842_919_2,
                0.011_317_327_267_529_96,
                28.293_318_168_824_904,
                1.343_324_822_730_580_5,
                -141.344_202_630_390_64,
            ],
        );
    }

    #[test]
    fn gap_call_matches_scipy_price_and_greeks() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = GapOption::new(OptionType::Call, 102.0, 105.0, 0.50);

        let result = engine.price(&inst, &market).unwrap();
        assert_relative_eq!(result.price, 6.661_365_601_229_662, epsilon = 3.0e-12);
        assert_scipy_greeks(
            result.greeks.unwrap(),
            [
                0.518_010_835_207_668_2,
                0.022_594_503_826_274_884,
                28.243_129_782_843_603,
                -8.281_746_671_272_419,
                22.569_858_959_768_58,
            ],
        );
    }

    #[test]
    fn gap_put_matches_scipy_price_and_greeks() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = GapOption::new(OptionType::Put, 98.0, 95.0, 1.0);

        let result = engine.price(&inst, &market).unwrap();
        assert_relative_eq!(result.price, 7.232_124_845_822_707, epsilon = 3.0e-12);
        assert_scipy_greeks(
            result.greeks.unwrap(),
            [
                -0.364_466_768_817_952_8,
                0.014_938_159_658_741_247,
                37.345_399_146_853_104,
                -3.213_168_344_611_644,
                -43.678_801_727_617_98,
            ],
        );
    }

    // --- Expiry edge case ---

    #[test]
    fn all_digitals_return_zero_greeks_at_expiry() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();

        let cash = CashOrNothingOption::new(OptionType::Call, 100.0, 10.0, 0.0);
        let g = engine.price(&cash, &market).unwrap().greeks.unwrap();
        assert_eq!(g.delta, 0.0);
        assert_eq!(g.gamma, 0.0);
        assert_eq!(g.vega, 0.0);
        assert_eq!(g.theta, 0.0);
        assert_eq!(g.rho, 0.0);

        let asset = AssetOrNothingOption::new(OptionType::Put, 100.0, 0.0);
        let g = engine.price(&asset, &market).unwrap().greeks.unwrap();
        assert_eq!(g.delta, 0.0);

        let gap = GapOption::new(OptionType::Call, 100.0, 100.0, 0.0);
        let g = engine.price(&gap, &market).unwrap().greeks.unwrap();
        assert_eq!(g.delta, 0.0);
    }

    // --- Sanity checks ---

    #[test]
    fn cash_or_nothing_greeks_present() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = CashOrNothingOption::new(OptionType::Call, 100.0, 10.0, 1.0);
        let result = engine.price(&inst, &market).unwrap();
        assert!(result.greeks.is_some());
        let g = result.greeks.unwrap();
        assert!(g.delta > 0.0, "call delta should be positive");
    }

    #[test]
    fn asset_or_nothing_greeks_present() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = AssetOrNothingOption::new(OptionType::Call, 100.0, 1.0);
        let result = engine.price(&inst, &market).unwrap();
        assert!(result.greeks.is_some());
        let g = result.greeks.unwrap();
        assert!(g.delta > 0.0, "call delta should be positive");
    }

    #[test]
    fn gap_greeks_present() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = GapOption::new(OptionType::Call, 100.0, 100.0, 1.0);
        let result = engine.price(&inst, &market).unwrap();
        assert!(result.greeks.is_some());
    }
}
