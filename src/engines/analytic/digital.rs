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
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;
    (d1, d2)
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
    let nd2 = normal_cdf(d2);

    // sign: +1 for call, -1 for put
    let (sign, nd2_signed) = match option_type {
        OptionType::Call => (1.0, nd2),
        OptionType::Put => (-1.0, 1.0 - nd2),
    };

    let delta = sign * cash * df_r * npd2 / (spot * sig_sqrt_t);
    let gamma = -sign * cash * df_r * npd2 * d1 / (spot * spot * vol * vol * expiry);

    // ∂d2/∂σ = -d1/σ, so vega_raw = sign * C·df_r·n(d2)·(-d1/σ)
    let vega = -sign * cash * df_r * npd2 * d1 / vol / 100.0;

    // theta = -∂P/∂T, with ∂d2/∂T = (r - q - σ²/2)/(σ√T) - d2/(2T)
    let dd2_dt = (rate - q - 0.5 * vol * vol) / sig_sqrt_t - d2 / (2.0 * expiry);
    let theta = cash * df_r * (rate * nd2_signed - sign * npd2 * dd2_dt);

    // rho = ∂P/∂r = C·df_r·(-T·Nd2_signed + sign·n(d2)·√T/σ), per 1%
    let rho = cash * df_r * (-expiry * nd2_signed + sign * npd2 * sqrt_t / vol) / 100.0;

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
    let nd1 = normal_cdf(d1);

    let (sign, nd1_signed) = match option_type {
        OptionType::Call => (1.0, nd1),
        OptionType::Put => (-1.0, 1.0 - nd1),
    };

    // delta = df_q·(Nd1_signed + sign·n(d1)/(σ√T))
    let delta = df_q * (nd1_signed + sign * npd1 / sig_sqrt_t);

    // gamma = -sign·df_q·n(d1)·d2/(S·σ²·T), since ∂d1/∂S = 1/(S·σ·√T)
    let gamma = -sign * df_q * npd1 * d2 / (spot * vol * vol * expiry);

    // vega_raw = -sign·S·df_q·n(d1)·d2/σ (since ∂d1/∂σ = -d2/σ)
    let vega = -sign * spot * df_q * npd1 * d2 / vol / 100.0;

    // theta = -∂P/∂T, with ∂d1/∂T = (r - q + σ²/2)/(σ√T) - d1/(2T)
    let dd1_dt = (rate - q + 0.5 * vol * vol) / sig_sqrt_t - d1 / (2.0 * expiry);
    let theta = spot * df_q * (q * nd1_signed - sign * npd1 * dd1_dt);

    // rho = sign·S·df_q·n(d1)·√T/σ (since ∂d1/∂r = √T/σ), per 1%
    let rho = sign * spot * df_q * npd1 * sqrt_t / vol / 100.0;

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

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }
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

        // Compute N(d2) once; derive put price via N(-d2) = 1 - N(d2).
        let nd2 = normal_cdf(d2);
        let price = match instrument.option_type {
            OptionType::Call => instrument.cash * df_r * nd2,
            OptionType::Put => instrument.cash * df_r * (1.0 - nd2),
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

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }
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

        // Compute N(d1) once; derive put price via N(-d1) = 1 - N(d1).
        let nd1 = normal_cdf(d1);
        let price = match instrument.option_type {
            OptionType::Call => market.spot * df_q * nd1,
            OptionType::Put => market.spot * df_q * (1.0 - nd1),
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

        let vol = market.vol_for(instrument.trigger_strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }
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
        let df_q = (-q * instrument.expiry).exp();

        // Compute N(d1), N(d2) once; derive put via N(-d) = 1 - N(d).
        let nd1 = normal_cdf(d1);
        let nd2 = normal_cdf(d2);
        let call = market
            .spot
            .mul_add(df_q * nd1, -(instrument.payoff_strike * df_r * nd2));
        let price = match instrument.option_type {
            OptionType::Call => call,
            OptionType::Put => call - market.spot * df_q + instrument.payoff_strike * df_r,
        };

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

        assert_relative_eq!(price, 6.9358, epsilon = 5e-2);
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

        assert_relative_eq!(price, 20.2069, epsilon = 2e-4);
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

        assert_relative_eq!(price, -0.0053, epsilon = 2e-4);
    }

    // --- Finite-difference verification helpers ---

    fn bump_spot(market: &Market, ds: f64) -> Market {
        Market::builder()
            .spot(market.spot + ds)
            .rate(market.rate)
            .dividend_yield(market.dividend_yield)
            .flat_vol(market.vol_for(100.0, 1.0)) // flat vol
            .build()
            .unwrap()
    }

    fn bump_rate(market: &Market, dr: f64) -> Market {
        Market::builder()
            .spot(market.spot)
            .rate(market.rate + dr)
            .dividend_yield(market.dividend_yield)
            .flat_vol(market.vol_for(100.0, 1.0))
            .build()
            .unwrap()
    }

    fn bump_vol(market: &Market, dv: f64) -> Market {
        Market::builder()
            .spot(market.spot)
            .rate(market.rate)
            .dividend_yield(market.dividend_yield)
            .flat_vol(market.vol_for(100.0, 1.0) + dv)
            .build()
            .unwrap()
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

    // --- Cash-or-nothing Greeks tests ---

    #[test]
    fn cash_or_nothing_call_greeks_vs_finite_diff() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = CashOrNothingOption::new(OptionType::Call, 105.0, 10.0, 0.50);

        let result = engine.price(&inst, &market).unwrap();
        let g = result.greeks.unwrap();

        let ds = 0.01;
        let p_up = engine.price(&inst, &bump_spot(&market, ds)).unwrap().price;
        let p_dn = engine.price(&inst, &bump_spot(&market, -ds)).unwrap().price;
        let fd_delta = (p_up - p_dn) / (2.0 * ds);
        let fd_gamma = (p_up - 2.0 * result.price + p_dn) / (ds * ds);

        assert_relative_eq!(g.delta, fd_delta, epsilon = 1e-4);
        assert_relative_eq!(g.gamma, fd_gamma, epsilon = 1e-2);

        // Vega (per 1%): bump vol by 0.01 (1%), compare to analytic
        let dv = 1e-5;
        let p_vup = engine.price(&inst, &bump_vol(&market, dv)).unwrap().price;
        let p_vdn = engine.price(&inst, &bump_vol(&market, -dv)).unwrap().price;
        let fd_vega = (p_vup - p_vdn) / (2.0 * dv) / 100.0;
        assert_relative_eq!(g.vega, fd_vega, epsilon = 1e-4);

        // Rho (per 1%)
        let dr = 1e-5;
        let p_rup = engine.price(&inst, &bump_rate(&market, dr)).unwrap().price;
        let p_rdn = engine.price(&inst, &bump_rate(&market, -dr)).unwrap().price;
        let fd_rho = (p_rup - p_rdn) / (2.0 * dr) / 100.0;
        assert_relative_eq!(g.rho, fd_rho, epsilon = 1e-4);

        // Theta: bump expiry
        let dt = 1e-5;
        let inst_up =
            CashOrNothingOption::new(inst.option_type, inst.strike, inst.cash, inst.expiry + dt);
        let inst_dn =
            CashOrNothingOption::new(inst.option_type, inst.strike, inst.cash, inst.expiry - dt);
        let p_tup = engine.price(&inst_up, &market).unwrap().price;
        let p_tdn = engine.price(&inst_dn, &market).unwrap().price;
        let fd_theta = -(p_tup - p_tdn) / (2.0 * dt);
        assert_relative_eq!(g.theta, fd_theta, epsilon = 1e-3);
    }

    #[test]
    fn cash_or_nothing_put_greeks_vs_finite_diff() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = CashOrNothingOption::new(OptionType::Put, 95.0, 5.0, 1.0);

        let result = engine.price(&inst, &market).unwrap();
        let g = result.greeks.unwrap();

        let ds = 0.01;
        let p_up = engine.price(&inst, &bump_spot(&market, ds)).unwrap().price;
        let p_dn = engine.price(&inst, &bump_spot(&market, -ds)).unwrap().price;
        let fd_delta = (p_up - p_dn) / (2.0 * ds);
        assert_relative_eq!(g.delta, fd_delta, epsilon = 1e-4);
        assert!(g.delta < 0.0, "put delta should be negative");
    }

    // --- Asset-or-nothing Greeks tests ---

    #[test]
    fn asset_or_nothing_call_greeks_vs_finite_diff() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = AssetOrNothingOption::new(OptionType::Call, 105.0, 0.50);

        let result = engine.price(&inst, &market).unwrap();
        let g = result.greeks.unwrap();

        let ds = 0.01;
        let p_up = engine.price(&inst, &bump_spot(&market, ds)).unwrap().price;
        let p_dn = engine.price(&inst, &bump_spot(&market, -ds)).unwrap().price;
        let fd_delta = (p_up - p_dn) / (2.0 * ds);
        let fd_gamma = (p_up - 2.0 * result.price + p_dn) / (ds * ds);

        assert_relative_eq!(g.delta, fd_delta, epsilon = 1e-4);
        assert_relative_eq!(g.gamma, fd_gamma, epsilon = 1e-2);

        let dv = 1e-5;
        let p_vup = engine.price(&inst, &bump_vol(&market, dv)).unwrap().price;
        let p_vdn = engine.price(&inst, &bump_vol(&market, -dv)).unwrap().price;
        let fd_vega = (p_vup - p_vdn) / (2.0 * dv) / 100.0;
        assert_relative_eq!(g.vega, fd_vega, epsilon = 1e-4);

        let dr = 1e-5;
        let p_rup = engine.price(&inst, &bump_rate(&market, dr)).unwrap().price;
        let p_rdn = engine.price(&inst, &bump_rate(&market, -dr)).unwrap().price;
        let fd_rho = (p_rup - p_rdn) / (2.0 * dr) / 100.0;
        assert_relative_eq!(g.rho, fd_rho, epsilon = 1e-4);

        let dt = 1e-5;
        let inst_up = AssetOrNothingOption::new(inst.option_type, inst.strike, inst.expiry + dt);
        let inst_dn = AssetOrNothingOption::new(inst.option_type, inst.strike, inst.expiry - dt);
        let p_tup = engine.price(&inst_up, &market).unwrap().price;
        let p_tdn = engine.price(&inst_dn, &market).unwrap().price;
        let fd_theta = -(p_tup - p_tdn) / (2.0 * dt);
        assert_relative_eq!(g.theta, fd_theta, epsilon = 1e-3);
    }

    #[test]
    fn asset_or_nothing_put_greeks_vs_finite_diff() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = AssetOrNothingOption::new(OptionType::Put, 95.0, 1.0);

        let result = engine.price(&inst, &market).unwrap();
        let g = result.greeks.unwrap();

        let ds = 0.01;
        let p_up = engine.price(&inst, &bump_spot(&market, ds)).unwrap().price;
        let p_dn = engine.price(&inst, &bump_spot(&market, -ds)).unwrap().price;
        let fd_delta = (p_up - p_dn) / (2.0 * ds);
        assert_relative_eq!(g.delta, fd_delta, epsilon = 1e-4);
    }

    // --- Gap option Greeks tests ---

    #[test]
    fn gap_call_greeks_vs_finite_diff() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = GapOption::new(OptionType::Call, 102.0, 105.0, 0.50);

        let result = engine.price(&inst, &market).unwrap();
        let g = result.greeks.unwrap();

        let ds = 0.01;
        let p_up = engine.price(&inst, &bump_spot(&market, ds)).unwrap().price;
        let p_dn = engine.price(&inst, &bump_spot(&market, -ds)).unwrap().price;
        let fd_delta = (p_up - p_dn) / (2.0 * ds);
        let fd_gamma = (p_up - 2.0 * result.price + p_dn) / (ds * ds);

        assert_relative_eq!(g.delta, fd_delta, epsilon = 1e-4);
        assert_relative_eq!(g.gamma, fd_gamma, epsilon = 1e-2);

        let dv = 1e-5;
        let p_vup = engine.price(&inst, &bump_vol(&market, dv)).unwrap().price;
        let p_vdn = engine.price(&inst, &bump_vol(&market, -dv)).unwrap().price;
        let fd_vega = (p_vup - p_vdn) / (2.0 * dv) / 100.0;
        assert_relative_eq!(g.vega, fd_vega, epsilon = 1e-4);

        let dr = 1e-5;
        let p_rup = engine.price(&inst, &bump_rate(&market, dr)).unwrap().price;
        let p_rdn = engine.price(&inst, &bump_rate(&market, -dr)).unwrap().price;
        let fd_rho = (p_rup - p_rdn) / (2.0 * dr) / 100.0;
        assert_relative_eq!(g.rho, fd_rho, epsilon = 1e-4);

        let dt = 1e-5;
        let inst_up = GapOption::new(
            inst.option_type,
            inst.payoff_strike,
            inst.trigger_strike,
            inst.expiry + dt,
        );
        let inst_dn = GapOption::new(
            inst.option_type,
            inst.payoff_strike,
            inst.trigger_strike,
            inst.expiry - dt,
        );
        let p_tup = engine.price(&inst_up, &market).unwrap().price;
        let p_tdn = engine.price(&inst_dn, &market).unwrap().price;
        let fd_theta = -(p_tup - p_tdn) / (2.0 * dt);
        assert_relative_eq!(g.theta, fd_theta, epsilon = 1e-3);
    }

    #[test]
    fn gap_put_greeks_vs_finite_diff() {
        let engine = DigitalAnalyticEngine::new();
        let market = test_market();
        let inst = GapOption::new(OptionType::Put, 98.0, 95.0, 1.0);

        let result = engine.price(&inst, &market).unwrap();
        let g = result.greeks.unwrap();

        let ds = 0.01;
        let p_up = engine.price(&inst, &bump_spot(&market, ds)).unwrap().price;
        let p_dn = engine.price(&inst, &bump_spot(&market, -ds)).unwrap().price;
        let fd_delta = (p_up - p_dn) / (2.0 * ds);
        assert_relative_eq!(g.delta, fd_delta, epsilon = 1e-4);

        let dv = 1e-5;
        let p_vup = engine.price(&inst, &bump_vol(&market, dv)).unwrap().price;
        let p_vdn = engine.price(&inst, &bump_vol(&market, -dv)).unwrap().price;
        let fd_vega = (p_vup - p_vdn) / (2.0 * dv) / 100.0;
        assert_relative_eq!(g.vega, fd_vega, epsilon = 1e-4);
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
