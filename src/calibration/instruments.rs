//! Calibration instrument contracts.

use serde::{Deserialize, Serialize};

use crate::calibration::core::InstrumentError;

/// Shared quote behavior for bid/ask-aware residual handling.
pub trait VolQuote {
    fn id(&self) -> &str;
    fn market_mid(&self) -> f64;
    fn market_bid(&self) -> Option<f64>;
    fn market_ask(&self) -> Option<f64>;
    fn weight(&self) -> f64;
    fn is_liquid(&self) -> bool;
}

/// Vanilla option implied-vol quote for model calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionVolQuote {
    pub id: String,
    pub strike: f64,
    pub maturity: f64,
    pub market_vol: f64,
    pub bid_vol: Option<f64>,
    pub ask_vol: Option<f64>,
    pub weight: f64,
    pub liquid: bool,
}

impl OptionVolQuote {
    pub fn new(id: impl Into<String>, strike: f64, maturity: f64, market_vol: f64) -> Self {
        Self {
            id: id.into(),
            strike,
            maturity,
            market_vol,
            bid_vol: None,
            ask_vol: None,
            weight: 1.0,
            liquid: true,
        }
    }
}

impl VolQuote for OptionVolQuote {
    fn id(&self) -> &str {
        &self.id
    }

    fn market_mid(&self) -> f64 {
        self.market_vol
    }

    fn market_bid(&self) -> Option<f64> {
        self.bid_vol
    }

    fn market_ask(&self) -> Option<f64> {
        self.ask_vol
    }

    fn weight(&self) -> f64 {
        self.weight.max(1e-12)
    }

    fn is_liquid(&self) -> bool {
        self.liquid
    }
}

/// ATM swaption vol quote used by Hull-White calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwaptionVolQuote {
    pub id: String,
    pub expiry: f64,
    pub tenor: f64,
    pub market_vol: f64,
    pub bid_vol: Option<f64>,
    pub ask_vol: Option<f64>,
    pub weight: f64,
    pub liquid: bool,
}

impl SwaptionVolQuote {
    pub fn new(id: impl Into<String>, expiry: f64, tenor: f64, market_vol: f64) -> Self {
        Self {
            id: id.into(),
            expiry,
            tenor,
            market_vol,
            bid_vol: None,
            ask_vol: None,
            weight: 1.0,
            liquid: true,
        }
    }
}

impl VolQuote for SwaptionVolQuote {
    fn id(&self) -> &str {
        &self.id
    }

    fn market_mid(&self) -> f64 {
        self.market_vol
    }

    fn market_bid(&self) -> Option<f64> {
        self.bid_vol
    }

    fn market_ask(&self) -> Option<f64> {
        self.ask_vol
    }

    fn weight(&self) -> f64 {
        self.weight.max(1e-12)
    }

    fn is_liquid(&self) -> bool {
        self.liquid
    }
}

/// Returns `(signed_error, effective_error, within_bid_ask)` in vol units.
pub fn bid_ask_aware_error<Q: VolQuote>(quote: &Q, model_vol: f64) -> (f64, f64, bool) {
    let signed = model_vol - quote.market_mid();

    match (quote.market_bid(), quote.market_ask()) {
        (Some(bid), Some(ask)) if bid <= ask => {
            if model_vol < bid {
                (signed, model_vol - bid, false)
            } else if model_vol > ask {
                (signed, model_vol - ask, false)
            } else {
                (signed, 0.0, true)
            }
        }
        _ => (signed, signed, false),
    }
}

pub fn make_error_record<Q: VolQuote>(quote: &Q, model_vol: f64) -> InstrumentError {
    let (signed, effective, within_bid_ask) = bid_ask_aware_error(quote, model_vol);
    InstrumentError {
        id: quote.id().to_string(),
        market_mid: quote.market_mid(),
        market_bid: quote.market_bid(),
        market_ask: quote.market_ask(),
        model: model_vol,
        signed_error: signed,
        effective_error: effective,
        abs_error: signed.abs(),
        weight: quote.weight(),
        within_bid_ask,
        liquid: quote.is_liquid(),
    }
}
