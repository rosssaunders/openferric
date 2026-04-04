use serde::{Deserialize, Serialize};

/// Custom LSP notification sent when pricing completes.
pub enum PricingNotification {}

impl tower_lsp::lsp_types::notification::Notification for PricingNotification {
    type Params = PricingResultPayload;
    const METHOD: &'static str = "openferric/pricingResult";
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PricingResultPayload {
    pub product_name: String,
    pub notional: f64,
    pub maturity: f64,
    pub underlyings: Vec<String>,
    pub price: f64,
    pub stderr: Option<f64>,
    pub greeks: Vec<GreeksEntry>,
    pub cross_greeks: Vec<CrossGreeksEntry>,
    pub payoff_profile: Vec<PayoffPoint>,
    pub error: Option<String>,
    pub market: Option<MarketSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarketSnapshot {
    pub rate: f64,
    pub assets: Vec<AssetSnapshot>,
    pub correlation: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetSnapshot {
    pub name: String,
    pub spot: f64,
    pub vol: f64,
    pub dividend_yield: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreeksEntry {
    pub asset: String,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrossGreeksEntry {
    pub asset_i: String,
    pub asset_j: String,
    pub cross_gamma: f64,
    pub corr_sens: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffPoint {
    pub spot_pct: f64,
    pub pv: f64,
}
