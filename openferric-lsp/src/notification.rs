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
    pub payoff_profile: Vec<PayoffPoint>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreeksEntry {
    pub asset: String,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffPoint {
    pub spot_pct: f64,
    pub pv: f64,
}
