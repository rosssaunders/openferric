use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openferric::dsl::ast::ProductDef;
use openferric::dsl::engine::DslMonteCarloEngine;
use openferric::dsl::ir::CompiledProduct;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::codelens;
use crate::completion;
use crate::diagnostics;
use crate::document_symbols;
use crate::goto_def;
use crate::hover;
use crate::notification::{
    self, AssetSnapshot, CrossGreeksEntry, GreeksEntry, MarketSnapshot, PayoffPoint,
    PricingResultPayload,
};
use crate::semantic_tokens;
use crate::symbols::{self, SymbolTable};

/// Per-document state cached by the LSP server.
pub struct DocumentState {
    pub source: String,
    pub ast: Option<ProductDef>,
    pub product: Option<CompiledProduct>,
    pub symbols: SymbolTable,
}

pub struct Backend {
    client: Client,
    documents: Mutex<HashMap<Url, DocumentState>>,
    pricing_enabled: Mutex<bool>,
    pricing_config: Mutex<PricingConfig>,
    market_config: Mutex<Option<serde_json::Value>>,
    /// Last code lenses computed by the background pricing task, per document.
    /// `code_lens` serves from this cache so the request path never runs
    /// Monte Carlo pricing inline.
    lens_cache: Arc<Mutex<HashMap<Url, Vec<CodeLens>>>>,
}

/// Bounds applied to the `numPaths` value coming from client configuration so
/// a misconfigured client cannot stall the server with huge simulations.
const MIN_NUM_PATHS: u64 = 1_000;
const MAX_NUM_PATHS: u64 = 200_000;

#[derive(Clone)]
pub struct PricingConfig {
    pub num_paths: u32,
    pub num_steps: u32,
    pub seed: u64,
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self {
            num_paths: 50_000,
            num_steps: 100,
            seed: 42,
        }
    }
}

impl Backend {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: Mutex::new(HashMap::new()),
            pricing_enabled: Mutex::new(true),
            pricing_config: Mutex::new(PricingConfig::default()),
            market_config: Mutex::new(None),
            lens_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn update_document(&self, uri: &Url, source: String) -> Vec<Diagnostic> {
        let (ast, product, diags) = diagnostics::parse_and_diagnose(&source);
        let symbol_table = ast
            .as_ref()
            .map(|a| symbols::build_symbol_table(a, &source))
            .unwrap_or_default();

        let state = DocumentState {
            source,
            ast,
            product,
            symbols: symbol_table,
        };
        self.documents.lock().unwrap().insert(uri.clone(), state);
        diags
    }

    /// Handle custom `openferric/updateMarket` notification from the client.
    pub async fn on_update_market(&self, params: serde_json::Value) {
        *self.market_config.lock().unwrap() = Some(params);
        self.send_pricing_notification();
    }

    /// Spawn a background task to compute pricing, refresh the code lens
    /// cache, and send a pricing notification. All Monte Carlo work runs on
    /// the blocking thread pool so the request path stays responsive.
    fn send_pricing_notification(&self) {
        let enabled = *self.pricing_enabled.lock().unwrap();
        if !enabled {
            return;
        }

        // Extract compiled product data under the lock.
        let product_data = {
            let docs = self.documents.lock().unwrap();
            docs.iter().find_map(|(uri, state)| {
                let product = state.product.clone()?;
                let span_start = state.ast.as_ref().map(|a| a.span.start).unwrap_or(0);
                Some((uri.clone(), product, span_start, state.source.clone()))
            })
        };
        let Some((uri, product, product_span_start, source)) = product_data else {
            return;
        };

        let pricing_cfg = self.pricing_config.lock().unwrap().clone();
        let market_cfg = self.market_config.lock().unwrap().clone();
        let client = self.client.clone();
        let lens_cache = Arc::clone(&self.lens_cache);

        tokio::spawn(async move {
            let computed = tokio::task::spawn_blocking(move || {
                let payload = compute_pricing_payload(&product, &pricing_cfg, market_cfg.as_ref());
                let lenses = codelens::code_lenses(
                    &product,
                    product_span_start,
                    &source,
                    &pricing_cfg,
                    market_cfg.as_ref(),
                );
                (payload, lenses)
            })
            .await;

            let Ok((payload, lenses)) = computed else {
                return;
            };

            lens_cache.lock().unwrap().insert(uri, lenses);
            client
                .send_notification::<notification::PricingNotification>(payload)
                .await;
            // Ask the client to re-request code lenses now that the cache is
            // fresh; ignore errors from clients without refresh support.
            let _ = client.code_lens_refresh().await;
        });
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        // Read initialization options for pricing config.
        if let Some(opts) = params.initialization_options {
            if let Some(pricing) = opts.get("pricing") {
                let mut cfg = self.pricing_config.lock().unwrap();
                if let Some(v) = pricing.get("numPaths").and_then(|v| v.as_u64()) {
                    // Clamp user-controlled path counts so a misconfigured
                    // client cannot stall the server.
                    cfg.num_paths = v.clamp(MIN_NUM_PATHS, MAX_NUM_PATHS) as u32;
                }
                if let Some(v) = pricing.get("numSteps").and_then(|v| v.as_u64()) {
                    cfg.num_steps = v.clamp(1, 10_000) as u32;
                }
                if let Some(v) = pricing.get("seed").and_then(|v| v.as_u64()) {
                    cfg.seed = v;
                }
                if let Some(v) = pricing.get("enabled").and_then(|v| v.as_bool()) {
                    *self.pricing_enabled.lock().unwrap() = v;
                }
            }
            if let Some(market) = opts.get("market") {
                *self.market_config.lock().unwrap() = Some(market.clone());
            }
        }

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![" ".into(), ".".into(), "(".into()]),
                    ..Default::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                code_lens_provider: Some(CodeLensOptions {
                    resolve_provider: Some(false),
                }),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        semantic_tokens::capabilities(),
                    ),
                ),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "openferric-lsp initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let diags = self.update_document(&uri, params.text_document.text);
        self.client.publish_diagnostics(uri, diags, None).await;
        self.send_pricing_notification();
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        if let Some(change) = params.content_changes.into_iter().last() {
            let diags = self.update_document(&uri, change.text);
            self.client.publish_diagnostics(uri, diags, None).await;
            self.send_pricing_notification();
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.lock().unwrap().remove(&uri);
        self.lens_cache.lock().unwrap().remove(&uri);
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        let docs = self.documents.lock().unwrap();
        let items = match docs.get(uri) {
            Some(state) => completion::completions(state, pos),
            None => vec![],
        };
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let docs = self.documents.lock().unwrap();
        Ok(docs.get(uri).and_then(|state| hover::hover(state, pos)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let docs = self.documents.lock().unwrap();
        Ok(docs.get(uri).and_then(|state| {
            goto_def::goto_definition(state, pos).map(|range| {
                GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range,
                })
            })
        }))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        let docs = self.documents.lock().unwrap();
        Ok(docs.get(uri).map(|state| {
            DocumentSymbolResponse::Flat(document_symbols::document_symbols(state, uri))
        }))
    }

    async fn code_lens(&self, params: CodeLensParams) -> Result<Option<Vec<CodeLens>>> {
        let enabled = *self.pricing_enabled.lock().unwrap();
        if !enabled {
            return Ok(None);
        }
        let uri = &params.text_document.uri;

        // Never price inline on the request path: serve the last result
        // computed by the background pricing task (kicked off by
        // didOpen/didChange/updateMarket), which also requests a code lens
        // refresh once fresh data is available.
        if let Some(lenses) = self.lens_cache.lock().unwrap().get(uri) {
            return Ok(Some(lenses.clone()));
        }

        // No cached result yet: show a placeholder lens at the product
        // definition while the background task finishes.
        let placeholder = {
            let docs = self.documents.lock().unwrap();
            docs.get(uri).and_then(|state| {
                state.product.as_ref()?;
                let span_start = state.ast.as_ref().map(|a| a.span.start).unwrap_or(0);
                let pos = diagnostics::offset_to_position(&state.source, span_start);
                Some(vec![CodeLens {
                    range: Range {
                        start: pos,
                        end: pos,
                    },
                    command: Some(Command {
                        title: "Pricing...".to_string(),
                        command: String::new(),
                        arguments: None,
                    }),
                    data: None,
                }])
            })
        };
        Ok(placeholder)
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;
        let docs = self.documents.lock().unwrap();
        Ok(docs.get(uri).map(|state| {
            SemanticTokensResult::Tokens(SemanticTokens {
                result_id: None,
                data: semantic_tokens::semantic_tokens(state),
            })
        }))
    }
}

/// Dividend (or carry-equivalent) yield for the market snapshot shown in the
/// pricing panel. Only equity assets carry an explicit dividend yield.
fn asset_dividend_yield(asset: &openferric::dsl::market::AssetMarketData) -> f64 {
    match asset {
        openferric::dsl::market::AssetMarketData::Equity { dividend_yield, .. } => *dividend_yield,
        _ => 0.0,
    }
}

/// Compute pricing payload in a blocking context (called from spawned task).
fn compute_pricing_payload(
    product: &CompiledProduct,
    pricing_cfg: &PricingConfig,
    market_json: Option<&serde_json::Value>,
) -> PricingResultPayload {
    let underlying_names: Vec<String> =
        product.underlyings.iter().map(|u| u.name.clone()).collect();

    let market = match codelens::build_market(product.num_underlyings, market_json) {
        Ok(m) => m,
        Err(e) => {
            return PricingResultPayload {
                product_name: product.name.clone(),
                notional: product.notional,
                maturity: product.maturity,
                underlyings: underlying_names,
                price: f64::NAN,
                stderr: None,
                greeks: vec![],
                cross_greeks: vec![],
                payoff_profile: vec![],
                error: Some(format!("Failed to build market data: {e}")),
                market: None,
            };
        }
    };

    let engine = DslMonteCarloEngine::new(
        pricing_cfg.num_paths as usize,
        pricing_cfg.num_steps as usize,
        pricing_cfg.seed,
    );

    // Price.
    let (price, stderr, error) = match engine.price_multi_asset(product, &market) {
        Ok(result) => (result.price, result.stderr, None),
        Err(e) => (0.0, None, Some(format!("{e}"))),
    };

    // Extended greeks per underlying (delta, gamma, vega, rho, vanna, volga).
    let mut greeks = Vec::new();
    for i in 0..product.num_underlyings {
        let name = product
            .underlyings
            .get(i)
            .map(|u| u.name.clone())
            .unwrap_or_else(|| format!("Asset {i}"));
        if let Ok(g) = engine.extended_greeks_multi_asset(product, &market, i, price) {
            greeks.push(GreeksEntry {
                asset: name,
                delta: g.delta,
                gamma: g.gamma,
                vega: g.vega,
                theta: g.theta,
                rho: g.rho,
                vanna: g.vanna,
                volga: g.volga,
            });
        }
    }

    // Cross-greeks for each unique pair (i < j).
    let mut cross_greeks = Vec::new();
    for i in 0..product.num_underlyings {
        for j in (i + 1)..product.num_underlyings {
            let name_i = product
                .underlyings
                .get(i)
                .map(|u| u.name.clone())
                .unwrap_or_else(|| format!("Asset {i}"));
            let name_j = product
                .underlyings
                .get(j)
                .map(|u| u.name.clone())
                .unwrap_or_else(|| format!("Asset {j}"));
            if let Ok(cg) = engine.cross_greeks_multi_asset(product, &market, i, j, price) {
                cross_greeks.push(CrossGreeksEntry {
                    asset_i: name_i,
                    asset_j: name_j,
                    cross_gamma: cg.cross_gamma,
                    corr_sens: cg.corr_sens,
                });
            }
        }
    }

    // Payoff profile: 21 spot levels from 50% to 150%.
    let payoff_engine =
        DslMonteCarloEngine::new(10_000, pricing_cfg.num_steps as usize, pricing_cfg.seed);
    let mut payoff_profile = Vec::with_capacity(21);
    for i in 0..=20 {
        let pct = 50.0 + (i as f64) * 5.0;
        let scale = pct / 100.0;
        let mut bumped_market = market.clone();
        for asset in &mut bumped_market.assets {
            let base_val = asset.initial_value();
            let bump = base_val * (scale - 1.0);
            *asset = asset.with_spot_bump(bump);
        }
        let pv = payoff_engine
            .price_multi_asset(product, &bumped_market)
            .map(|r| r.price)
            .unwrap_or(0.0);
        payoff_profile.push(PayoffPoint { spot_pct: pct, pv });
    }

    // Build market snapshot for the UI.
    let market_snapshot = MarketSnapshot {
        rate: market.rate,
        assets: product
            .underlyings
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let a = &market.assets[i];
                AssetSnapshot {
                    name: u.name.clone(),
                    spot: a.initial_value(),
                    vol: a.vol(),
                    dividend_yield: asset_dividend_yield(a),
                }
            })
            .collect(),
        correlation: market.correlation.clone(),
    };

    PricingResultPayload {
        product_name: product.name.clone(),
        notional: product.notional,
        maturity: product.maturity,
        underlyings: underlying_names,
        price,
        stderr,
        greeks,
        cross_greeks,
        payoff_profile,
        error,
        market: Some(market_snapshot),
    }
}
