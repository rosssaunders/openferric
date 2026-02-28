use std::collections::HashMap;
use std::sync::Mutex;

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
use crate::notification::{self, GreeksEntry, PayoffPoint, PricingResultPayload};
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
}

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

    /// Spawn a background task to compute pricing and send a notification.
    fn send_pricing_notification(&self) {
        let enabled = *self.pricing_enabled.lock().unwrap();
        if !enabled {
            return;
        }

        // Extract compiled product data under the lock.
        let product_data = {
            let docs = self.documents.lock().unwrap();
            docs.values()
                .find_map(|state| state.product.clone())
        };
        let Some(product) = product_data else { return };

        let pricing_cfg = self.pricing_config.lock().unwrap().clone();
        let market_cfg = self.market_config.lock().unwrap().clone();
        let client = self.client.clone();

        tokio::spawn(async move {
            let payload = compute_pricing_payload(&product, &pricing_cfg, market_cfg.as_ref());
            client
                .send_notification::<notification::PricingNotification>(payload)
                .await;
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
                    cfg.num_paths = v as u32;
                }
                if let Some(v) = pricing.get("numSteps").and_then(|v| v.as_u64()) {
                    cfg.num_steps = v as u32;
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
                    trigger_characters: Some(vec![
                        " ".into(),
                        ".".into(),
                        "(".into(),
                    ]),
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
        self.client
            .publish_diagnostics(uri, diags, None)
            .await;
        self.send_pricing_notification();
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        if let Some(change) = params.content_changes.into_iter().last() {
            let diags = self.update_document(&uri, change.text);
            self.client
                .publish_diagnostics(uri, diags, None)
                .await;
            self.send_pricing_notification();
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.lock().unwrap().remove(&uri);
        self.client
            .publish_diagnostics(uri, vec![], None)
            .await;
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
        // Extract data under the lock, then release before pricing.
        let code_lens_input = {
            let docs = self.documents.lock().unwrap();
            docs.get(uri).and_then(|state| {
                let product = state.product.clone()?;
                let product_span_start = state.ast.as_ref().map(|a| a.span.start).unwrap_or(0);
                let source = state.source.clone();
                Some((product, product_span_start, source))
            })
        };
        let Some((product, product_span_start, source)) = code_lens_input else {
            return Ok(None);
        };
        let pricing_cfg = self.pricing_config.lock().unwrap().clone();
        let market_cfg = self.market_config.lock().unwrap().clone();
        Ok(Some(codelens::code_lenses(
            &product,
            product_span_start,
            &source,
            &pricing_cfg,
            market_cfg.as_ref(),
        )))
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

/// Compute pricing payload in a blocking context (called from spawned task).
fn compute_pricing_payload(
    product: &CompiledProduct,
    pricing_cfg: &PricingConfig,
    market_json: Option<&serde_json::Value>,
) -> PricingResultPayload {
    let underlying_names: Vec<String> = product
        .underlyings
        .iter()
        .map(|u| u.name.clone())
        .collect();

    let market = match codelens::build_market(product.num_underlyings, market_json) {
        Some(m) => m,
        None => {
            return PricingResultPayload {
                product_name: product.name.clone(),
                notional: product.notional,
                maturity: product.maturity,
                underlyings: underlying_names,
                price: 0.0,
                stderr: None,
                greeks: vec![],
                payoff_profile: vec![],
                error: Some("Failed to build market data".into()),
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

    // Greeks per underlying.
    let mut greeks = Vec::new();
    for i in 0..product.num_underlyings {
        let name = product
            .underlyings
            .get(i)
            .map(|u| u.name.clone())
            .unwrap_or_else(|| format!("Asset {i}"));
        if let Ok(g) = engine.greeks_multi_asset(product, &market, i) {
            greeks.push(GreeksEntry {
                asset: name,
                delta: g.delta,
                gamma: g.gamma,
                vega: g.vega,
                theta: g.theta,
                rho: g.rho,
            });
        }
    }

    // Payoff profile: 21 spot levels from 50% to 150%.
    let payoff_engine = DslMonteCarloEngine::new(10_000, pricing_cfg.num_steps as usize, pricing_cfg.seed);
    let mut payoff_profile = Vec::with_capacity(21);
    for i in 0..=20 {
        let pct = 50.0 + (i as f64) * 5.0;
        let scale = pct / 100.0;
        let mut bumped_market = market.clone();
        for asset in &mut bumped_market.assets {
            asset.spot *= scale;
        }
        let pv = payoff_engine
            .price_multi_asset(product, &bumped_market)
            .map(|r| r.price)
            .unwrap_or(0.0);
        payoff_profile.push(PayoffPoint { spot_pct: pct, pv });
    }

    PricingResultPayload {
        product_name: product.name.clone(),
        notional: product.notional,
        maturity: product.maturity,
        underlyings: underlying_names,
        price,
        stderr,
        greeks,
        payoff_profile,
        error,
    }
}
