use openferric::dsl::ast::*;
use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::offset_to_position;

/// Build document symbols for the outline panel.
#[allow(deprecated)] // SymbolInformation::deprecated field
pub fn document_symbols(state: &DocumentState, uri: &Url) -> Vec<SymbolInformation> {
    let ast = match &state.ast {
        Some(a) => a,
        None => return vec![],
    };

    let mut symbols = Vec::new();

    // Product itself.
    symbols.push(SymbolInformation {
        name: format!("product \"{}\"", ast.name),
        kind: tower_lsp::lsp_types::SymbolKind::CLASS,
        location: Location {
            uri: uri.clone(), // placeholder, overridden by client
            range: span_to_range(&state.source, ast.span.start, ast.span.end),
        },
        tags: None,
        deprecated: None,
        container_name: None,
    });

    for item in &ast.body {
        match item {
            ProductItem::Notional(val, span) => {
                symbols.push(SymbolInformation {
                    name: format!("notional: {val}"),
                    kind: tower_lsp::lsp_types::SymbolKind::PROPERTY,
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(&state.source, span.start, span.end),
                    },
                    tags: None,
                    deprecated: None,
                    container_name: Some(ast.name.clone()),
                });
            }
            ProductItem::Maturity(val, span) => {
                symbols.push(SymbolInformation {
                    name: format!("maturity: {val}"),
                    kind: tower_lsp::lsp_types::SymbolKind::PROPERTY,
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(&state.source, span.start, span.end),
                    },
                    tags: None,
                    deprecated: None,
                    container_name: Some(ast.name.clone()),
                });
            }
            ProductItem::Underlyings(decls, span) => {
                symbols.push(SymbolInformation {
                    name: "underlyings".into(),
                    kind: tower_lsp::lsp_types::SymbolKind::NAMESPACE,
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(&state.source, span.start, span.end),
                    },
                    tags: None,
                    deprecated: None,
                    container_name: Some(ast.name.clone()),
                });
                for decl in decls {
                    symbols.push(SymbolInformation {
                        name: format!("{} = asset({})", decl.name, decl.asset_index),
                        kind: tower_lsp::lsp_types::SymbolKind::VARIABLE,
                        location: Location {
                            uri: uri.clone(),
                            range: span_to_range(&state.source, decl.span.start, decl.span.end),
                        },
                        tags: None,
                        deprecated: None,
                        container_name: Some("underlyings".into()),
                    });
                }
            }
            ProductItem::State(decls, span) => {
                symbols.push(SymbolInformation {
                    name: "state".into(),
                    kind: tower_lsp::lsp_types::SymbolKind::NAMESPACE,
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(&state.source, span.start, span.end),
                    },
                    tags: None,
                    deprecated: None,
                    container_name: Some(ast.name.clone()),
                });
                for decl in decls {
                    symbols.push(SymbolInformation {
                        name: format!("{}: {}", decl.name, decl.type_name),
                        kind: tower_lsp::lsp_types::SymbolKind::VARIABLE,
                        location: Location {
                            uri: uri.clone(),
                            range: span_to_range(&state.source, decl.span.start, decl.span.end),
                        },
                        tags: None,
                        deprecated: None,
                        container_name: Some("state".into()),
                    });
                }
            }
            ProductItem::Schedule(sched) => {
                let freq = match sched.frequency {
                    ScheduleFreq::Monthly => "monthly",
                    ScheduleFreq::Quarterly => "quarterly",
                    ScheduleFreq::SemiAnnual => "semi_annual",
                    ScheduleFreq::Annual => "annual",
                    ScheduleFreq::Custom(p) => {
                        // Use a static str for common custom values, else fallback.
                        if (p - 1.0 / 12.0).abs() < 1e-10 {
                            "monthly"
                        } else {
                            "custom"
                        }
                    }
                };
                symbols.push(SymbolInformation {
                    name: format!("schedule {freq} {}..{}", sched.start, sched.end),
                    kind: tower_lsp::lsp_types::SymbolKind::EVENT,
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(&state.source, sched.span.start, sched.span.end),
                    },
                    tags: None,
                    deprecated: None,
                    container_name: Some(ast.name.clone()),
                });
            }
        }
    }

    symbols
}

fn span_to_range(source: &str, start: usize, end: usize) -> Range {
    Range {
        start: offset_to_position(source, start),
        end: offset_to_position(source, end),
    }
}
