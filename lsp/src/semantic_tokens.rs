use openferric::dsl::analysis;
use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;

const LEGEND_TYPE: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::NUMBER,
    SemanticTokenType::STRING,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::ENUM_MEMBER,
    SemanticTokenType::COMMENT,
];

pub fn capabilities() -> SemanticTokensOptions {
    SemanticTokensOptions {
        legend: SemanticTokensLegend {
            token_types: LEGEND_TYPE.to_vec(),
            token_modifiers: vec![],
        },
        full: Some(SemanticTokensFullOptions::Bool(true)),
        range: None,
        ..Default::default()
    }
}

pub fn semantic_tokens(state: &DocumentState) -> Vec<SemanticToken> {
    let data = analysis::semantic_token_data(&state.source, &state.symbols);

    data.into_iter()
        .map(|d| SemanticToken {
            delta_line: d.delta_line,
            delta_start: d.delta_start,
            length: d.length,
            token_type: d.token_type,
            token_modifiers_bitset: d.modifiers,
        })
        .collect()
}
