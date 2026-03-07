use openferric::dsl::analysis::{self, CompletionCandidateKind};
use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::position_to_offset;

/// Provide context-aware completions.
pub fn completions(state: &DocumentState, pos: Position) -> Vec<CompletionItem> {
    let offset = position_to_offset(&state.source, pos);
    let candidates = analysis::completions(&state.source, &state.symbols, offset);

    candidates
        .into_iter()
        .map(|c| CompletionItem {
            label: c.label,
            kind: Some(match c.kind {
                CompletionCandidateKind::Keyword => CompletionItemKind::KEYWORD,
                CompletionCandidateKind::Variable => CompletionItemKind::VARIABLE,
                CompletionCandidateKind::Function => CompletionItemKind::FUNCTION,
                CompletionCandidateKind::EnumMember => CompletionItemKind::ENUM_MEMBER,
                CompletionCandidateKind::Constant => CompletionItemKind::CONSTANT,
            }),
            detail: c.detail,
            documentation: c.documentation.map(Documentation::String),
            ..Default::default()
        })
        .collect()
}
