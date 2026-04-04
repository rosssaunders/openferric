use openferric::dsl::analysis;
use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::{offset_to_position, position_to_offset};

/// Provide hover information at the cursor position.
pub fn hover(state: &DocumentState, pos: Position) -> Option<Hover> {
    let offset = position_to_offset(&state.source, pos);
    let info = analysis::hover_info(&state.source, &state.symbols, offset)?;

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: info.markdown,
        }),
        range: Some(Range {
            start: offset_to_position(&state.source, info.start),
            end: offset_to_position(&state.source, info.end),
        }),
    })
}
