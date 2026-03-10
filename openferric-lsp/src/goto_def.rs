use openferric::dsl::analysis;
use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::{offset_to_position, position_to_offset};

/// Go to the definition of the symbol at the cursor position.
pub fn goto_definition(state: &DocumentState, pos: Position) -> Option<Range> {
    let offset = position_to_offset(&state.source, pos);
    let span = analysis::goto_definition(&state.source, &state.symbols, offset)?;

    Some(Range {
        start: offset_to_position(&state.source, span.start),
        end: offset_to_position(&state.source, span.end),
    })
}
