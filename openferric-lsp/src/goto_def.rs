use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::{offset_to_position, position_to_offset};
use crate::symbols::SymbolKind;

/// Go to the definition of the symbol at the cursor position.
pub fn goto_definition(state: &DocumentState, pos: Position) -> Option<Range> {
    let offset = position_to_offset(&state.source, pos);

    // Find the reference at cursor.
    let sym_ref = state.symbols.reference_at(offset)?;

    // Builtins have no source definition.
    if sym_ref.kind == SymbolKind::Builtin || sym_ref.kind == SymbolKind::BuiltinFn {
        return None;
    }

    // Zero span means no real definition (builtins).
    if sym_ref.def_span.start == 0 && sym_ref.def_span.end == 0 {
        return None;
    }

    Some(Range {
        start: offset_to_position(&state.source, sym_ref.def_span.start),
        end: offset_to_position(&state.source, sym_ref.def_span.end),
    })
}
