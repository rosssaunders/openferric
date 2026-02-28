use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::{offset_to_position, position_to_offset};
use crate::symbols::SymbolKind;

/// Provide hover information at the cursor position.
pub fn hover(state: &DocumentState, pos: Position) -> Option<Hover> {
    let offset = position_to_offset(&state.source, pos);

    // Check if cursor is on a reference.
    if let Some(sym_ref) = state.symbols.reference_at(offset) {
        let markdown = format_symbol_hover(
            &sym_ref.name,
            sym_ref.kind,
            sym_ref.type_hint,
            sym_ref.doc,
        );
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: Some(span_to_range(&state.source, sym_ref.span.start, sym_ref.span.end)),
        });
    }

    // Check if cursor is on a declaration.
    if let Some(sym) = state.symbols.declaration_at(offset) {
        let markdown = format_symbol_hover(&sym.name, sym.kind, sym.type_hint, sym.doc);
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: Some(span_to_range(&state.source, sym.def_span.start, sym.def_span.end)),
        });
    }

    // Check if cursor is on a keyword.
    keyword_hover(state, offset)
}

fn format_symbol_hover(name: &str, kind: SymbolKind, type_hint: &str, doc: &str) -> String {
    let kind_label = match kind {
        SymbolKind::Underlying => "underlying equity",
        SymbolKind::StateVar => "state variable (mutable across dates)",
        SymbolKind::Local => "local variable",
        SymbolKind::Builtin => "built-in",
        SymbolKind::BuiltinFn => "built-in function",
    };

    match kind {
        SymbolKind::BuiltinFn => {
            format!("`{type_hint}` \u{2014} {doc}")
        }
        _ => {
            format!("`{name}: {type_hint}` \u{2014} {kind_label}")
        }
    }
}

fn keyword_hover(state: &DocumentState, offset: usize) -> Option<Hover> {
    let word = word_at_offset(&state.source, offset)?;

    let doc = match word.text {
        "product" => "Define a structured product",
        "notional" => "Product face value",
        "maturity" => "Product maturity in year fractions",
        "underlyings" => "Declare underlying assets for the product",
        "state" => "Declare mutable state variables that persist across observation dates",
        "schedule" => "Define an observation schedule with frequency and date range",
        "let" => "Bind a local variable within the current observation",
        "if" => "Conditional branch",
        "then" => "Begin the body of a conditional branch",
        "else" => "Alternative branch of a conditional",
        "pay" => "Record a cashflow payment at the current observation date",
        "redeem" => "Record final payment and terminate the product",
        "set" => "Update a state variable's value",
        "skip" => "Skip the current observation date without payment",
        "and" => "Logical AND operator",
        "or" => "Logical OR operator",
        "not" => "Logical NOT operator",
        "from" => "Schedule start date",
        "to" => "Schedule end date",
        "monthly" => "Schedule frequency: every month (period = 1/12)",
        "quarterly" => "Schedule frequency: every quarter (period = 0.25)",
        "semi_annual" => "Schedule frequency: every 6 months (period = 0.5)",
        "annual" => "Schedule frequency: every year (period = 1.0)",
        "true" | "false" => "Boolean literal",
        "bool" => "Boolean type",
        "float" => "Floating-point number type",
        "asset" => "Reference an underlying by index: asset(N)",
        _ => return None,
    };

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("`{0}` \u{2014} {doc}", word.text),
        }),
        range: Some(span_to_range(&state.source, word.start, word.end)),
    })
}

struct WordAt<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

fn word_at_offset(source: &str, offset: usize) -> Option<WordAt<'_>> {
    if offset > source.len() {
        return None;
    }
    let bytes = source.as_bytes();
    let mut start = offset;
    while start > 0 && is_ident_char(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = offset;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    if start == end {
        return None;
    }
    Some(WordAt {
        text: &source[start..end],
        start,
        end,
    })
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn span_to_range(source: &str, start: usize, end: usize) -> Range {
    Range {
        start: offset_to_position(source, start),
        end: offset_to_position(source, end),
    }
}
