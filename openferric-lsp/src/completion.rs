use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::position_to_offset;
use crate::symbols::SymbolKind;

/// Provide context-aware completions.
pub fn completions(state: &DocumentState, pos: Position) -> Vec<CompletionItem> {
    let offset = position_to_offset(&state.source, pos);
    let context = determine_context(&state.source, offset);

    match context {
        Context::TopLevel => top_level_completions(),
        Context::AfterSchedule => frequency_completions(),
        Context::StatementPosition => statement_completions(),
        Context::Expression => expression_completions(state),
        Context::AfterSet => state_var_completions(state),
        Context::TypePosition => type_completions(),
    }
}

#[derive(Debug)]
enum Context {
    TopLevel,
    AfterSchedule,
    StatementPosition,
    Expression,
    AfterSet,
    TypePosition,
}

fn determine_context(source: &str, offset: usize) -> Context {
    let before = &source[..offset.min(source.len())];

    // Get the current line up to the cursor.
    let line_start = before.rfind('\n').map_or(0, |nl| nl + 1);
    let line = before[line_start..].trim();

    // After "set " → state variable names.
    if line.starts_with("set ") && !line.contains('=') {
        return Context::AfterSet;
    }

    // After "schedule" keyword on the same line → frequency.
    if line.starts_with("schedule") && !line.contains("from") {
        return Context::AfterSchedule;
    }

    // After ":" in state declaration → type position.
    if line.contains(':') && !line.contains('=') {
        // Check if we're inside a state block.
        if is_in_state_block(before) {
            return Context::TypePosition;
        }
    }

    // Inside a schedule body (indented inside schedule) → statement or expression.
    if is_in_schedule_body(before) {
        if line.is_empty() || line_is_statement_start(line) {
            return Context::StatementPosition;
        }
        return Context::Expression;
    }

    // Top-level (inside product body).
    Context::TopLevel
}

fn is_in_state_block(before: &str) -> bool {
    // Check if we see "state" as a block keyword above.
    for line in before.lines().rev() {
        let trimmed = line.trim();
        if trimmed == "state" {
            return true;
        }
        if trimmed.starts_with("schedule")
            || trimmed.starts_with("underlyings")
            || trimmed.starts_with("product")
        {
            return false;
        }
    }
    false
}

fn is_in_schedule_body(before: &str) -> bool {
    for line in before.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with("schedule") {
            return true;
        }
        if trimmed.starts_with("product") || trimmed.starts_with("underlyings") || trimmed == "state" {
            return false;
        }
    }
    false
}

fn line_is_statement_start(line: &str) -> bool {
    let first_word = line.split_whitespace().next().unwrap_or("");
    matches!(
        first_word,
        "" | "let" | "if" | "pay" | "redeem" | "set" | "skip" | "else"
    )
}

fn top_level_completions() -> Vec<CompletionItem> {
    vec![
        keyword_item("notional:", "Product face value"),
        keyword_item("maturity:", "Product maturity in year fractions"),
        keyword_item("underlyings", "Declare underlying assets"),
        keyword_item("state", "Declare mutable state variables"),
        keyword_item("schedule", "Define observation schedule"),
    ]
}

fn frequency_completions() -> Vec<CompletionItem> {
    vec![
        enum_item("monthly", "Every month (period = 1/12)"),
        enum_item("quarterly", "Every quarter (period = 0.25)"),
        enum_item("semi_annual", "Every 6 months (period = 0.5)"),
        enum_item("annual", "Every year (period = 1.0)"),
    ]
}

fn statement_completions() -> Vec<CompletionItem> {
    vec![
        keyword_item("let", "Bind a local variable"),
        keyword_item("if", "Conditional branch"),
        keyword_item("pay", "Record a cashflow payment"),
        keyword_item("redeem", "Final payment and terminate"),
        keyword_item("set", "Update a state variable"),
        keyword_item("skip", "Skip this observation date"),
    ]
}

fn expression_completions(state: &DocumentState) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Declared symbols (locals, state vars, underlyings).
    for sym in &state.symbols.declarations {
        match sym.kind {
            SymbolKind::Local | SymbolKind::StateVar | SymbolKind::Underlying => {
                items.push(CompletionItem {
                    label: sym.name.clone(),
                    kind: Some(CompletionItemKind::VARIABLE),
                    detail: Some(format!("{}: {}", sym.name, sym.type_hint)),
                    documentation: Some(Documentation::String(sym.doc.to_string())),
                    ..Default::default()
                });
            }
            SymbolKind::Builtin => {
                items.push(CompletionItem {
                    label: sym.name.clone(),
                    kind: Some(CompletionItemKind::CONSTANT),
                    detail: Some(format!("{}: {}", sym.name, sym.type_hint)),
                    documentation: Some(Documentation::String(sym.doc.to_string())),
                    ..Default::default()
                });
            }
            SymbolKind::BuiltinFn => {
                items.push(CompletionItem {
                    label: sym.name.clone(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    detail: Some(sym.type_hint.to_string()),
                    documentation: Some(Documentation::String(sym.doc.to_string())),
                    ..Default::default()
                });
            }
        }
    }

    items
}

fn state_var_completions(state: &DocumentState) -> Vec<CompletionItem> {
    state
        .symbols
        .declarations
        .iter()
        .filter(|s| s.kind == SymbolKind::StateVar)
        .map(|s| CompletionItem {
            label: s.name.clone(),
            kind: Some(CompletionItemKind::VARIABLE),
            detail: Some(format!("{}: {}", s.name, s.type_hint)),
            ..Default::default()
        })
        .collect()
}

fn type_completions() -> Vec<CompletionItem> {
    vec![
        keyword_item("bool", "Boolean type"),
        keyword_item("float", "Floating-point number type"),
    ]
}

fn keyword_item(label: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: label.into(),
        kind: Some(CompletionItemKind::KEYWORD),
        detail: Some(detail.into()),
        ..Default::default()
    }
}

fn enum_item(label: &str, detail: &str) -> CompletionItem {
    CompletionItem {
        label: label.into(),
        kind: Some(CompletionItemKind::ENUM_MEMBER),
        detail: Some(detail.into()),
        ..Default::default()
    }
}
