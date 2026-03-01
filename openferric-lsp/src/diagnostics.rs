use openferric::dsl::ast::ProductDef;
use openferric::dsl::error::DslError;
use openferric::dsl::ir::CompiledProduct;
use openferric::dsl::{compiler, lexer, parser};
use tower_lsp::lsp_types::*;

/// Parse and compile source, returning the AST, compiled product, and diagnostics.
pub fn parse_and_diagnose(
    source: &str,
) -> (Option<ProductDef>, Option<CompiledProduct>, Vec<Diagnostic>) {
    let tokens = match lexer::tokenize(source) {
        Ok(t) => t,
        Err(e) => return (None, None, vec![error_to_diagnostic(source, &e)]),
    };

    let ast = match parser::parse(tokens) {
        Ok(a) => a,
        Err(e) => return (None, None, vec![error_to_diagnostic(source, &e)]),
    };

    match compiler::compile(&ast) {
        Ok(product) => (Some(ast), Some(product), vec![]),
        Err(e) => (Some(ast), None, vec![error_to_diagnostic(source, &e)]),
    }
}

fn error_to_diagnostic(source: &str, error: &DslError) -> Diagnostic {
    let (message, range) = match error {
        DslError::LexError { message, span } => {
            (message.clone(), span_to_range(source, span.start, span.end))
        }
        DslError::ParseError { message, span } => {
            (message.clone(), span_to_range(source, span.start, span.end))
        }
        DslError::CompileError { message, span } => {
            let range = span
                .map(|s| span_to_range(source, s.start, s.end))
                .unwrap_or(Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 1),
                });
            (message.clone(), range)
        }
        DslError::EvalError(msg) => (
            msg.clone(),
            Range {
                start: Position::new(0, 0),
                end: Position::new(0, 1),
            },
        ),
    };

    Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("openferric".into()),
        message,
        ..Default::default()
    }
}

/// Convert byte offsets to LSP Position range by counting newlines.
fn span_to_range(source: &str, start: usize, end: usize) -> Range {
    Range {
        start: offset_to_position(source, start),
        end: offset_to_position(source, end),
    }
}

pub fn offset_to_position(source: &str, offset: usize) -> Position {
    let offset = offset.min(source.len());
    let before = &source[..offset];
    let line = before.matches('\n').count() as u32;
    let col = before.rfind('\n').map_or(offset, |nl| offset - nl - 1) as u32;
    Position::new(line, col)
}

pub fn position_to_offset(source: &str, pos: Position) -> usize {
    let mut line = 0u32;
    for (i, c) in source.char_indices() {
        if line == pos.line {
            let col_offset = pos.character as usize;
            return (i + col_offset).min(source.len());
        }
        if c == '\n' {
            line += 1;
        }
    }
    source.len()
}
