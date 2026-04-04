use openferric::dsl::analysis;
use openferric::dsl::ast::ProductDef;
use openferric::dsl::ir::CompiledProduct;
use tower_lsp::lsp_types::*;

/// Parse and compile source, returning the AST, compiled product, and LSP diagnostics.
pub fn parse_and_diagnose(
    source: &str,
) -> (Option<ProductDef>, Option<CompiledProduct>, Vec<Diagnostic>) {
    let (ast, product, diags) = analysis::parse_and_diagnose(source);

    let lsp_diags = diags
        .into_iter()
        .map(|d| {
            let range = Range {
                start: offset_to_position(source, d.start),
                end: offset_to_position(source, d.end),
            };
            Diagnostic {
                range,
                severity: Some(match d.severity {
                    analysis::DiagnosticSeverity::Error => DiagnosticSeverity::ERROR,
                    analysis::DiagnosticSeverity::Warning => DiagnosticSeverity::WARNING,
                }),
                source: Some("openferric".into()),
                message: d.message,
                ..Default::default()
            }
        })
        .collect();

    (ast, product, lsp_diags)
}

pub fn offset_to_position(source: &str, offset: usize) -> Position {
    let (line, col) = analysis::offset_to_line_col(source, offset);
    Position::new(line, col)
}

pub fn position_to_offset(source: &str, pos: Position) -> usize {
    analysis::line_col_to_offset(source, pos.line, pos.character)
}
