use openferric::dsl::lexer::{self, TokenKind};
use tower_lsp::lsp_types::*;

use crate::backend::DocumentState;
use crate::diagnostics::offset_to_position;
use crate::symbols::SymbolKind;

// Semantic token types — indices into LEGEND_TYPE.
const KEYWORD: u32 = 0;
const VARIABLE: u32 = 1;
const FUNCTION: u32 = 2;
const NUMBER: u32 = 3;
const STRING: u32 = 4;
const OPERATOR: u32 = 5;
const ENUM_MEMBER: u32 = 6;
#[allow(dead_code)]
const COMMENT: u32 = 7;

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
    let tokens = match lexer::tokenize(&state.source) {
        Ok(t) => t,
        Err(_) => return vec![],
    };

    let mut result = Vec::new();
    let mut prev_line = 0u32;
    let mut prev_start = 0u32;

    for token in &tokens {
        let token_type = match &token.kind {
            // Keywords
            TokenKind::Product
            | TokenKind::Notional
            | TokenKind::Maturity
            | TokenKind::Underlyings
            | TokenKind::State
            | TokenKind::Schedule
            | TokenKind::From
            | TokenKind::To
            | TokenKind::Let
            | TokenKind::If
            | TokenKind::Then
            | TokenKind::Else
            | TokenKind::Pay
            | TokenKind::Redeem
            | TokenKind::Set
            | TokenKind::Skip
            | TokenKind::Asset
            | TokenKind::Bool
            | TokenKind::Float
            | TokenKind::True
            | TokenKind::False => KEYWORD,

            // Logical operators as keywords
            TokenKind::And | TokenKind::Or | TokenKind::Not => OPERATOR,

            // Frequencies
            TokenKind::Monthly
            | TokenKind::Quarterly
            | TokenKind::SemiAnnual
            | TokenKind::Annual => ENUM_MEMBER,

            // Literals
            TokenKind::Number(_) => NUMBER,
            TokenKind::StringLit(_) => STRING,

            // Identifiers — resolve via symbol table
            TokenKind::Ident(name) => {
                if let Some(sym_ref) = state.symbols.reference_at(token.span.start) {
                    match sym_ref.kind {
                        SymbolKind::BuiltinFn => FUNCTION,
                        SymbolKind::Builtin => VARIABLE,
                        SymbolKind::Local => VARIABLE,
                        SymbolKind::StateVar => VARIABLE,
                        SymbolKind::Underlying => VARIABLE,
                    }
                } else {
                    // Check if it's a known function name
                    match name.as_str() {
                        "worst_of" | "best_of" | "performances" | "price" | "min" | "max"
                        | "abs" | "exp" | "log" => FUNCTION,
                        _ => VARIABLE,
                    }
                }
            }

            // Skip indentation and punctuation tokens
            TokenKind::Indent | TokenKind::Dedent => continue,
            TokenKind::LParen | TokenKind::RParen | TokenKind::Colon | TokenKind::Comma => continue,
            TokenKind::Eq
            | TokenKind::EqEq
            | TokenKind::Ne
            | TokenKind::Lt
            | TokenKind::Le
            | TokenKind::Gt
            | TokenKind::Ge
            | TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash => OPERATOR,
        };

        let pos = offset_to_position(&state.source, token.span.start);
        let length = (token.span.end - token.span.start) as u32;

        let delta_line = pos.line - prev_line;
        let delta_start = if delta_line == 0 {
            pos.character - prev_start
        } else {
            pos.character
        };

        result.push(SemanticToken {
            delta_line,
            delta_start,
            length,
            token_type,
            token_modifiers_bitset: 0,
        });

        prev_line = pos.line;
        prev_start = pos.character;
    }

    result
}
