//! Indentation-sensitive lexer for the DSL.
//!
//! Processes source line-by-line, emitting `Indent` and `Dedent` tokens
//! based on leading whitespace (like Python/F#). Braces are not used.

use crate::dsl::error::{DslError, Span};

/// Token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Token types.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Number(f64),
    StringLit(String),
    True,
    False,

    // Identifiers
    Ident(String),

    // Keywords
    Product,
    Notional,
    Maturity,
    Underlyings,
    State,
    Schedule,
    From,
    To,
    Let,
    If,
    Then,
    Else,
    Pay,
    Redeem,
    Set,
    Skip,
    And,
    Or,
    Not,
    Asset,

    // Frequencies
    Monthly,
    Quarterly,
    SemiAnnual,
    Annual,

    // Type keywords
    Bool,
    Float,

    // Indentation
    Indent,
    Dedent,

    // Punctuation
    LParen,
    RParen,
    Colon,
    Comma,
    Eq,   // =
    EqEq, // ==
    Ne,   // !=
    Lt,   // <
    Le,   // <=
    Gt,   // >
    Ge,   // >=
    Plus,
    Minus,
    Star,
    Slash,
}

/// Tokenize DSL source text into a vector of tokens.
///
/// Uses indentation-based block structure: emits `Indent` when a line is
/// more indented than the previous, and `Dedent` when less indented.
/// Blank lines and comment-only lines are skipped.
pub fn tokenize(source: &str) -> Result<Vec<Token>, DslError> {
    let mut tokens = Vec::new();
    let mut indent_stack: Vec<usize> = vec![0];
    let mut global_pos: usize = 0;

    for line in source.split('\n') {
        let line_start = global_pos;
        let line_clean = line.trim_end_matches('\r');

        // Count leading spaces.
        let indent = line_clean.bytes().take_while(|&b| b == b' ').count();
        let content = line_clean[indent..].trim_end();

        // Skip blank and comment-only lines.
        if content.is_empty() || content.starts_with("//") {
            global_pos += line.len() + 1;
            continue;
        }

        // Emit INDENT / DEDENT based on indentation change.
        let current = *indent_stack.last().unwrap();
        if indent > current {
            indent_stack.push(indent);
            tokens.push(Token {
                kind: TokenKind::Indent,
                span: Span::new(line_start, line_start + indent),
            });
        } else if indent < current {
            while *indent_stack.last().unwrap() > indent {
                indent_stack.pop();
                tokens.push(Token {
                    kind: TokenKind::Dedent,
                    span: Span::new(line_start, line_start + indent),
                });
            }
            if *indent_stack.last().unwrap() != indent {
                return Err(DslError::LexError {
                    message: format!(
                        "inconsistent indentation: expected {} spaces, got {indent}",
                        indent_stack.last().unwrap()
                    ),
                    span: Span::new(line_start, line_start + indent),
                });
            }
        }

        // Tokenize line content.
        let content_offset = line_start + indent;
        tokenize_content(content, content_offset, &mut tokens)?;

        global_pos += line.len() + 1;
    }

    // Emit remaining DEDENTs at end of input.
    while indent_stack.len() > 1 {
        indent_stack.pop();
        tokens.push(Token {
            kind: TokenKind::Dedent,
            span: Span::new(global_pos, global_pos),
        });
    }

    Ok(tokens)
}

/// Tokenize the content of a single line (leading whitespace already stripped).
fn tokenize_content(
    content: &str,
    offset: usize,
    tokens: &mut Vec<Token>,
) -> Result<(), DslError> {
    let bytes = content.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        // Skip spaces within line.
        while pos < bytes.len() && bytes[pos] == b' ' {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }

        // Inline comment — stop processing this line.
        if pos + 1 < bytes.len() && bytes[pos] == b'/' && bytes[pos + 1] == b'/' {
            break;
        }

        let start = offset + pos;
        let ch = bytes[pos];

        // String literal.
        if ch == b'"' {
            let (s, end) = lex_string(content, pos, offset)?;
            tokens.push(Token {
                kind: TokenKind::StringLit(s),
                span: Span::new(start, end),
            });
            pos = end - offset;
            continue;
        }

        // Number literal.
        if ch.is_ascii_digit()
            || (ch == b'.' && pos + 1 < bytes.len() && bytes[pos + 1].is_ascii_digit())
        {
            let (num, end) = lex_number(content, pos, offset)?;
            tokens.push(Token {
                kind: TokenKind::Number(num),
                span: Span::new(start, end),
            });
            pos = end - offset;
            continue;
        }

        // Identifier or keyword.
        if ch.is_ascii_alphabetic() || ch == b'_' {
            let local_end = lex_ident_end(content, pos);
            let word = &content[pos..local_end];
            let kind = match word {
                "product" => TokenKind::Product,
                "notional" => TokenKind::Notional,
                "maturity" => TokenKind::Maturity,
                "underlyings" => TokenKind::Underlyings,
                "state" => TokenKind::State,
                "schedule" => TokenKind::Schedule,
                "from" => TokenKind::From,
                "to" => TokenKind::To,
                "let" => TokenKind::Let,
                "if" => TokenKind::If,
                "then" => TokenKind::Then,
                "else" => TokenKind::Else,
                "pay" => TokenKind::Pay,
                "redeem" => TokenKind::Redeem,
                "set" => TokenKind::Set,
                "skip" => TokenKind::Skip,
                "and" => TokenKind::And,
                "or" => TokenKind::Or,
                "not" => TokenKind::Not,
                "asset" => TokenKind::Asset,
                "true" => TokenKind::True,
                "false" => TokenKind::False,
                "monthly" => TokenKind::Monthly,
                "quarterly" => TokenKind::Quarterly,
                "semi_annual" => TokenKind::SemiAnnual,
                "annual" => TokenKind::Annual,
                "bool" => TokenKind::Bool,
                "float" => TokenKind::Float,
                _ => TokenKind::Ident(word.to_string()),
            };
            let global_end = offset + local_end;
            tokens.push(Token {
                kind,
                span: Span::new(start, global_end),
            });
            pos = local_end;
            continue;
        }

        // Two-character operators.
        if pos + 1 < bytes.len() {
            let two = &content[pos..pos + 2];
            let kind = match two {
                "==" => Some(TokenKind::EqEq),
                "!=" => Some(TokenKind::Ne),
                "<=" => Some(TokenKind::Le),
                ">=" => Some(TokenKind::Ge),
                _ => None,
            };
            if let Some(kind) = kind {
                tokens.push(Token {
                    kind,
                    span: Span::new(start, offset + pos + 2),
                });
                pos += 2;
                continue;
            }
        }

        // Single-character operators / punctuation.
        let kind = match ch {
            b'(' => Some(TokenKind::LParen),
            b')' => Some(TokenKind::RParen),
            b':' => Some(TokenKind::Colon),
            b',' => Some(TokenKind::Comma),
            b'=' => Some(TokenKind::Eq),
            b'<' => Some(TokenKind::Lt),
            b'>' => Some(TokenKind::Gt),
            b'+' => Some(TokenKind::Plus),
            b'-' => Some(TokenKind::Minus),
            b'*' => Some(TokenKind::Star),
            b'/' => Some(TokenKind::Slash),
            _ => None,
        };

        if let Some(kind) = kind {
            tokens.push(Token {
                kind,
                span: Span::new(start, offset + pos + 1),
            });
            pos += 1;
        } else {
            return Err(DslError::LexError {
                message: format!(
                    "unexpected character '{}'",
                    content[pos..].chars().next().unwrap()
                ),
                span: Span::new(start, start + 1),
            });
        }
    }

    Ok(())
}

fn lex_string(
    content: &str,
    start: usize,
    offset: usize,
) -> Result<(String, usize), DslError> {
    let bytes = content.as_bytes();
    let mut pos = start + 1;
    let mut s = String::new();
    while pos < bytes.len() {
        if bytes[pos] == b'"' {
            return Ok((s, offset + pos + 1));
        }
        if bytes[pos] == b'\\' && pos + 1 < bytes.len() {
            pos += 1;
            match bytes[pos] {
                b'n' => s.push('\n'),
                b't' => s.push('\t'),
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                _ => s.push(bytes[pos] as char),
            }
        } else {
            s.push(bytes[pos] as char);
        }
        pos += 1;
    }
    Err(DslError::LexError {
        message: "unterminated string literal".to_string(),
        span: Span::new(offset + start, offset + pos),
    })
}

fn lex_number(
    content: &str,
    start: usize,
    offset: usize,
) -> Result<(f64, usize), DslError> {
    let bytes = content.as_bytes();
    let mut pos = start;
    let mut num_str = String::new();

    while pos < bytes.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b'_') {
        if bytes[pos] != b'_' {
            num_str.push(bytes[pos] as char);
        }
        pos += 1;
    }

    if pos < bytes.len() && bytes[pos] == b'.' {
        num_str.push('.');
        pos += 1;
        while pos < bytes.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b'_') {
            if bytes[pos] != b'_' {
                num_str.push(bytes[pos] as char);
            }
            pos += 1;
        }
    }

    if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
        num_str.push('e');
        pos += 1;
        if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
            num_str.push(bytes[pos] as char);
            pos += 1;
        }
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            num_str.push(bytes[pos] as char);
            pos += 1;
        }
    }

    let global_end = offset + pos;
    num_str.parse::<f64>().map(|n| (n, global_end)).map_err(|_| {
        DslError::LexError {
            message: format!("invalid number literal '{num_str}'"),
            span: Span::new(offset + start, global_end),
        }
    })
}

fn lex_ident_end(content: &str, start: usize) -> usize {
    let bytes = content.as_bytes();
    let mut pos = start;
    while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
        pos += 1;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_product_header_with_indent() {
        let source = "product \"Test\"\n    notional: 1_000_000\n";
        let tokens = tokenize(source).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Product);
        assert_eq!(tokens[1].kind, TokenKind::StringLit("Test".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Indent);
        assert_eq!(tokens[3].kind, TokenKind::Notional);
        assert_eq!(tokens[4].kind, TokenKind::Colon);
        assert_eq!(tokens[5].kind, TokenKind::Number(1_000_000.0));
        assert_eq!(tokens[6].kind, TokenKind::Dedent);
    }

    #[test]
    fn tokenize_operators() {
        let tokens = tokenize("a <= 0.60 and not b >= 1.0").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Ident("a".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Le);
        assert_eq!(tokens[2].kind, TokenKind::Number(0.60));
        assert_eq!(tokens[3].kind, TokenKind::And);
        assert_eq!(tokens[4].kind, TokenKind::Not);
        assert_eq!(tokens[5].kind, TokenKind::Ident("b".to_string()));
        assert_eq!(tokens[6].kind, TokenKind::Ge);
        assert_eq!(tokens[7].kind, TokenKind::Number(1.0));
    }

    #[test]
    fn tokenize_comments_are_skipped() {
        let tokens = tokenize("a // this is a comment\nb").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].kind, TokenKind::Ident("a".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Ident("b".to_string()));
    }

    #[test]
    fn tokenize_schedule_keywords() {
        let tokens = tokenize("schedule quarterly from 0.25 to 1.5").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Schedule);
        assert_eq!(tokens[1].kind, TokenKind::Quarterly);
        assert_eq!(tokens[2].kind, TokenKind::From);
        assert_eq!(tokens[3].kind, TokenKind::Number(0.25));
        assert_eq!(tokens[4].kind, TokenKind::To);
        assert_eq!(tokens[5].kind, TokenKind::Number(1.5));
    }

    #[test]
    fn indent_dedent_nesting() {
        let source = "a\n    b\n        c\n    d\ne\n";
        let tokens = tokenize(source).unwrap();
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &TokenKind::Ident("a".to_string()),
                &TokenKind::Indent,
                &TokenKind::Ident("b".to_string()),
                &TokenKind::Indent,
                &TokenKind::Ident("c".to_string()),
                &TokenKind::Dedent,
                &TokenKind::Ident("d".to_string()),
                &TokenKind::Dedent,
                &TokenKind::Ident("e".to_string()),
            ]
        );
    }

    #[test]
    fn blank_lines_are_ignored() {
        let source = "a\n\n    b\n\n    c\n";
        let tokens = tokenize(source).unwrap();
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &TokenKind::Ident("a".to_string()),
                &TokenKind::Indent,
                &TokenKind::Ident("b".to_string()),
                &TokenKind::Ident("c".to_string()),
                &TokenKind::Dedent,
            ]
        );
    }

    #[test]
    fn then_keyword_is_recognized() {
        let tokens = tokenize("if x then").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::If);
        assert_eq!(tokens[1].kind, TokenKind::Ident("x".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Then);
    }
}
