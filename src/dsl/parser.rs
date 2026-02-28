//! Parser for the indentation-based DSL.
//!
//! Transforms a token stream (with Indent/Dedent tokens from the lexer)
//! into an AST. Uses `then` for if-statement bodies instead of braces.

use crate::dsl::ast::*;
use crate::dsl::error::{DslError, Span};
use crate::dsl::lexer::{Token, TokenKind};

/// Parser state wrapping a token stream.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_kind(&self) -> Option<&TokenKind> {
        self.peek().map(|t| &t.kind)
    }

    fn advance(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<&Token, DslError> {
        match self.peek() {
            Some(tok) if &tok.kind == expected => {
                self.pos += 1;
                Ok(&self.tokens[self.pos - 1])
            }
            Some(tok) => Err(DslError::ParseError {
                message: format!("expected {expected:?}, got {:?}", tok.kind),
                span: tok.span,
            }),
            None => Err(DslError::ParseError {
                message: format!("expected {expected:?}, got end of input"),
                span: self.eof_span(),
            }),
        }
    }

    fn expect_number(&mut self) -> Result<(f64, Span), DslError> {
        match self.peek() {
            Some(Token {
                kind: TokenKind::Number(n),
                span,
            }) => {
                let n = *n;
                let span = *span;
                self.pos += 1;
                Ok((n, span))
            }
            Some(tok) => Err(DslError::ParseError {
                message: format!("expected number, got {:?}", tok.kind),
                span: tok.span,
            }),
            None => Err(DslError::ParseError {
                message: "expected number, got end of input".to_string(),
                span: self.eof_span(),
            }),
        }
    }

    fn expect_string(&mut self) -> Result<(String, Span), DslError> {
        match self.peek() {
            Some(Token {
                kind: TokenKind::StringLit(s),
                span,
            }) => {
                let s = s.clone();
                let span = *span;
                self.pos += 1;
                Ok((s, span))
            }
            Some(tok) => Err(DslError::ParseError {
                message: format!("expected string, got {:?}", tok.kind),
                span: tok.span,
            }),
            None => Err(DslError::ParseError {
                message: "expected string, got end of input".to_string(),
                span: self.eof_span(),
            }),
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), DslError> {
        match self.peek() {
            Some(Token {
                kind: TokenKind::Ident(s),
                span,
            }) => {
                let s = s.clone();
                let span = *span;
                self.pos += 1;
                Ok((s, span))
            }
            Some(Token {
                kind: TokenKind::Notional,
                span,
            }) => {
                let span = *span;
                self.pos += 1;
                Ok(("notional".to_string(), span))
            }
            Some(Token {
                kind: TokenKind::Maturity,
                span,
            }) => {
                let span = *span;
                self.pos += 1;
                Ok(("maturity".to_string(), span))
            }
            Some(tok) => Err(DslError::ParseError {
                message: format!("expected identifier, got {:?}", tok.kind),
                span: tok.span,
            }),
            None => Err(DslError::ParseError {
                message: "expected identifier, got end of input".to_string(),
                span: self.eof_span(),
            }),
        }
    }

    fn current_span(&self) -> Span {
        self.peek()
            .map(|t| t.span)
            .unwrap_or_else(|| self.eof_span())
    }

    fn eof_span(&self) -> Span {
        if let Some(last) = self.tokens.last() {
            Span::new(last.span.end, last.span.end)
        } else {
            Span::new(0, 0)
        }
    }
}

/// Parse a token stream into a `ProductDef` AST.
pub fn parse(tokens: Vec<Token>) -> Result<ProductDef, DslError> {
    let mut parser = Parser::new(tokens);
    parse_product(&mut parser)
}

fn parse_product(p: &mut Parser) -> Result<ProductDef, DslError> {
    let start = p.current_span();
    p.expect(&TokenKind::Product)?;
    let (name, _) = p.expect_string()?;
    p.expect(&TokenKind::Indent)?;

    let mut body = Vec::new();
    while !matches!(p.peek_kind(), Some(TokenKind::Dedent) | None) {
        body.push(parse_product_item(p)?);
    }

    let end_span = p.current_span();
    p.expect(&TokenKind::Dedent)?;

    Ok(ProductDef {
        name,
        span: Span::new(start.start, end_span.end),
        body,
    })
}

fn parse_product_item(p: &mut Parser) -> Result<ProductItem, DslError> {
    match p.peek_kind() {
        Some(TokenKind::Notional) => {
            let span = p.current_span();
            p.advance();
            p.expect(&TokenKind::Colon)?;
            let (val, end_span) = p.expect_number()?;
            Ok(ProductItem::Notional(
                val,
                Span::new(span.start, end_span.end),
            ))
        }
        Some(TokenKind::Maturity) => {
            let span = p.current_span();
            p.advance();
            p.expect(&TokenKind::Colon)?;
            let (val, end_span) = p.expect_number()?;
            Ok(ProductItem::Maturity(
                val,
                Span::new(span.start, end_span.end),
            ))
        }
        Some(TokenKind::Underlyings) => {
            let span = p.current_span();
            p.advance();
            p.expect(&TokenKind::Indent)?;
            let mut decls = Vec::new();
            while !matches!(p.peek_kind(), Some(TokenKind::Dedent) | None) {
                decls.push(parse_underlying_decl(p)?);
            }
            let end = p.current_span();
            p.expect(&TokenKind::Dedent)?;
            Ok(ProductItem::Underlyings(
                decls,
                Span::new(span.start, end.end),
            ))
        }
        Some(TokenKind::State) => {
            let span = p.current_span();
            p.advance();
            p.expect(&TokenKind::Indent)?;
            let mut decls = Vec::new();
            while !matches!(p.peek_kind(), Some(TokenKind::Dedent) | None) {
                decls.push(parse_state_decl(p)?);
            }
            let end = p.current_span();
            p.expect(&TokenKind::Dedent)?;
            Ok(ProductItem::State(decls, Span::new(span.start, end.end)))
        }
        Some(TokenKind::Schedule) => {
            let sched = parse_schedule(p)?;
            Ok(ProductItem::Schedule(sched))
        }
        Some(_) => {
            let tok = p.peek().unwrap();
            Err(DslError::ParseError {
                message: format!("unexpected token {:?} in product body", tok.kind),
                span: tok.span,
            })
        }
        None => Err(DslError::ParseError {
            message: "unexpected end of input in product body".to_string(),
            span: p.eof_span(),
        }),
    }
}

fn parse_underlying_decl(p: &mut Parser) -> Result<UnderlyingDecl, DslError> {
    let (name, span) = p.expect_ident()?;
    p.expect(&TokenKind::Eq)?;
    p.expect(&TokenKind::Asset)?;
    p.expect(&TokenKind::LParen)?;
    let (idx, end_span) = p.expect_number()?;
    p.expect(&TokenKind::RParen)?;
    Ok(UnderlyingDecl {
        name,
        asset_index: idx as usize,
        span: Span::new(span.start, end_span.end),
    })
}

fn parse_state_decl(p: &mut Parser) -> Result<StateDecl, DslError> {
    let (name, span) = p.expect_ident()?;
    p.expect(&TokenKind::Colon)?;

    let type_name = match p.peek_kind() {
        Some(TokenKind::Bool) => {
            p.advance();
            "bool".to_string()
        }
        Some(TokenKind::Float) => {
            p.advance();
            "float".to_string()
        }
        Some(TokenKind::Ident(s)) => {
            let s = s.clone();
            p.advance();
            s
        }
        _ => {
            return Err(DslError::ParseError {
                message: "expected type name (bool or float)".to_string(),
                span: p.current_span(),
            });
        }
    };

    p.expect(&TokenKind::Eq)?;
    let initial_value = parse_expr(p)?;

    Ok(StateDecl {
        name,
        type_name,
        initial_value,
        span: Span::new(span.start, p.current_span().start),
    })
}

fn parse_schedule(p: &mut Parser) -> Result<ScheduleDef, DslError> {
    let span = p.current_span();
    p.expect(&TokenKind::Schedule)?;

    let frequency = match p.peek_kind() {
        Some(TokenKind::Monthly) => {
            p.advance();
            ScheduleFreq::Monthly
        }
        Some(TokenKind::Quarterly) => {
            p.advance();
            ScheduleFreq::Quarterly
        }
        Some(TokenKind::SemiAnnual) => {
            p.advance();
            ScheduleFreq::SemiAnnual
        }
        Some(TokenKind::Annual) => {
            p.advance();
            ScheduleFreq::Annual
        }
        Some(TokenKind::Number(n)) => {
            let n = *n;
            p.advance();
            ScheduleFreq::Custom(n)
        }
        _ => {
            return Err(DslError::ParseError {
                message: "expected schedule frequency".to_string(),
                span: p.current_span(),
            });
        }
    };

    p.expect(&TokenKind::From)?;
    let (start, _) = p.expect_number()?;
    p.expect(&TokenKind::To)?;
    let (end, _) = p.expect_number()?;

    p.expect(&TokenKind::Indent)?;

    let mut body = Vec::new();
    while !matches!(p.peek_kind(), Some(TokenKind::Dedent) | None) {
        body.push(parse_statement(p)?);
    }

    let end_span = p.current_span();
    p.expect(&TokenKind::Dedent)?;

    Ok(ScheduleDef {
        frequency,
        start,
        end,
        body,
        span: Span::new(span.start, end_span.end),
    })
}

fn parse_statement(p: &mut Parser) -> Result<AstStatement, DslError> {
    let span = p.current_span();
    match p.peek_kind() {
        Some(TokenKind::Let) => {
            p.advance();
            let (name, _) = p.expect_ident()?;
            p.expect(&TokenKind::Eq)?;
            let expr = parse_expr(p)?;
            Ok(AstStatement {
                kind: AstStatementKind::Let { name, expr },
                span: Span::new(span.start, p.current_span().start),
            })
        }
        Some(TokenKind::If) => parse_if_statement(p),
        Some(TokenKind::Pay) => {
            p.advance();
            let amount = parse_expr(p)?;
            Ok(AstStatement {
                kind: AstStatementKind::Pay { amount },
                span: Span::new(span.start, p.current_span().start),
            })
        }
        Some(TokenKind::Redeem) => {
            p.advance();
            let amount = parse_expr(p)?;
            Ok(AstStatement {
                kind: AstStatementKind::Redeem { amount },
                span: Span::new(span.start, p.current_span().start),
            })
        }
        Some(TokenKind::Set) => {
            p.advance();
            let (name, _) = p.expect_ident()?;
            p.expect(&TokenKind::Eq)?;
            let expr = parse_expr(p)?;
            Ok(AstStatement {
                kind: AstStatementKind::SetState { name, expr },
                span: Span::new(span.start, p.current_span().start),
            })
        }
        Some(TokenKind::Skip) => {
            p.advance();
            Ok(AstStatement {
                kind: AstStatementKind::Skip,
                span,
            })
        }
        _ => Err(DslError::ParseError {
            message: format!(
                "expected statement (let, if, pay, redeem, set, skip), got {:?}",
                p.peek_kind()
            ),
            span,
        }),
    }
}

fn parse_if_statement(p: &mut Parser) -> Result<AstStatement, DslError> {
    let span = p.current_span();
    p.expect(&TokenKind::If)?;
    let condition = parse_expr(p)?;
    p.expect(&TokenKind::Then)?;

    // Body is an indented block.
    p.expect(&TokenKind::Indent)?;
    let mut then_body = Vec::new();
    while !matches!(p.peek_kind(), Some(TokenKind::Dedent) | None) {
        then_body.push(parse_statement(p)?);
    }
    p.expect(&TokenKind::Dedent)?;

    // Optional else clause.
    let mut else_body = Vec::new();
    if matches!(p.peek_kind(), Some(TokenKind::Else)) {
        p.advance();
        if matches!(p.peek_kind(), Some(TokenKind::If)) {
            // else if  (on the same line as else)
            let elif = parse_if_statement(p)?;
            else_body.push(elif);
        } else {
            // else block
            p.expect(&TokenKind::Indent)?;
            while !matches!(p.peek_kind(), Some(TokenKind::Dedent) | None) {
                else_body.push(parse_statement(p)?);
            }
            p.expect(&TokenKind::Dedent)?;
        }
    }

    Ok(AstStatement {
        kind: AstStatementKind::If {
            condition,
            then_body,
            else_body,
        },
        span: Span::new(span.start, p.current_span().start),
    })
}

// --- Expression parsing (precedence climbing) ---

fn parse_expr(p: &mut Parser) -> Result<AstExpr, DslError> {
    parse_or_expr(p)
}

fn parse_or_expr(p: &mut Parser) -> Result<AstExpr, DslError> {
    let mut left = parse_and_expr(p)?;
    while matches!(p.peek_kind(), Some(TokenKind::Or)) {
        p.advance();
        let right = parse_and_expr(p)?;
        let span = Span::new(left.span.start, right.span.end);
        left = AstExpr {
            kind: AstExprKind::BinOp {
                op: AstBinOp::Or,
                lhs: Box::new(left),
                rhs: Box::new(right),
            },
            span,
        };
    }
    Ok(left)
}

fn parse_and_expr(p: &mut Parser) -> Result<AstExpr, DslError> {
    let mut left = parse_not_expr(p)?;
    while matches!(p.peek_kind(), Some(TokenKind::And)) {
        p.advance();
        let right = parse_not_expr(p)?;
        let span = Span::new(left.span.start, right.span.end);
        left = AstExpr {
            kind: AstExprKind::BinOp {
                op: AstBinOp::And,
                lhs: Box::new(left),
                rhs: Box::new(right),
            },
            span,
        };
    }
    Ok(left)
}

fn parse_not_expr(p: &mut Parser) -> Result<AstExpr, DslError> {
    if matches!(p.peek_kind(), Some(TokenKind::Not)) {
        let span = p.current_span();
        p.advance();
        let operand = parse_not_expr(p)?;
        let end = operand.span.end;
        return Ok(AstExpr {
            kind: AstExprKind::UnaryOp {
                op: AstUnaryOp::Not,
                operand: Box::new(operand),
            },
            span: Span::new(span.start, end),
        });
    }
    parse_comparison(p)
}

fn parse_comparison(p: &mut Parser) -> Result<AstExpr, DslError> {
    let left = parse_additive(p)?;
    let op = match p.peek_kind() {
        Some(TokenKind::EqEq) => AstBinOp::Eq,
        Some(TokenKind::Ne) => AstBinOp::Ne,
        Some(TokenKind::Lt) => AstBinOp::Lt,
        Some(TokenKind::Le) => AstBinOp::Le,
        Some(TokenKind::Gt) => AstBinOp::Gt,
        Some(TokenKind::Ge) => AstBinOp::Ge,
        _ => return Ok(left),
    };
    p.advance();
    let right = parse_additive(p)?;
    let span = Span::new(left.span.start, right.span.end);
    Ok(AstExpr {
        kind: AstExprKind::BinOp {
            op,
            lhs: Box::new(left),
            rhs: Box::new(right),
        },
        span,
    })
}

fn parse_additive(p: &mut Parser) -> Result<AstExpr, DslError> {
    let mut left = parse_multiplicative(p)?;
    loop {
        let op = match p.peek_kind() {
            Some(TokenKind::Plus) => AstBinOp::Add,
            Some(TokenKind::Minus) => AstBinOp::Sub,
            _ => break,
        };
        p.advance();
        let right = parse_multiplicative(p)?;
        let span = Span::new(left.span.start, right.span.end);
        left = AstExpr {
            kind: AstExprKind::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            },
            span,
        };
    }
    Ok(left)
}

fn parse_multiplicative(p: &mut Parser) -> Result<AstExpr, DslError> {
    let mut left = parse_unary(p)?;
    loop {
        let op = match p.peek_kind() {
            Some(TokenKind::Star) => AstBinOp::Mul,
            Some(TokenKind::Slash) => AstBinOp::Div,
            _ => break,
        };
        p.advance();
        let right = parse_unary(p)?;
        let span = Span::new(left.span.start, right.span.end);
        left = AstExpr {
            kind: AstExprKind::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            },
            span,
        };
    }
    Ok(left)
}

fn parse_unary(p: &mut Parser) -> Result<AstExpr, DslError> {
    if matches!(p.peek_kind(), Some(TokenKind::Minus)) {
        let span = p.current_span();
        p.advance();
        let operand = parse_primary(p)?;
        let end = operand.span.end;
        return Ok(AstExpr {
            kind: AstExprKind::UnaryOp {
                op: AstUnaryOp::Neg,
                operand: Box::new(operand),
            },
            span: Span::new(span.start, end),
        });
    }
    parse_primary(p)
}

fn parse_primary(p: &mut Parser) -> Result<AstExpr, DslError> {
    let span = p.current_span();
    match p.peek_kind().cloned() {
        Some(TokenKind::Number(n)) => {
            p.advance();
            Ok(AstExpr {
                kind: AstExprKind::NumberLit(n),
                span,
            })
        }
        Some(TokenKind::True) => {
            p.advance();
            Ok(AstExpr {
                kind: AstExprKind::BoolLit(true),
                span,
            })
        }
        Some(TokenKind::False) => {
            p.advance();
            Ok(AstExpr {
                kind: AstExprKind::BoolLit(false),
                span,
            })
        }
        Some(TokenKind::LParen) => {
            p.advance();
            let expr = parse_expr(p)?;
            p.expect(&TokenKind::RParen)?;
            Ok(expr)
        }
        Some(TokenKind::Ident(name)) => {
            p.advance();
            if matches!(p.peek_kind(), Some(TokenKind::LParen)) {
                p.advance();
                let mut args = Vec::new();
                if !matches!(p.peek_kind(), Some(TokenKind::RParen)) {
                    args.push(parse_expr(p)?);
                    while matches!(p.peek_kind(), Some(TokenKind::Comma)) {
                        p.advance();
                        args.push(parse_expr(p)?);
                    }
                }
                let end = p.current_span();
                p.expect(&TokenKind::RParen)?;
                Ok(AstExpr {
                    kind: AstExprKind::FnCall { name, args },
                    span: Span::new(span.start, end.end),
                })
            } else {
                Ok(AstExpr {
                    kind: AstExprKind::Ident(name),
                    span,
                })
            }
        }
        Some(TokenKind::Notional) => {
            p.advance();
            Ok(AstExpr {
                kind: AstExprKind::Ident("notional".to_string()),
                span,
            })
        }
        Some(TokenKind::Maturity) => {
            p.advance();
            Ok(AstExpr {
                kind: AstExprKind::Ident("maturity".to_string()),
                span,
            })
        }
        _ => Err(DslError::ParseError {
            message: format!("expected expression, got {:?}", p.peek_kind()),
            span,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::lexer::tokenize;

    fn parse_str(s: &str) -> Result<ProductDef, DslError> {
        let tokens = tokenize(s)?;
        parse(tokens)
    }

    #[test]
    fn parse_minimal_product() {
        let product = parse_str(
            "product \"Test\"\n    notional: 1000\n    maturity: 1.0\n",
        )
        .unwrap();

        assert_eq!(product.name, "Test");
        assert_eq!(product.body.len(), 2);
        match &product.body[0] {
            ProductItem::Notional(v, _) => assert_eq!(*v, 1000.0),
            _ => panic!("expected notional"),
        }
        match &product.body[1] {
            ProductItem::Maturity(v, _) => assert_eq!(*v, 1.0),
            _ => panic!("expected maturity"),
        }
    }

    #[test]
    fn parse_underlyings_block() {
        let source = "\
product \"Test\"
    notional: 100
    maturity: 1.0
    underlyings
        SPX = asset(0)
        SX5E = asset(1)
";
        let product = parse_str(source).unwrap();

        let underlyings = product
            .body
            .iter()
            .find_map(|item| match item {
                ProductItem::Underlyings(u, _) => Some(u),
                _ => None,
            })
            .unwrap();
        assert_eq!(underlyings.len(), 2);
        assert_eq!(underlyings[0].name, "SPX");
        assert_eq!(underlyings[0].asset_index, 0);
        assert_eq!(underlyings[1].name, "SX5E");
        assert_eq!(underlyings[1].asset_index, 1);
    }

    #[test]
    fn parse_schedule_with_if_then() {
        let source = "\
product \"Autocall\"
    notional: 1_000_000
    maturity: 1.5
    underlyings
        SPX = asset(0)
    schedule quarterly from 0.25 to 1.5
        let wof = worst_of(performances())
        if wof >= 1.0 and not is_final then
            pay notional * 0.08 * observation_date
            redeem notional
        if is_final then
            redeem notional
";
        let product = parse_str(source).unwrap();

        assert_eq!(product.name, "Autocall");
        let schedule = product
            .body
            .iter()
            .find_map(|item| match item {
                ProductItem::Schedule(s) => Some(s),
                _ => None,
            })
            .unwrap();
        assert_eq!(schedule.frequency, ScheduleFreq::Quarterly);
        assert_eq!(schedule.start, 0.25);
        assert_eq!(schedule.end, 1.5);
        assert_eq!(schedule.body.len(), 3); // let, if, if
    }

    #[test]
    fn parse_state_block() {
        let source = "\
product \"Test\"
    notional: 100
    maturity: 1.0
    state
        ki_hit: bool = false
        total: float = 0.0
";
        let product = parse_str(source).unwrap();

        let states = product
            .body
            .iter()
            .find_map(|item| match item {
                ProductItem::State(s, _) => Some(s),
                _ => None,
            })
            .unwrap();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].name, "ki_hit");
        assert_eq!(states[0].type_name, "bool");
        assert_eq!(states[1].name, "total");
        assert_eq!(states[1].type_name, "float");
    }

    #[test]
    fn parse_else_if_chain() {
        let source = "\
product \"Test\"
    notional: 100
    maturity: 1.0
    schedule annual from 1.0 to 1.0
        let x = 1.0
        if x > 2.0 then
            pay 100
        else if x > 1.0 then
            pay 50
        else
            pay 0
";
        let product = parse_str(source).unwrap();

        let schedule = product
            .body
            .iter()
            .find_map(|item| match item {
                ProductItem::Schedule(s) => Some(s),
                _ => None,
            })
            .unwrap();
        assert_eq!(schedule.body.len(), 2); // let, if-else-if-else
    }
}
