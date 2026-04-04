# DSL Grammar (EBNF)

The DSL uses indentation-based blocks. `INDENT` and `DEDENT` are virtual tokens
emitted by the lexer based on leading whitespace changes.

```ebnf
product       = "product" STRING INDENT product_item* DEDENT ;
product_item  = notional | maturity | underlyings | state | schedule ;

notional      = "notional" ":" NUMBER ;
maturity      = "maturity" ":" NUMBER ;

underlyings   = "underlyings" INDENT underlying_decl* DEDENT ;
underlying_decl = IDENT "=" "asset" "(" NUMBER ")" ;

state         = "state" INDENT state_decl* DEDENT ;
state_decl    = IDENT ":" type_name "=" expr ;
type_name     = "bool" | "float" ;

schedule      = "schedule" frequency "from" NUMBER "to" NUMBER INDENT statement* DEDENT ;
frequency     = "monthly" | "quarterly" | "semi_annual" | "annual" | NUMBER ;

statement     = let_stmt | if_stmt | pay_stmt | redeem_stmt | set_stmt | "skip" ;
let_stmt      = "let" IDENT "=" expr ;
if_stmt       = "if" expr "then" INDENT statement* DEDENT
                ( "else" ( if_stmt | INDENT statement* DEDENT ) )? ;
pay_stmt      = "pay" expr ;
redeem_stmt   = "redeem" expr ;
set_stmt      = "set" IDENT "=" expr ;

expr          = or_expr ;
or_expr       = and_expr ( "or" and_expr )* ;
and_expr      = not_expr ( "and" not_expr )* ;
not_expr      = "not" not_expr | comparison ;
comparison    = additive ( ( "==" | "!=" | "<" | "<=" | ">" | ">=" ) additive )? ;
additive      = multiplicative ( ( "+" | "-" ) multiplicative )* ;
multiplicative = unary ( ( "*" | "/" ) unary )* ;
unary         = "-" primary | primary ;
primary       = NUMBER | "true" | "false" | IDENT ( "(" arg_list? ")" )?
              | "notional" | "(" expr ")" ;
arg_list      = expr ( "," expr )* ;

NUMBER        = [0-9_]+ ( "." [0-9_]+ )? ( [eE] [+-]? [0-9]+ )? ;
STRING        = '"' ( [^"\\] | '\\' . )* '"' ;
IDENT         = [a-zA-Z_] [a-zA-Z0-9_]* ;
COMMENT       = "//" [^\n]* ;
```
