# OpenFerric Structured Product DSL

A text-based domain-specific language for defining and pricing exotic structured
products without writing Rust code.

## Overview

The DSL complements existing hand-coded instruments (`VanillaOption`,
`BarrierOption`, etc.) by adding a composable layer for exotic structured
products such as autocallables, Phoenix notes, accumulators, and range accruals.

Products defined in DSL text are compiled to an intermediate representation (IR)
and evaluated inside a multi-asset Monte Carlo engine with correlated GBM path
generation.

```text
DSL text ──→ Lexer ──→ Parser ──→ AST ──→ Compiler ──→ IR ──→ Evaluator
                                                                   ↑
                                                   MC Engine ──────┘
                                                   (correlated GBM paths)
```

## Quick Start (Rust)

```rust
use openferric::dsl::{parse_and_compile, DslMonteCarloEngine, MultiAssetMarket};

let source = "\
product \"Zero Coupon\"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        redeem notional
";

let product = parse_and_compile(source).unwrap();
let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
let engine = DslMonteCarloEngine::new(100_000, 252, 42);
let result = engine.price_multi_asset(&product, &market).unwrap();

println!("Price: {:.2}", result.price);
println!("Stderr: {:.4}", result.stderr.unwrap());
```

## Language Reference

The DSL uses **indentation-based blocks** (no braces) and the `then` keyword for
conditionals, inspired by F# and Python.

### Product Definition

Every DSL file defines a single product with an indented body:

```
product "<name>"
    notional: <number>
    maturity: <number>
    ...
```

- **`notional`** (required): The face value / nominal amount. Available as
  `notional` in expressions.
- **`maturity`** (required): Product maturity in year fractions.

### Underlyings

Declares the underlying assets by name and maps each to an index into the
multi-asset market data:

```
underlyings
    SPX  = asset(0)
    SX5E = asset(1)
    NKY  = asset(2)
```

Asset indices correspond to positions in the `MultiAssetMarket.assets` vector.
For single-asset products, declare one underlying with `asset(0)`.

### State Variables

Mutable variables that persist across observation dates within a single MC path.
Use these for knock-in flags, coupon memory, accumulated totals, etc.

```
state
    ki_hit: bool = false
    missed_coupons: float = 0.0
    total_accumulated: float = 0.0
```

Supported types: `bool`, `float`. State resets to initial values at the start of
each MC path.

### Schedules

A schedule defines a series of observation dates and the logic executed at each:

```
schedule <frequency> from <start> to <end>
    <statements>
```

**Frequencies:**

| Keyword        | Period          |
|----------------|-----------------|
| `monthly`      | 1/12 year       |
| `quarterly`    | 0.25 year       |
| `semi_annual`  | 0.5 year        |
| `annual`       | 1.0 year        |
| `<number>`     | custom period   |

The compiler generates observation dates from `start` to `end` (inclusive) at the
given frequency. For example, `schedule quarterly from 0.25 to 1.5` produces
dates `[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`.

A product can have multiple schedule blocks if needed.

### Statements

Statements execute at each observation date on each MC path.

#### `let`

Bind a local variable (scoped to the current observation date):

```
let wof = worst_of(performances())
let coupon = notional * 0.08 * observation_date
```

#### `if` / `else`

Conditional execution uses the `then` keyword followed by an indented body:

```
if wof >= 1.0 and not is_final then
    pay coupon
    redeem notional

if condition then
    ...
else if other_condition then
    ...
else
    ...
```

#### `pay`

Record a cashflow at the current observation date. The cashflow is discounted
back to time 0 using `exp(-r * t)`:

```
pay notional * 0.08 * observation_date
```

Multiple `pay` statements can appear at the same observation date; their amounts
accumulate.

#### `redeem`

Record a final payment and **terminate** the product (early redemption). No
further observation dates are processed for this path:

```
redeem notional
redeem notional * wof
```

#### `set`

Mutate a state variable:

```
set ki_hit = true
set missed_coupons = missed_coupons + 0.02
```

#### `skip`

Skip all remaining observation dates without any payment (early exit):

```
skip
```

### Expressions

#### Literals

- Numbers: `100`, `0.08`, `1_000_000`, `3.14e-2`
- Booleans: `true`, `false`

Number literals support `_` as a visual separator (e.g., `1_000_000`).

#### Arithmetic Operators

| Operator | Meaning        | Precedence |
|----------|----------------|------------|
| `*`      | Multiplication | High       |
| `/`      | Division       | High       |
| `+`      | Addition       | Medium     |
| `-`      | Subtraction    | Medium     |
| `-`      | Negation (unary) | Highest  |

#### Comparison Operators

| Operator | Meaning                |
|----------|------------------------|
| `==`     | Equal                  |
| `!=`     | Not equal              |
| `<`      | Less than              |
| `<=`     | Less than or equal     |
| `>`      | Greater than           |
| `>=`     | Greater than or equal  |

#### Logical Operators

| Operator | Meaning     | Precedence |
|----------|-------------|------------|
| `not`    | Logical NOT | Highest    |
| `and`    | Logical AND | Medium     |
| `or`     | Logical OR  | Lowest     |

#### Built-in Variables

| Name               | Type  | Description                                      |
|--------------------|-------|--------------------------------------------------|
| `notional`         | float | Product notional amount                          |
| `observation_date` | float | Current observation date (year fraction)         |
| `is_final`         | bool  | True on the last observation date of a schedule  |

#### Built-in Functions

| Function           | Signature            | Description                                        |
|--------------------|----------------------|----------------------------------------------------|
| `performances()`   | `() -> [float]`      | Returns `S_i(t) / S_i(0)` for each underlying     |
| `worst_of(...)`    | `([float]) -> float` | Minimum value. Typically `worst_of(performances())` |
| `best_of(...)`     | `([float]) -> float` | Maximum value. Typically `best_of(performances())`  |
| `price(index)`     | `(int) -> float`     | Current spot price of asset at `index`             |
| `min(a, b)`        | `(float, float) -> float` | Minimum of two values                        |
| `max(a, b)`        | `(float, float) -> float` | Maximum of two values                        |
| `abs(x)`           | `(float) -> float`   | Absolute value                                     |
| `exp(x)`           | `(float) -> float`   | Exponential                                        |
| `log(x)`           | `(float) -> float`   | Natural logarithm                                  |

#### Operator Precedence (low to high)

1. `or`
2. `and`
3. `not`
4. `==`, `!=`, `<`, `<=`, `>`, `>=`
5. `+`, `-`
6. `*`, `/`
7. Unary `-`
8. Function calls, literals, parenthesized expressions

### Comments

Line comments start with `//`:

```
// This is a comment
let wof = worst_of(performances())  // inline comment
```

## Product Examples

### Worst-of Autocallable

```
product "WoF Autocall 18m"
    notional: 1_000_000
    maturity: 1.5

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)
        NKY  = asset(2)

    state
        ki_hit: bool = false

    schedule quarterly from 0.25 to 1.5
        let wof = worst_of(performances())

        // Check knock-in barrier
        if wof <= 0.60 then
            set ki_hit = true

        // Early autocall if above par (excluding final date)
        if wof >= 1.0 and not is_final then
            pay notional * 0.08 * observation_date
            redeem notional

        // Maturity logic
        if is_final then
            pay notional * 0.08 * 1.5
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
```

### Phoenix with Memory Coupons

Pays a coupon when worst-of performance is above the coupon barrier. Missed
coupons accumulate and are paid when the barrier is subsequently breached.

```
product "Phoenix Memory"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false
        missed_coupons: float = 0.0

    schedule quarterly from 0.25 to 1.0
        let wof = worst_of(performances())

        if wof <= 0.60 then
            set ki_hit = true

        // Coupon with memory
        if wof >= 0.70 then
            pay notional * (0.02 + missed_coupons)
            set missed_coupons = 0.0
        else
            set missed_coupons = missed_coupons + 0.02

        // Early autocall
        if wof >= 1.0 and not is_final then
            redeem notional

        // Maturity
        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
```

### Accumulator (TARF)

Accumulates units when the spot is between two barriers, with a target cap:

```
product "Accumulator"
    notional: 100
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        accumulated: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if accumulated < 1.0 then
            if perf >= 0.90 and perf <= 1.10 then
                set accumulated = accumulated + 0.0833
                pay notional * 0.0833 * (perf - 1.0)

        if is_final then
            redeem notional
```

### Equity Range Accrual

Pays a coupon proportional to the number of days the underlying stays within a
range:

```
product "Range Accrual"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        days_in_range: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if perf >= 0.80 and perf <= 1.20 then
            set days_in_range = days_in_range + 1.0

        if is_final then
            let accrual_frac = days_in_range / 12.0
            pay notional * 0.10 * accrual_frac
            redeem notional
```

## Market Data

### Single-Asset

```rust
use openferric::dsl::MultiAssetMarket;

let market = MultiAssetMarket::single(
    100.0,  // spot
    0.20,   // volatility
    0.05,   // risk-free rate
    0.02,   // dividend yield
);
```

### Multi-Asset

```rust
use openferric::dsl::{MultiAssetMarket, AssetData};

let market = MultiAssetMarket {
    assets: vec![
        AssetData { spot: 100.0, vol: 0.20, dividend_yield: 0.02 },
        AssetData { spot: 200.0, vol: 0.22, dividend_yield: 0.03 },
        AssetData { spot: 50.0,  vol: 0.25, dividend_yield: 0.01 },
    ],
    correlation: vec![
        vec![1.0, 0.7, 0.5],
        vec![0.7, 1.0, 0.6],
        vec![0.5, 0.6, 1.0],
    ],
    rate: 0.03,
};
```

The correlation matrix is validated and repaired to nearest positive
semi-definite if needed (Higham 2002).

## Engine Configuration

```rust
use openferric::dsl::DslMonteCarloEngine;

let engine = DslMonteCarloEngine::new(
    100_000,  // num_paths
    252,      // num_steps (time steps per path)
    42,       // seed
);

// Price
let result = engine.price_multi_asset(&product, &market).unwrap();
println!("Price:  {:.2}", result.price);
println!("Stderr: {:.4}", result.stderr.unwrap());

// Greeks (bump-and-reprice for asset 0)
let greeks = engine.greeks_multi_asset(&product, &market, 0).unwrap();
println!("Delta: {:.4}", greeks.delta);
println!("Gamma: {:.6}", greeks.gamma);
println!("Vega:  {:.2}", greeks.vega);
println!("Rho:   {:.2}", greeks.rho);
```

### Greeks

Greeks are computed via bump-and-reprice (standard for path-dependent exotics):

| Greek | Method                            |
|-------|-----------------------------------|
| Delta | Central difference, 1% spot bump  |
| Gamma | Central difference, 1% spot bump  |
| Vega  | Forward difference, 1% vol bump   |
| Rho   | Forward difference, 1bp rate bump |

Each Greek is computed per asset (pass `asset_index` to `greeks_multi_asset`).

## Serialization

The compiled IR (`CompiledProduct`) derives `serde::Serialize` and
`serde::Deserialize`, enabling round-trip persistence:

```rust
let product = parse_and_compile(source).unwrap();

// Serialize to JSON
let json = serde_json::to_string(&product).unwrap();

// Deserialize back
let restored: CompiledProduct = serde_json::from_str(&json).unwrap();
assert_eq!(product, restored);
```

This allows compiled products to be cached, stored in databases, or transmitted
over the wire without re-parsing.

## Error Handling

The DSL produces span-annotated errors at each pipeline stage:

```rust
match parse_and_compile(source) {
    Ok(product) => { /* use product */ }
    Err(e) => {
        eprintln!("DSL error: {e}");
        // Example output:
        // "compile error at 45-52: undefined variable 'foo'"
        // "parse error at 12-15: expected identifier, got Number(1.0)"
    }
}
```

Error types:
- **`LexError`**: Unexpected character or malformed token.
- **`ParseError`**: Unexpected token or missing construct.
- **`CompileError`**: Type mismatch, undefined variable, duplicate declaration.
- **`EvalError`**: Runtime error during evaluation (e.g., asset index out of
  range).

## Grammar (EBNF)

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
