---
name: openferric-dsl
description: "Write and price structured products using the OpenFerric DSL. TRIGGER when: user asks to define, price, or modify a structured product (autocallable, phoenix note, accumulator, range accrual, TARF), or mentions the OpenFerric DSL, or wants to write DSL product definitions. DO NOT TRIGGER for: standard option pricing, Black-Scholes, or Rust API usage without DSL."
license: MIT
compatibility: Requires Rust toolchain (stable 1.87+) and the openferric crate
metadata:
  author: rosssaunders
  version: "0.1"
---

# OpenFerric Structured Product DSL

Write text-based definitions for exotic structured products that compile to bytecode and run inside a multi-asset Monte Carlo engine.

## Pipeline

```
DSL text --> Lexer --> Parser --> AST --> Compiler --> IR --> Evaluator
                                                               ^
                                               MC Engine ------+
                                               (correlated GBM paths)
```

## When to Use the DSL

Use the DSL when the user wants to:
- Define a structured product (autocallable, phoenix, accumulator, range accrual, TARF, etc.)
- Price an exotic payoff without writing Rust code
- Prototype or iterate on product structures quickly
- Compute Greeks (delta, gamma, vega, rho) for path-dependent exotics

Do NOT use the DSL for:
- Vanilla European/American options (use `VanillaOption` + `BlackScholesEngine` directly)
- Fixed income instruments (use the `rates/` module)
- CDS pricing (use the `credit/` module)

## Product Structure

Every DSL product follows this template:

```
product "<name>"
    notional: <number>
    maturity: <number>

    underlyings
        <NAME> = asset(<index>)

    state                          // optional
        <name>: <type> = <default>

    schedule <frequency> from <start> to <end>
        <statements>
```

### Required fields
- `notional` - face value / nominal amount
- `maturity` - product maturity in year fractions

### Underlyings
Map named assets to indices in the multi-asset market:
```
underlyings
    SPX  = asset(0)
    SX5E = asset(1)
```

### State variables
Mutable values that persist across observation dates within a path:
```
state
    ki_hit: bool = false
    missed_coupons: float = 0.0
```
Types: `bool`, `float`. State resets at each MC path start.

### Schedule frequencies
`monthly` (1/12y), `quarterly` (0.25y), `semi_annual` (0.5y), `annual` (1.0y), or a custom number.

## Statements

| Statement | Effect |
|-----------|--------|
| `let x = expr` | Bind a local variable (scoped to current observation date) |
| `if expr then` | Conditional block (indent body). Supports `else if` and `else` |
| `pay expr` | Record a cashflow at current date (discounted to t=0) |
| `redeem expr` | Final payment + terminate the product (early redemption) |
| `set var = expr` | Mutate a state variable |
| `skip` | Skip remaining dates, no payment |

## Expressions

### Built-in variables
- `notional` - product notional
- `observation_date` - current date (year fraction)
- `is_final` - true on last observation date

### Built-in functions
- `performances()` - returns S_i(t)/S_i(0) for each underlying
- `worst_of(arr)` - minimum value (typically `worst_of(performances())`)
- `best_of(arr)` - maximum value
- `price(index)` - current spot of asset at index
- `min(a, b)`, `max(a, b)`, `abs(x)`, `exp(x)`, `log(x)`

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, unary `-`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical: `and`, `or`, `not`
- Comments: `// line comment`

### Precedence (low to high)
`or` < `and` < `not` < comparisons < `+`/`-` < `*`/`/` < unary `-` < calls/literals/parens

## Rust API

### Parse and price

```rust
use openferric::dsl::{parse_and_compile, DslMonteCarloEngine, MultiAssetMarket};

let product = parse_and_compile(source).unwrap();
let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
let engine = DslMonteCarloEngine::new(100_000, 252, 42);
let result = engine.price_multi_asset(&product, &market).unwrap();
```

### Multi-asset market

```rust
use openferric::dsl::{MultiAssetMarket, AssetData};

let market = MultiAssetMarket {
    assets: vec![
        AssetData { spot: 100.0, vol: 0.20, dividend_yield: 0.02 },
        AssetData { spot: 200.0, vol: 0.22, dividend_yield: 0.03 },
    ],
    correlation: vec![vec![1.0, 0.7], vec![0.7, 1.0]],
    rate: 0.03,
};
```

### Greeks (bump-and-reprice)

```rust
let greeks = engine.greeks_multi_asset(&product, &market, 0).unwrap();
// greeks.delta, greeks.gamma, greeks.vega, greeks.rho
```

| Greek | Method |
|-------|--------|
| Delta | Central difference, 1% spot bump |
| Gamma | Central difference, 1% spot bump |
| Vega  | Forward difference, 1% vol bump |
| Rho   | Forward difference, 1bp rate bump |

### Serialization

`CompiledProduct` implements `serde::Serialize` + `Deserialize` for JSON/MessagePack round-trips.

## Error handling

```rust
match parse_and_compile(source) {
    Ok(product) => { /* use product */ }
    Err(e) => eprintln!("DSL error: {e}"),
    // "compile error at 45-52: undefined variable 'foo'"
}
```

Error types: `LexError`, `ParseError`, `CompileError`, `EvalError`.

## Common Patterns

See [references/examples.md](references/examples.md) for complete product examples including worst-of autocallables, phoenix memory coupons, accumulators, and range accruals.

See [references/grammar.md](references/grammar.md) for the full EBNF grammar specification.
