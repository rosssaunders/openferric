# OpenFerric DSL — AI Assistant Guide

When the user asks to "price a deal", "structure a product", or describes a structured product payoff, generate an `.of` file using the OpenFerric DSL. Save it in `examples/dsl/` or the workspace root.

After generating the file, tell the user to open it in VS Code — the LSP will automatically price it and show results in the Pricing Dashboard sidebar panel.

## DSL Quick Reference

Every `.of` file defines one product:

```
product "Product Name"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule quarterly from 0.25 to 1.0
        // schedule body
```

### Structure

| Block | Required | Description |
|-------|----------|-------------|
| `product "Name"` | Yes | Product name (quoted string) |
| `notional: N` | Yes | Notional amount |
| `maturity: T` | Yes | Maturity in years |
| `underlyings` | Yes | Asset declarations |
| `state` | No | Persistent variables across observations |
| `schedule freq from T1 to T2` | Yes | Observation schedule + payoff logic |

### Underlyings

```
underlyings
    SPX  = asset(0)      // first asset in market data
    SX5E = asset(1)      // second asset
    NKY  = asset(2)      // third asset
```

Asset indices map to the market data array in VS Code settings (`openferric.market.default.assets`). Default: asset(0) has spot=100, vol=20%, div_yield=2%.

### State Variables

Persist across observation dates within a schedule. Reset per Monte Carlo path.

```
state
    ki_hit: bool = false
    missed_coupons: float = 0.0
    max_perf: float = 1.0
```

Types: `bool`, `float`.

### Schedule Frequencies

| Keyword | Period |
|---------|--------|
| `monthly` | 1/12 year |
| `quarterly` | 0.25 year |
| `semi_annual` | 0.5 year |
| `annual` | 1.0 year |

### Statements (inside schedule blocks)

| Statement | Example | Description |
|-----------|---------|-------------|
| `let` | `let wof = worst_of(performances())` | Local variable (resets each observation) |
| `pay` | `pay notional * 0.05` | Cashflow to investor |
| `redeem` | `redeem notional` | Return principal + terminate early |
| `set` | `set ki_hit = true` | Update a state variable |
| `if/then/else` | `if perf >= 1.0 then ...` | Conditional logic |
| `skip` | `skip` | No-op |

### Built-in Functions

| Function | Description |
|----------|-------------|
| `performances()` | Vector of S_i(t)/S_i(0) for all underlyings |
| `worst_of(v)` | Minimum of a vector |
| `best_of(v)` | Maximum of a vector |
| `min(a, b)` | Minimum of two numbers |
| `max(a, b)` | Maximum of two numbers |
| `abs(x)` | Absolute value |
| `exp(x)` | Exponential |
| `log(x)` | Natural logarithm |

### Built-in Identifiers

| Name | Available | Description |
|------|-----------|-------------|
| `notional` | Always | The product's notional |
| `maturity` | Always | The product's maturity |
| `observation_date` | In schedule | Current observation time (year fraction) |
| `is_final` | In schedule | `true` on the last observation |

### Operators

Arithmetic: `+`, `-`, `*`, `/`
Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
Logical: `and`, `or`, `not`

### Syntax Rules

- Indentation-based blocks (like Python) — use 4 spaces
- Comments: `// single line`
- Numbers: `1_000_000`, `0.08`, `1.0`
- Strings: `"double quoted"`
- `if` requires `then` keyword
- `else if` chains are supported
- `redeem` terminates the product early (no further observations)

## Common Product Patterns

### Pattern: Autocall

Early redeem if performance above barrier:
```
if wof >= 1.0 and not is_final then
    pay notional * coupon * observation_date
    redeem notional
```

### Pattern: Knock-In Barrier

Track if barrier breached, apply at maturity:
```
state
    ki_hit: bool = false
// in schedule:
if perf <= 0.60 then
    set ki_hit = true
// at maturity:
if is_final then
    if ki_hit and wof < 1.0 then
        redeem notional * wof
    else
        redeem notional
```

### Pattern: Knock-Out Barrier

Terminate early with rebate if barrier breached:
```
if perf >= 1.40 and not is_final then
    pay notional * 0.05
    redeem notional
```

### Pattern: Memory Coupon

Accumulate missed coupons, pay when barrier is next above:
```
state
    missed_coupons: float = 0.0
// in schedule:
if wof >= 0.70 then
    pay notional * (0.02 + missed_coupons)
    set missed_coupons = 0.0
else
    set missed_coupons = missed_coupons + 0.02
```

### Pattern: Step-Down Barrier

Barrier decreases over time:
```
let barrier = 1.05 - observation_date * 0.2 + 0.05
if wof >= barrier and not is_final then
    redeem notional
```

### Pattern: Snowball Coupon

Coupon grows each period if above barrier, resets to zero otherwise:
```
state
    current_coupon: float = 0.0
// in schedule:
if perf >= 0.80 then
    set current_coupon = current_coupon + 0.01
    pay notional * current_coupon
else
    set current_coupon = 0.0
```

### Pattern: Asian / Average

Average performance across observations:
```
state
    sum_perf: float = 0.0
    num_obs: float = 0.0
// in schedule:
set sum_perf = sum_perf + perf
set num_obs = num_obs + 1.0
if is_final then
    let avg = sum_perf / num_obs
```

### Pattern: Tiered / Wedding Cake Coupon

```
if wof >= 1.0 then
    pay notional * 0.03
else if wof >= 0.80 then
    pay notional * 0.02
else if wof >= 0.60 then
    pay notional * 0.01
```

### Pattern: Range Accrual

Count observations where spot is within range:
```
state
    days_in_range: float = 0.0
// in schedule:
if perf >= 0.80 and perf <= 1.20 then
    set days_in_range = days_in_range + 1.0
if is_final then
    let frac = days_in_range / 12.0
    pay notional * 0.10 * frac
```

### Pattern: Capital Protection

Guarantee return of principal:
```
let upside = max(perf - 1.0, 0.0) * 0.60
pay notional * upside
redeem notional   // always redeem at par
```

### Pattern: Dispersion / Spread

Profit from divergence between best and worst:
```
let bof = best_of(performances())
let wof = worst_of(performances())
let spread = bof - wof
pay notional * spread * 0.50
```

## Natural Language → DSL Mapping

When the user describes a product, map their language to DSL constructs:

| User says | DSL construct |
|-----------|---------------|
| "autocallable", "early redemption" | `if wof >= barrier and not is_final then redeem` |
| "knock-in", "barrier at X%" | `state ki_hit: bool` + `if perf <= X then set ki_hit = true` |
| "knock-out", "KO at X%" | `if perf >= X then redeem` with rebate |
| "worst-of", "basket" | `worst_of(performances())` with multiple underlyings |
| "best-of", "rainbow" | `best_of(performances())` |
| "coupon of X%", "pays X% pa" | `pay notional * X` (per-period fraction) |
| "memory coupon" | State variable `missed_coupons` pattern |
| "snowball" | State variable `current_coupon` growing pattern |
| "capital protected", "principal protected" | Always `redeem notional` at maturity |
| "participation X%" | Multiply upside by X |
| "cap at X%" | `min(upside, X)` |
| "step-down" | Barrier computed from `observation_date` |
| "cliquet", "ratchet" | Track periodic returns with local floor/cap |
| "accumulator", "TARF" | Running total with target knockout |
| "range accrual" | Count days in range, pay proportional coupon |
| "digital" | All-or-nothing coupon: `if perf >= barrier then pay` |
| "express" | Autocall with growing premium |
| "lookback" | Track `max_perf` via state variable |
| "asian", "average price" | Accumulate `sum_perf` / `num_obs` |
| "shark fin" | Capital protected + KO barrier with rebate |
| "twin win" | Profit from abs(move) unless KI hit |
| "airbag", "cushion" | Buffer zone absorbs first N% of losses |
| "on SPX", "on S&P" | `SPX = asset(0)` |
| "on SPX and Euro Stoxx" | `SPX = asset(0)` + `SX5E = asset(1)` |
| "quarterly" | `schedule quarterly from ...` |
| "monthly monitoring" | `schedule monthly from ...` |
| "1 year", "18 months" | `maturity: 1.0` or `maturity: 1.5` |
| "notional 1 million" | `notional: 1_000_000` |

## Important Notes

- Time is always in year fractions (1 year = 1.0, 6 months = 0.5)
- Rates are continuously compounded
- `performances()` returns S(t)/S(0) — so 1.0 = at par, 0.8 = down 20%
- Barriers are expressed as fractions: 60% barrier = 0.60
- Coupons per period: 8% p.a. quarterly = `notional * 0.02` per quarter
- The schedule `from/to` defines the first and last observation dates
- `redeem` terminates the product — no further observations occur on that path
- Default market: spot=100, vol=20%, rate=5%, div_yield=2% (configurable in VS Code settings)
