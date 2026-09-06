# Pricing correctness review — 2026-09-05

## Conclusion and scope

The initial accelerated-native workspace run passed **1,769 tests**, with one
explicitly ignored performance benchmark. Nevertheless, targeted economic
oracles exposed material mispricing and incorrect sensitivities. Passing the
existing suite was not sufficient assurance.

This review inspected analytic vanilla/Asian/barrier pricing, generic Monte
Carlo/control variates, PDE boundaries, and rates/credit cashflows. It also
examined the reference, invariant and cross-engine suites that exercise trees,
LSM, FFT/Heston/Lévy pricing and the other product families. This is not a claim
that every formula in the library was independently re-derived line by line.

The confirmed findings below are fixed. Their regression coverage is in
`tests/audit_pricing_boundaries.rs`, `tests/market_validation.rs`,
`python/tests/test_pricing_boundaries.py`, and `wasm/src/pricing.rs`.
The zero-volatility Greek contract is also exercised through the real WASM
boundary in `wasm/src/abi_tests.rs`.

## Confirmed findings and corrections

| Priority | Finding and reproduced consequence | Correction |
|---|---|---|
| P1 | Asian analytic pricing used the last observation date as the payment date. A time-zero fixing with a payoff of 5, payment at 1.5 years and rate -5% returned 5 instead of 5.389420754423158. Arithmetic Asian controls inherited the same settlement mismatch. | Separate undiscounted expected payoff from settlement discounting. Both analytic and MC prices discount at contractual expiry. |
| P1 | Arithmetic Asian controls used contractual fixing dates while path payoffs rounded them to grid dates. With fixings `[0, 0.37, 1]` and four steps, the call returned 7.3716537950 against the simulated-contract oracle 6.6421413478, despite a four-standard-error budget of about 0.0125. | Compute the control expectation on the actual rounded grid; share control construction between the generic and dedicated engines. |
| P1 | Geometric Asian dividend yield was smeared over expiry. A cash dividend after the only fixing changed a value that must already be fixed: 9.5754915562 instead of 13.0968403375 in the regression scenario. | Apply proportional drops according to affected fixing counts; ignore later dividends. Reject cash dividends that affect fixings rather than claim a lognormal closed form. |
| P1 | Double-barrier integration assumed the strike was inside the corridor. For a knock-out call with S=100, K=60, barriers 80/120, r=3%, q=1%, vol=25%, T=1, the price was 5.9682719438 instead of 7.1669963717. | Clip integration limits, not payoff strikes, to the survival corridor. Return zero when the payoff cannot be positive on survival. Test both directions and in/out parity. |
| P1 | Caplet/floorlet guards returned zero for valid zero-strike/zero-forward limits. A zero-strike caplet worth 9,500 returned zero. Negative lognormal forwards also silently returned zero. | Preserve the limiting cashflows and reuse the stable Black kernel. Invalid lognormal inputs return `NaN`; there is no implicit normal-model fallback. |
| P1 | CDS options returned zero at expiry, including a regression with intrinsic value 40,000. | Pay intrinsic at expiry; distinguish already expired options. Validate inputs and reuse the stable Black kernel, including zero-spread limits. |
| P1 | CDS annuity/spread helpers rounded the number of coupons, losing or extending the final accrual. A 0.1-year quarterly annuity returned zero instead of 0.09950124791926823. | Include the final short stub and stop cashflows at contractual maturity. Preserve the helpers' stated midpoint-default methodology. |
| P1 | Analytic/AAD discrete-dividend Greeks differentiated the adjusted spot, not the market inputs. The mixed-dividend call delta was 0.3538901594 instead of the market-bump derivative 0.2936732199. | Apply the prepaid-forward chain rule to delta, gamma, rho and calendar theta on both entry points. A saturated prepaid-spot clamp has zero market delta, confirmed by repricing a spot bump. |
| P2 | Black-76 returned all-zero Greeks at zero volatility. An ITM call with r=-3%, T=1.5 had delta 0 instead of 1.0460278599. | Return the deterministic derivatives, mark ATM delta/gamma as undefined, and retain right-hand ATM vega. |
| P1 | American PDE boundaries assumed immediate exercise. With S=0.1, K=100, r=-5%, q=0, vol=20%, T=1, CN returned 100.9024672656 instead of 105.0271096376. Calls likewise lost continuation carry at the upper boundary. | Use the negative-rate put boundary and maximize discounted call exercise over the remaining horizon. Share the boundary implementation across CN and the other one-dimensional solvers. Include maturity in the escrowed put boundary. |
| P1 | PDE evaluation silently clamped spots outside the configured grid to its endpoint. A call with spot 1,000 evaluated on an upper bound of 110 returned the boundary price rather than reporting an invalid domain. | Reject non-finite bounds and spots at or beyond the upper grid boundary in the one-dimensional PDE engines. Require an adequately wide domain instead of silently changing spot. |

### Greek conventions

For prepaid spot `M = S P exp(-qT) - sum(cash_pv)`, where `P` is the product of
applicable proportional-dividend factors:

- `dM/dS = P exp(-qT)`; gamma receives the square of this factor.
- `dM/dr = sum(ex_date * cash_pv)` contributes to rho.
- Calendar theta rolls both expiry and ex-dates, so
  `dM/dcalendar_time = q M - (r-q) sum(cash_pv)`.

These are frozen-volatility derivatives away from an ex-date or clamp kink;
they are not sensitivities of a recalibrated smile. The finite-difference
regression rolls dividend dates as well as expiry when checking theta.

## Oracle quality

`tests/fixtures/pricing_boundary_references.py` regenerates the cached values
without importing OpenFerric. It was executed with SciPy 1.17.1.

- Single-fixing Asian options reduce to a lognormal payoff observed at fixing
  and discounted to payment. Calls and puts cover positive/negative rates,
  time-zero fixings, delayed payment, and fixing at expiry.
- The three-fixing arithmetic option is integrated by conditioning on its
  middle fixing and evaluating the terminal conditional lognormal payoff.
  Independent QUADPACK references cover the coarse rounded grid and an aligned
  grid. MC comparisons use four reported standard errors plus the independent
  integration budget. The finite integration range `[-12,12]` leaves a
  negligible Gaussian/lognormal tail for these parameters.
- Double-barrier references integrate the killed log-Brownian transition
  density using a 64-mode sine expansion, independently of the production
  image series. Integration is split at the payoff kink. Reported quadrature
  errors are below `1e-12`; the pricing assertions allow `2e-12`. For this grid
  the omitted spectral tail is exponentially smaller than roundoff.
- Credit stubs and zero-strike optionlets use explicit discounted cashflows.
  Greek tests use exact deterministic derivatives and market-input bumps.
- The negative-rate American put and non-dividend American call reduce to
  Europeans. Deep-ITM examples make the tail option value negligible and
  isolate the boundary/carry error. CN's `4e-9` budget includes its second-order
  discounting error and binary64 roundoff; implicit FD uses a first-order
  timestep error budget rather than a percentage-of-price tolerance.

Upstream comparisons used when checking conventions:
[QuantLib's discrete geometric Asian engine](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/pricingengines/asian/analytic_discr_geom_av_price.cpp)
separates fixing-time variance from exercise-date discounting, and
[QuantLib's Black formula](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/pricingengines/blackformula.cpp)
provides the discounted intrinsic and zero-strike reductions. ATM Greek kink
handling is an explicitly documented OpenFerric convention, not a claim that
all packages choose the same convention.

## API and model scope

- `MonteCarloInstrument::control_variate` now takes `steps: usize`. Custom Rust
  implementations/callers must supply the simulation grid. No compatibility
  shim is provided. Python and WASM signatures are unchanged.
- CN vanilla grids remain strike-scaled (`S_max = multiplier * strike`), as
  documented. Deep-ITM inputs may require a larger multiplier; an out-of-grid
  spot now returns an error instead of an endpoint price.
- Asian MC still approximates off-grid observation dates. Its standard error
  does **not** measure that bias; align dates or demonstrate grid convergence.
- A geometric average under additive cash jumps is not generally lognormal.
  The analytic engine now explicitly rejects that unsupported case instead
  of silently using an effective yield.
- The simple CDS spread/annuity helpers remain flat-hazard, midpoint-default
  calculations without the full ISDA dated accrual machinery. Use the ISDA
  implementation for that contract convention.
- Numerical engines still require convergence checks in timestep, domain,
  quadrature and series resolution. This review is not blanket certification
  of every extreme parameter combination or production calibration.
- No line/branch coverage percentage was measured. The assessment is of
  economic oracles and reproduced errors, not a coverage-percentage claim.

## Validation

| Validation | Result |
|---|---|
| `cargo test --locked --workspace --features accelerated-native` | 1,785 passed, zero failures; one existing performance benchmark ignored. |
| `cargo test --locked -p openferric` | 1,655 passed, zero failures; the same performance benchmark ignored. |
| Targeted pricing-boundary regressions | All 15 passed on scalar and accelerated-native configurations. |
| `cargo build --locked -p openferric-python` followed by the complete `python/tests` suite | 252 passed. Tests loaded the newly built debug extension directly with `importlib`, not an older installed wheel. |
| `wasm-pack test --node wasm --locked` | 15 WASM tests passed, including the new zero-volatility Greek ABI regression. |
| `cargo clippy --locked --workspace --all-targets --all-features` | Passed without warnings. |
| `cargo fmt --all --check` and `git diff --check` | Passed. |
| Independent SciPy reference generator | Executed successfully with SciPy 1.17.1. |

Counts overlap across configurations and are not distinct tests to be added
together. The original nine boundary regressions failed before their fixes;
additional dividend-date and negative-rate regressions also failed before
their respective corrections.

Not run here: release-wheel packaging, ARM64 runtime tests, GPU runtime pricing,
JIT execution tests, or SIMD128/threaded WASM variants. All-feature Clippy checks
host compilation, not those runtime behaviors. Existing QuantLib reference
suites were run, but a fresh QuantLib-Python engine was not installed or executed;
new independent numerical values were generated with SciPy.
