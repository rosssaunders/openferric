# Pricing audit: contracts, lifecycle and numerical error

Passing tests is evidence for the stated contracts and models, not a guarantee
that every market convention or economic model is correct for every trade.

## Contract corrections

- Callable range-accrual coupons observe short rates daily on an ACT/365F grid,
  with actual-length final stubs. Earned observations survive subsequent calls;
  same-time calls precede new observations. Payment dates are unchanged. The
  rate lattice uses cell-average indicator smoothing, not a single coupon reset.
  A business-day/calendar observation convention is not inferred from floats.
- `BermudanSwaptionEngine.price` uses one co-terminal annual-pay swap starting
  at `option_expiry`. Exercise must be on its reset dates and the lattice grid.
  Unsupported dates return NaN. `price_rolling_tenor` explicitly prices the
  different contract entering a fresh fixed-tenor swap at each exercise.
- `HullWhiteCalibrator::new(curve)` requires a valuation curve. Quotes are ATM
  absolute normal volatilities for physical, annual-pay, single-curve swaptions.
  It fits Jamshidian prices converted to that quote convention, not frozen-weight
  volatility approximations. Negative rates and negative strikes are supported.

## Stochastic rates and lifecycle

- `HullWhite::simulate_path` samples the centered OU factor jointly with its
  exact time integral. Money-market discount factors fit the valuation curve
  without an Euler discounting approximation. Contractual event dates are exact.
- TARN and snowball `price_hull_white_mc` evaluates nonlinear coupon and target
  recursions on stochastic paths. Existing `price(projected_rates, curve)` is
  explicitly a deterministic scenario, not a risk-neutral optionality price.
- `RateNoteHistory` keeps contract times on the original time axis; the supplied
  curve starts at `valuation_time`. Historical reset fixings reconstruct the
  accumulated target and snowball coupon. Payments on valuation date are
  considered settled. Missing fixings fail rather than being replaced by today's
  forwards. Already settled trades have zero remaining value.
- Callable notes accept historical underlying fixings through
  `price_hull_white_tree_with_history`. The trade must still be outstanding;
  an already delivered call notice needs its recorded settlement claim.
  Rate-dependent notice periods remain explicitly unsupported by this lattice.
- FRA `npv_with_fixing` distinguishes advance and arrears settlement and whether
  valuation-date payment has occurred. An earlier barrier knock-in remains
  vanilla; an earlier knock-out with settled rebate has zero remaining value
  through the Black-Scholes `price_with_history` entry point.
- MBS `price_hull_white_mc` and `oas_hull_white` couple monthly prepayment to
  simulated refinancing rates and stochastic discount factors. Refinancing uses
  a specified model zero-yield tenor, mortgage spread and floor. OAS is a
  continuously compounded discount spread, solved with common random paths.
  The former deterministic `oas` is now `z_spread` (nominal monthly convention).
  The OTS-style prepayment model is not proprietary borrower/burnout calibration.

## Error accounting and independent tests

LSM trains on one sample and prices the frozen stopping policy on an independent
sample, including the convenience American/Bermudan functions. `training_paths`
is separate from `num_paths`. Vanilla continuation uses a centered, scaled cubic
basis to reduce regression conditioning and approximation bias. Its standard
error is conditional on the policy:
it does not measure policy suboptimality, exercise-grid, local-vol/Heston Euler,
calibration, floating-point backend, or economic-model error. Time/state/path
refinement and multiple seeds remain necessary for application tolerances.
Heston path generators now retain the full-truncation auxiliary variance across
steps instead of repeatedly projecting it to zero. `step_full_truncation`
exposes that scheme; `step_euler` remains explicitly projected Euler. A negative
auxiliary state is carried forward, not interpreted as negative physical variance.
This removes the extra boundary bias exposed by the independent-policy test.

JSON parsing enables exact binary64 round trips in Rust and Python. Adversarial
finite values and simulated discount factors are checked bit-for-bit, so a
serialization path cannot silently perturb pricing inputs by an ulp.

`tests/pricing_correctness_gaps.rs` checks analytic lifecycle cashflows, exact
Gaussian daily coupons, a QuantLib callable bond-option price, independent
SciPy nonlinear-note and MBS prices, and stochastic discounting identities.
`tests/hull_white_tree_reference.rs` separates co-terminal and rolling contracts.
The HW calibrator's committed QuantLib 1.43 quotes are independent of the fitted
implementation. No test imports QuantLib or reads `vendor/QuantLib` at runtime.
Python price/Greek tests exercise the new entry points, not just serialization.

The nonlinear-note Sobol references use SciPy 1.17.1, 2^20 points and independent
scrambles 314159/271828, eight joint-Gaussian OU/integral coordinates, four
semiannual coupons, flat 3%, a=.1 and sigma=.03. Their discrepancy is added to
the implementation's sampling budget. The MBS quadrature reference has three
remaining payments: only the second month's prepayment is stochastic; its
cashflows and final balance integrate separately under their payment measures.
