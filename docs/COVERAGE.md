# Coverage

Detailed module-by-module coverage of OpenFerric's pricing and analytics library.

## Pricing Validation Standard

Pricing tests use an economic oracle, not a broad plausibility range:

- Analytic and deterministic prices are checked against independently evaluated
  formulas or published package values at floating-point or source precision.
- Numerical quadrature is checked against independent adaptive quadrature and
  integrals are partitioned at payoff discontinuities and kinks.
- Tree, PDE, LSM, FFT, and interpolation tests either lock a stated grid or
  demonstrate convergence to an analytic or external reference. Their tolerance
  represents measured discretization error, not a percentage price band.
- MC and randomized QMC prices are checked against an analytic or independently
  generated target using the estimator's reported standard error (normally four
  standard errors, combining reference and implementation errors where needed).
- Seeded snapshots, sign checks, parity, monotonicity, and no-arbitrage bounds are
  supplemental regression/property tests; none substitutes for a price oracle.

Here, "exact" means the target price is exact for the stated contract and model.
Stochastic and discretized methods necessarily compare to that target with an
explicit statistical or numerical error budget rather than bitwise equality.

The [September 2026 pricing-boundary review](reviews/2026-09-05-pricing-review.md)
records additional defects found despite a passing workspace suite, their
regressions in `tests/audit_pricing_boundaries.rs`, and remaining validation
limits. Its cached SciPy oracles can be regenerated independently with
`python3 tests/fixtures/pricing_boundary_references.py`.

Reference values in this audit were regenerated with Python 3.11.15,
[QuantLib-Python 1.43](https://pypi.org/project/QuantLib/) and
[SciPy 1.17.1](https://docs.scipy.org/doc/scipy/reference/), with
[NumPy 2.4.3 Gauss-Hermite nodes](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermgauss.html)
for Gaussian-factor integration and mpmath 1.4.1 for selected high-precision
closed-form cross-checks. Each cached value is accompanied by its
contract/model parameters;
SciPy Sobol references additionally record replicate counts and reference
standard errors. Correlated first-to-default is also checked against
[FinancePy 1.0.1](https://pypi.org/project/financepy/), with its contract dates,
day count, factor-loading conversion, sample count and reference error recorded
in the test. Lévy-process FFT values use the open-source
[fypy Carr-Madan implementation](https://github.com/jkirkby3/fypy), and product
grids are cross-checked against the upstream
[QuantLib test suite](https://github.com/lballabio/QuantLib/tree/master/test-suite).
`vendor/QuantLib` is citation-only: CI and local tests consume the committed
literals and never build, import, or read that submodule at runtime.
Four-decimal Haug book values are retained only as provenance checks within half
of their last printed digit; every such product also has a full-precision formula,
package value, or stated finite-grid regression target.

## Pricing Validation Map

| Product or methodology | Primary validation suites | Independent oracle |
|---|---|---|
| Vanilla equity, Greeks, FX, Black-76, Bachelier | `strata_black_scholes`, `european_quantlib`, `quantlib_reference`, Python/WASM pricing tests | Strata/QuantLib grids and closed forms |
| Barriers, digitals, lookbacks, compound, chooser, quanto, rainbow, spreads | `barrier_quantlib`, `strata_barrier`, `digital_reference`, `exotic_reference`, `equity_exotics_exact_reference`, `haug_rainbow_spread` | QuantLib/Haug grids and independent SciPy formulas |
| Asian, basket, autocall, range accrual, TARF and convertible products | `asian_quantlib`, `equity_exotics_exact_reference`, `audit_convertible`, DSL tests and focused product tests | QuantLib/formula values, exact discounted cashflows and independently scrambled SciPy Sobol prices/Greeks/termination statistics with combined sampling error |
| Employee stock options and real options | `instruments::employee_stock_option`, `hjm_adjustments_test` and focused `pricing::real_option` tests | Independent 80-digit Decimal CRR recurrences on the stated finite grids, plus analytic reductions and deterministic exercise cases |
| Swing options | `engines::tree::swing` tests | Black-Scholes strip reduction and QuantLib-Python 1.43 `FdSimpleBSSwingEngine` values with tree convergence; `min_exercises=2` has supplemental state-path coverage but is non-binding for the nonnegative payoff |
| Structured notes | focused `instruments::structured_notes` tests | Exact deterministic discounted cashflows and a separately written in-repo full-slice Hull-White recurrence; no independent external callable-note price is claimed |
| American/Bermudan, binomial/trinomial, PDE, LSM | `american_approx_reference`, `bermudan_quantlib`, `hull_white_tree_reference`, `pde_solvers_issue35`, `lsm_reference`, `cross_engine_consistency`, tree module tests | QuantLib, Black-Scholes/CRR, published approximations, Jamshidian's closed form, an independent continuous-state Hull-White Gaussian program, and a local-vol log-price trinomial chain |
| FFT/FRFT, Heston, VG, CGMY, NIG | `fang_oosterlee_heston`, `fft_levy_reference`, `heston_quantlib`, `variance_gamma_model_quantlib`, `models::cgmy`, Python/WASM FFT tests | QuantLib, fypy and Fang-Oosterlee values; independent NumPy high-resolution Carr-Madan grids cross-checked by SciPy Lewis integrals |
| SABR, SVI, Heston, Hull-White and mixture calibration | focused calibration/model tests and Python/WASM vol tests | Exact synthetic quote repricing, identifiable parameter recovery, and solver-conditioned error budgets |
| MC, QMC, SIMD/parallel MC, AAD, rough volatility, SLV and LMM | `cross_engine_consistency`, `models::lmm` and focused engine/model tests | Closed-form BSM/Margrabe/Black-76/moment targets, non-flat SLV repricing and independent SciPy/NumPy Sobol rough-Bergomi and correlated multi-forward LMM grids with reported sampling/calibration error |
| GPU Monte Carlo | `engines::gpu::gpu_mc` tests | CPU-side reduction/layout/shader invariants always run; Black-Scholes and deterministic-payoff price locks run only when a WebGPU adapter is available and otherwise report an explicit skip |
| Bonds, curves, FRA, swaps, OIS/basis, caps/floors, swaptions, XCCY, inflation, CMS | `rates_*`, `rates_capfloor_quantlib`, `strata_bond_reference` and focused rates module tests | QuantLib/Strata values, exact discounted cashflows, QuantLib Black cap/floor and optionlet NPVs, Black-76 reductions, SciPy conditional-lognormal quadrature and QuantLib spread-basket QMC |
| CDS, CDS options/index, ISDA, copulas, first/nth default, CDO | `credit_isda_quantlib`, `credit_quantlib_cds_test`, `cdo_heterogeneous_reference` and focused credit module tests | QuantLib and FinancePy values, exact survival cashflows, SciPy quadrature, direct finite-pool Bernoulli enumeration and distribution formulas |
| Commodity, weather and catastrophe bonds | `commodity_reference`, `commodity_weather_test` and focused instrument tests | Black-76/Kirk, exact Poisson and discounted-cashflow calculations |
| MBS/PSA | focused `instruments::mbs` and `pricing::mbs` tests | Independently evaluated Decimal PSA and documented OTS refinancing-incentive/seasonality cashflow paths under the gross-WAC convention; no low-precision published example is presented as an exact external NPV |
| Funding-rate/perpetual swaps | `funding_swap_reference` and focused funding-rate tests | Independent 80-digit Decimal realised P&L, piecewise-linear funding integral, non-flat discounted MTM and same-bump funding/discount DV01 arithmetic |
| VaR/ES, XVA, KVA/FVA/MVA, margin/liquidation, portfolio sensitivities and statistical primitives | `var_es_reference`, `math::timeseries`, `math::correlation` and focused risk tests | Exact empirical order statistics, Gaussian formulas, discounted cashflows, closed-form margin rules, independent SciPy Sobol first-passage references, Bartlett ACF/PACF bands and chi-square central-moment sampling errors |
| Rust, Python and WebAssembly surfaces | workspace tests, `python/tests`, and `wasm-pack test --node` | The same full-precision references exercised through each binding |

## Known Model Scope

The [all-instrument review](reviews/2026-09-05-all-instruments-review.md)
enumerates every instrument family, the additional rates/credit/commodity/note
findings, API migrations, independent regression evidence and validation limits.

The following are implementation boundaries, not tolerance concessions. Tests pin
the stated model exactly and use reductions or invariants where no like-for-like
external engine exists:

- Supported vanilla engines use escrowed dividend adjustments; jump-aware MC
  applies ex-dividend events explicitly. Analytic single/double barriers and
  exotics, swing/convertible trees and the single-market DSL bridge reject active
  discrete dividends rather than silently smear or discard them. References
  must match the particular engine's dividend model and dates.
- Dated FRA, IRS and cap/floor contracts separate `curve_day_count` from coupon
  accrual conventions. Single/dual range coupons require `accrual_factor`
  independently of payment time. See the all-instrument review for migrations.
- Callable notes and Bermudan swaptions share a centered-OU Hull–White lattice
  with exact grid-date curve fitting. Conditional coupon bonds and off-grid
  events still require convergence checks; rate-dependent call notice is
  explicitly unsupported.
- Geometric Asian analytic pricing discounts to contractual expiry, which may
  follow the final fixing. Proportional dividends are weighted by the fixings
  they affect; cash dividends on or before a fixing are explicitly unsupported
  by that lognormal closed form. Use Monte Carlo for those contracts. Generic
  and dedicated Asian MC round fixings to their uniform simulation grid;
  their geometric controls use that same grid and the correct payment date.
  Reported standard errors exclude observation-date discretization error.
- Black-76 zero-volatility Greeks retain deterministic delta, theta and rho.
  At the ATM kink, delta and gamma are undefined (`NaN`) and vega is the
  right derivative. WASM's finite-output wrapper reports an error at that
  kink rather than returning misleading zero sensitivities.
- CDO pricing includes both the legacy large-homogeneous-portfolio model and a
  finite heterogeneous one-factor Gaussian-copula recursion with per-name
  exposures, recoveries, survival curves and factor loadings. The finite engine
  requires an explicit commensurate loss unit so it never silently rounds name
  losses. Its requested factor-quadrature order is a minimum: the engine
  doubles it until consecutive prices converge and returns an error at the
  safety cap rather than accepting an unchecked near-unit-loading result. Raw
  survival-curve nodes are revalidated at pricing time, including after direct
  Rust or Python mutation. Base-correlation pricing accepts a fixed
  maturity-slice attachment / detachment pair; market-surface calibration and
  bespoke mapping are not inferred by the engine.
- MBS cashflows support the stated
  [SIFMA PSA/CPR prepayment path](https://www.sifma.org/wp-content/uploads/2017/08/chsf.pdf)
  and an explicit deterministic refinancing-rate scenario using the transparent
  OTS refinancing-incentive coefficients, seasoning ramp, and seasonality factors
  shown in the
  [MathWorks modified Richard--Roll example](https://www.mathworks.com/help/fininst/prepayment-modeling-with-a-two-factor-hull-white-model.html).
  MathWorks feeds its pass-through `CouponRate` to that example; OpenFerric uses
  the pool's gross WAC because `coupon_rate - servicing_fee` separately defines
  investor pass-through interest and gross `coupon_rate` drives amortization.
  Scenario pricing accepts monthly refinancing rates and discount yields, and
  scenario duration rebuilds prepayment cashflows under each parallel rate bump.
  The benchmark intentionally omits proprietary borrower calibration, default,
  burnout, and a stochastic OAS rate engine; it is not labeled as the full
  Richard--Roll model. Published SIFMA/Fabozzi examples are too coarsely rounded
  to improve on the existing Decimal cashflow locks, so no external MBS NPV
  anchor is claimed.
- Cross-currency swaps accept explicit fixed- and floating-leg frequencies;
  legacy methods remain annual compatibility wrappers. Quarterly dual-curve
  cashflows and par rates are pinned to independent high-precision Decimal
  sums. The API remains year-fraction based rather than pretending that a dated
  QuantLib calendar/day-count contract is identical.
- Standard dated CDS pricing uses T+1 calendar-day step-in and T+3
  business-day cash settlement. Regular accrual boundaries and payment dates
  are Following-adjusted on the supplied calendar, while contractual maturity
  remains unadjusted. Act/365F curve times, the multi-period final coupon's
  final-day-inclusive Act/360, the one-period exception, half-day
  default-accrual bias, and the accrued rebate are covered by full
  QuantLib-Python 1.43 ISDA-engine fixtures across clean/dirty NPV, both legs,
  rebate, and fair spread, including same-day and holiday-adjusted zero-lag
  settlement. The older unadjusted Act/360 calculation remains available only
  through the explicitly named legacy analytic API in both Rust and Python.
- London, Tokyo and Sydney are pinned to QuantLib's United Kingdom Settlement,
  Japan and Australia Settlement variants, including UK royal one-offs,
  Japan's 2019 succession and 2020/2021 Olympic moves, and Australia's 2022
  mourning holiday. Hong Kong and Singapore include QuantLib's finite
  exchange-published movable-holiday tables for 2019-2026 and 2019-2027;
  outside those table ranges callers must supply a `CustomCalendar`. Full
  weekday holiday-set tests cover each exceptional modern rule family.
- The funding-rate swap payoff is linear in realised rates, so under
  deterministic discounting its value depends only on conditional forward
  means and its volatility sensitivity is mathematically zero. Nonlinear
  funding options require a separate stochastic model. Realised P&L, non-flat
  discounted MTM and both one-basis-point curve sensitivities are recomputed by
  an independent 80-digit Decimal cashflow program.
- `InterestRateFutureQuote::convexity_adjustment` implements the standalone
  Hull approximation `sigma^2 * T1 * T2 / 2`. QuantLib's
  `HullWhite::convexityBias` additionally depends on the futures quote, accrual
  period and mean reversion; even its zero-mean-reversion expression does not
  reduce to this API for the same `(T1, T2)` arguments. The validation map
  therefore does not claim a like-for-like QuantLib Hull-White oracle.
- Liquidation probability and conditional first-passage time are checked
  against independently scrambled SciPy Sobol paths using the exact Gaussian
  Vasicek transition on the contract's monitoring grid. Seeded engine values
  additionally lock path-stream partitioning, while a zero-volatility CIR path
  pins its Euler methodology exactly. A position already below maintenance
  margin is a time-zero liquidation and is never advanced to a later simulated
  passage time.
- General correlated CMS-spread prices are checked against independent SciPy
  conditional-lognormal quadrature and a converged QuantLib-Python 1.43
  low-discrepancy spread-basket engine. Correlated first-to-default is checked
  against SciPy/NumPy Gaussian-factor quadrature and an exactly aligned
  FinancePy 1.0.1 Gaussian-copula contract. Package and implementation sampling
  errors are combined explicitly.
- The Hull-White tree's non-zero-volatility single-exercise swaption converges
  to Jamshidian's closed form and a QuantLib-Python 1.43 value. The library's
  rolling-tenor multi-exercise contract is not the conventional fixed-underlying
  package payoff, so it is checked against an independent continuous-state
  Gaussian dynamic program using the exact joint law of the short-rate factor
  and stochastic discount integral.
- Non-flat stochastic-local-volatility calibration is checked across strikes
  against the exact sampled/interpolated market Black-Scholes surface, with
  separate particle, calibration and pricing uncertainty. Non-zero-eta rough
  Bergomi is checked both by exact conditional-Gaussian quadrature on a two-step
  grid and by an independently assembled eight-step Volterra/Euler Gaussian law
  integrated with replicated SciPy/NumPy Sobol QMC. Andreasen-Huge is
  checked against a non-flat Bachelier-induced Black smile at calibrated nodes
  and off-grid points with the analytic interpolation remainder; the production
  finite-grid node solves and off-grid interpolation outputs are locked
  separately at binary64 operation-roundoff scale.
- The non-zero-vol-of-vol Heston Bermudan has a QuantLib 1.43 finite-difference
  target. Custom non-flat local-vol and time-varying-strike Bermudans are checked
  against an independent recombining log-price Markov chain, with CN convergence
  and LSM reported error tested separately.
- Swing tests reduce unconstrained rights to a strip of Black-Scholes calls and
  converge to a QuantLib-Python 1.43 finite-difference price for the constrained
  monthly contract. A `min_exercises=2, max_exercises=6` case executes the
  constrained state path, but minimum exercise is not economically binding
  because unused rights can be spent for zero payoff; that case is supplemental.
  Real-option defer, expansion, and abandonment trees are checked against
  Black-Scholes reductions, independent high-precision multi-stage CRR
  recurrences and deterministic immediate exercise. Structured
  notes and convertibles pin exact discounted cashflows plus finite-grid
  recurrences for non-zero-volatility callable and fully featured exercise
  cases. The callable-note recurrence is independently written at the lattice
  and event-order layer but deliberately reuses the model's calibrated-theta
  and bond-price primitives, which the separate Jamshidian/QuantLib tests cover;
  exercise/order/no-arbitrage properties remain supplemental. A dated
  QuantLib-Python bridge matched deterministic bond cashflows within 1.5e-14
  and reconciled clean/dirty call semantics, but the 1,200-step non-zero-vol
  OpenFerric tree remained 3.36e-5 from the analytic Hull-White bond-option
  value. The external callable-note lock is therefore deferred rather than
  presented with a false closed-form tolerance.
- GPU Monte Carlo integration tests request a real WebGPU adapter. If none is
  available they print an explicit skip and return; CPU reduction, request
  validation, WGSL parsing and shader validation still run in every build.
- Native and WebAssembly SIMD batch pricers evaluate the tail-accurate Cody
  normal CDF independently per lane while retaining vectorized log, discount
  and payoff arithmetic. Cached SciPy price/Greek grids are checked with an
  explicit binary64/vector-math budget. The separately named
  `normal_cdf_batch_approx` primitive deliberately retains the faster
  Abramowitz-Stegun approximation and has its own approximation-error tests;
  it is not used by the batch Black-Scholes pricing or Greek paths.

## Equity Derivatives

| Model/Product | Module |
|---|---|
| Black-Scholes-Merton | `engines::analytic::black_scholes` |
| Greeks (Δ, Γ, V, Θ, ρ, vanna, volga) | `greeks` |
| American options (CRR binomial) | `engines::numerical::american_binomial` |
| Barrier options (8 types) | `engines::analytic::barrier_analytic` |
| Asian options (geometric + arithmetic MC) | `engines::analytic::asian_geometric`, `engines::monte_carlo` |
| Lookback (fixed + floating strike) | `engines::analytic::exotic` |
| Digital / binary options | `engines::analytic::digital` |
| Double barrier (Ikeda-Kunitomo) | `engines::analytic::double_barrier` |
| Rainbow (best/worst of two, Stulz) | `engines::analytic::rainbow` |
| Power options | `engines::analytic::power` |
| Compound options | `engines::analytic::exotic` |
| Chooser options | `engines::analytic::exotic` |
| Quanto options | `engines::analytic::exotic` |
| Forward start / cliquet | `instruments::cliquet` |
| Variance / volatility swaps | `engines::analytic::variance_swap` |
| Employee stock options | `instruments::employee_stock_option` |
| Convertible bonds | `engines::tree::convertible` |
| Discrete dividend BSM | `pricing::discrete_div` |
| Spread options (Kirk + Margrabe) | `engines::analytic::spread` |

## Volatility

| Model | Module |
|---|---|
| Heston stochastic vol | `engines::fft::carr_madan`, `engines::fft::char_fn` |
| SABR (Hagan 2002) | `vol::sabr` |
| Local vol (Dupire) | `vol::local_vol` |
| SVI parameterization | `vol::surface` |
| Vol smile (sticky strike/delta) | `vol::smile` |
| Vanna-volga method | `vol::smile` |
| Andreasen-Huge (arb-free interpolation) | `vol::andreasen_huge` |
| Fengler (arb-free smoothing) | `vol::fengler` |
| Mixture of lognormals | `vol::mixture` |
| Implied vol solver (Newton-Raphson) | `vol::implied` |
| Vol surface builder | `vol::builder` |

## Rates & Fixed Income

| Product | Module |
|---|---|
| Yield curve bootstrapping | `rates::yield_curve` |
| Bond pricing (dirty/clean, duration, convexity, YTM) | `rates::bond` |
| Interest rate swaps (NPV, par rate, DV01) | `rates::swap` |
| FRAs | `rates::fra` |
| Caps / floors | `rates::capfloor` |
| Swaptions (Black) | `rates::swaption` |
| Cross-currency swaps | `rates::xccy_swap` |
| OIS / basis swaps | `rates::ois` |
| Multi-curve OIS framework | `rates::multi_curve` |
| Inflation swaps (ZC + YoY) | `rates::inflation` |
| CMS spread options | `rates::cms` |
| Futures pricing | `rates::futures` |
| Convexity / timing / quanto adjustments | `rates::adjustments` |
| Day count conventions (ACT/360, ACT/365, 30/360, ACT/ACT) | `rates::day_count` |

## FX

| Product | Module |
|---|---|
| Garman-Kohlhagen | `engines::analytic::fx` |
| FX Greeks (domestic + foreign rho) | `engines::analytic::fx` |
| Black-76 (futures options) | `engines::analytic::black76` |
| Bachelier / normal model | `engines::analytic::bachelier` |

## Credit

| Model | Module |
|---|---|
| CDS pricing (NPV, fair spread) | `credit::cds` |
| Survival curves | `credit::survival_curve` |
| Hazard rate bootstrap | `credit::bootstrap` |
| ISDA standard model | `credit::isda` |
| CDS index pricing | `credit::cds_index` |
| Nth-to-default (Gaussian copula) | `credit::cds_index` |
| CDO tranche pricing (LHP and finite heterogeneous/base correlation) | `credit::cdo`, `credit::heterogeneous_cdo` |
| Copula simulation | `credit::copula` |
| CDS options (Black model) | `credit::cds_option` |

## Structured Products

| Product | Module |
|---|---|
| TARFs (target redemption forwards) | `instruments::tarf`, `pricing::tarf` |
| Range accruals (single + dual rate) | `instruments::range_accrual`, `pricing::range_accrual` |
| Autocallables | `instruments::autocallable`, `pricing::autocallable` |
| MBS pass-through (PSA/CPR and OTS rate-incentive prepayment) | `instruments::mbs` |
| IO/PO strips | `instruments::mbs` |
| WAL, OAS, effective duration | `instruments::mbs` |

## Risk

| Measure | Module |
|---|---|
| Historical VaR | `risk::var` |
| Parametric / delta-normal VaR | `risk::var` |
| Cornish-Fisher VaR | `risk::var` |
| Expected Shortfall (CVaR) | `risk::var` |
| CVA / DVA | `risk::xva` |
| FVA (Funding Value Adjustment) | `risk::fva` |
| MVA (Margin Value Adjustment) | `risk::mva` |
| KVA (Capital Value Adjustment) | `risk::kva` |
| Wrong-way risk (alpha, copula, Hull-White) | `risk::wrong_way_risk` |
| Margin and liquidation first-passage risk (Vasicek/CIR) | `risk::margin`, `risk::liquidation` |
| Portfolio Greeks aggregation | `risk::portfolio` |
| Scenario analysis | `risk::portfolio` |

## Numerical Engines

| Engine | Module | Notes |
|---|---|---|
| Analytic (closed-form) | `engines::analytic` | 15+ engines |
| CRR binomial tree | `engines::numerical` | Up to 1000 steps |
| Trinomial tree | `engines::tree::trinomial` | European + American |
| Generalized binomial (FX/futures/commodity) | `engines::tree::generalized_binomial` | Cost-of-carry parameter |
| Two-asset binomial (Rubinstein) | `engines::tree::two_asset_tree` | Spread/rainbow options |
| Bermudan swaption tree | `engines::tree::bermudan_swaption` | Early exercise |
| Explicit FD (forward Euler) | `engines::pde::explicit_fd` | CFL-constrained |
| Implicit FD (backward Euler) | `engines::pde::implicit_fd` | Unconditionally stable |
| Crank-Nicolson PDE | `engines::pde::crank_nicolson` | European + American |
| Hopscotch | `engines::pde::hopscotch` | Alternating explicit/implicit |
| Longstaff-Schwartz LSM | `engines::lsm` | American MC |
| Monte Carlo (GBM, Heston) | `engines::monte_carlo` | Antithetic + control variate |
| MC Greeks (pathwise + likelihood ratio) | `engines::monte_carlo::mc_greeks` | |
| SIMD Monte Carlo | `engines::monte_carlo::mc_simd` | AVX2 vectorized GBM |
| Parallel Monte Carlo (Rayon) | `engines::monte_carlo::mc_parallel` | Behind `parallel` feature |
| FFT Carr-Madan | `engines::fft::carr_madan` | O(N log N) strike grid |
| Fractional FFT | `engines::fft::frft` | Non-uniform strikes |
| Swing option (DP tree) | `engines::tree::swing` | Energy derivatives |
| Convertible bond tree | `engines::tree::convertible` | Call/put provisions |

## Stochastic Models

| Model | Module |
|---|---|
| Geometric Brownian Motion | `models` |
| Heston | `models` |
| SABR | `models` |
| Hull-White (1-factor) | `models::short_rate` |
| Vasicek | `models::short_rate` |
| Cox-Ingersoll-Ross | `models::short_rate` |
| HW calibration (swaption vols) | `models::hw_calibration` |
| HJM (single + multi-factor) | `models::hjm` |
| LIBOR Market Model (BGM) | `models::lmm` |
| Schwartz (commodity) | `models::commodity` |
| Variance Gamma | `models::variance_gamma` |
| CGMY | `models::cgmy` |
| NIG (Normal Inverse Gaussian) | `models::nig` |
| Rough Bergomi | `models::rough_bergomi` |
| Stochastic local vol | `models::slv` |

## Other

| Feature | Module |
|---|---|
| Energy / commodity derivatives | `instruments::commodity`, `models::commodity` |
| Weather derivatives (HDD/CDD) | `instruments::weather` |
| Catastrophe bonds | `instruments::weather` |
| Real options (defer/expand/abandon) | `instruments::real_option`, `pricing::real_option` |
| FFT characteristic functions (BS, Heston, VG, CGMY) | `engines::fft::char_fn` |
| Fast normal CDF (Hart) | `math::fast_norm` |
| BSM inverse CDF | `math::fast_norm` |
| Bivariate normal CDF | `math` |
| Cubic spline interpolation | `math` |

## Live Market Tools

The WASM-based web dashboard provides live Deribit vol surface visualization.
