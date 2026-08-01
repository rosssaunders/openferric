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

Reference values in this audit were regenerated with
[QuantLib-Python 1.43](https://pypi.org/project/QuantLib/) and
[SciPy 1.17.1](https://docs.scipy.org/doc/scipy/reference/). Each cached value is
accompanied by its contract/model parameters;
SciPy Sobol references additionally record replicate counts and reference
standard errors. Lévy-process FFT values use the open-source
[fypy Carr-Madan implementation](https://github.com/jkirkby3/fypy), and product
grids are cross-checked against the upstream
[QuantLib test suite](https://github.com/lballabio/QuantLib/tree/master/test-suite).
Four-decimal Haug book values are retained only as provenance checks within half
of their last printed digit; every such product also has a full-precision formula,
package value, or stated finite-grid regression target.

## Pricing Validation Map

| Product or methodology | Primary validation suites | Independent oracle |
|---|---|---|
| Vanilla equity, Greeks, FX, Black-76, Bachelier | `strata_black_scholes`, `european_quantlib`, `quantlib_reference`, Python/WASM pricing tests | Strata/QuantLib grids and closed forms |
| Barriers, digitals, lookbacks, compound, chooser, quanto, rainbow, spreads | `barrier_quantlib`, `strata_barrier`, `digital_reference`, `exotic_reference`, `equity_exotics_exact_reference`, `haug_rainbow_spread` | QuantLib/Haug grids and independent SciPy formulas |
| Asian, basket, autocall, range accrual, TARF, swing, convertible, real options, structured notes | `asian_quantlib`, `equity_exotics_exact_reference`, DSL tests and focused module tests | QuantLib/SciPy Sobol values, analytic reductions, deterministic cashflows |
| American/Bermudan, binomial/trinomial, PDE, LSM | `american_approx_reference`, `bermudan_quantlib`, `pde_solvers_issue35`, `lsm_reference`, `cross_engine_consistency`, tree module tests | QuantLib, Black-Scholes/CRR, published approximations, deterministic Hull-White reduction |
| FFT/FRFT, Heston, VG, CGMY, NIG | `fang_oosterlee_heston`, `fft_levy_reference`, `heston_quantlib`, `variance_gamma_model_quantlib`, Python/WASM FFT tests | QuantLib, fypy, Lewis and Fang-Oosterlee values |
| SABR, SVI, Heston, Hull-White and mixture calibration | focused calibration/model tests and Python/WASM vol tests | Exact synthetic quote repricing, identifiable parameter recovery, and solver-conditioned error budgets |
| MC, QMC, SIMD/parallel MC, AAD, rough volatility and SLV | `cross_engine_consistency` and focused engine/model tests | Closed-form BSM/Margrabe/moment targets with reported sampling error |
| Bonds, curves, FRA, swaps, OIS/basis, caps/floors, swaptions, XCCY, inflation, CMS | `rates_*`, `strata_bond_reference` and focused rates module tests | QuantLib/Strata values, exact discounted cashflows, Black-76 reductions |
| CDS, CDS options/index, ISDA, copulas, first/nth default, CDO | `credit_isda_quantlib`, `credit_quantlib_cds_test` and focused credit module tests | QuantLib values, exact survival cashflows, SciPy quadrature and distribution formulas |
| Commodity, weather, catastrophe bonds, MBS/PSA and funding swaps | `commodity_reference`, `commodity_weather_test` and focused instrument tests | Black-76/Kirk, exact Poisson/cashflow calculations, SIFMA PSA formulas |
| VaR/ES, XVA, KVA/FVA/MVA and portfolio sensitivities | `var_es_reference` and focused risk module tests | Exact empirical order statistics, Gaussian formulas, discounted exposure/capital cashflows, and reported sampling error |
| Rust, Python and WebAssembly surfaces | workspace tests, `python/tests`, and `wasm-pack test --node` | The same full-precision references exercised through each binding |

## Known Model Scope

The following are implementation boundaries, not tolerance concessions. Tests pin
the stated model exactly and use reductions or invariants where no like-for-like
external engine exists:

- Discrete cash and proportional dividends use escrowed spot/strike adjustments
  in analytic, tree, and PDE engines and explicit ex-dividend path jumps in
  Monte Carlo engines. References align the dividend dates and cashflows with
  QuantLib's escrowed model before comparing prices.
- CDO pricing is the large-homogeneous-portfolio Gaussian-copula model. General
  heterogeneous base-correlation tranches are outside the current engine.
- MBS cashflows use the stated
  [SIFMA PSA/CPR prepayment path](https://www.sifma.org/wp-content/uploads/2017/08/chsf.pdf)
  and a flat discount yield; the model does not make prepayments interest-rate
  dependent.
- Cross-currency swap reference cases use the engine's annual coupon periods,
  rather than silently comparing them with QuantLib's common quarterly setup.
- Dated CDS schedules use the library's weekends-only calendar unless an explicit
  business calendar is supplied; QuantLib TARGET-calendar references are adjusted
  to the same dates before comparison.
- The funding-rate swap has no volatility state, so its reported volatility
  sensitivity is exactly zero by construction.
- General correlated CMS-spread, first-to-default, and non-zero-volatility
  Bermudan-swaption cases have no identical external contract/model fixture in
  the current reference set. They are covered by exact one-factor or deterministic
  reductions, independently evaluated cashflows, and convergence/statistical
  checks; broader model equivalence is not claimed.
- Non-flat stochastic-local-volatility and non-zero-eta rough-Bergomi prices do
  not yet have like-for-like external package grids. Coverage uses exact
  Black-Scholes/constant-leverage reductions, analytic covariance identities,
  and reported Monte Carlo errors. Andreasen-Huge similarly has exact quote-node
  repricing and off-grid interpolation budgets, but no external non-flat grid.
- Structured-note, swing, convertible, and real-option tests use exact limiting
  cases plus exercise/order/no-arbitrage properties where the non-trivial model
  has no matching independent package implementation.
- The WebAssembly SIMD batch pricer uses `f64x2`, but deliberately retains its
  Abramowitz-Stegun normal-CDF approximation.  Tests separate binary64 SIMD
  roundoff from that approximation and pin its SciPy-grid maxima (about
  `1.3e-5` in price, `7.2e-8` in delta, and `3.3e-7` in theta); scalar binding
  prices use the higher-accuracy analytic path.

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
| CDO tranche pricing (LHP) | `credit::cdo` |
| Copula simulation | `credit::copula` |
| CDS options (Black model) | `credit::cds_option` |

## Structured Products

| Product | Module |
|---|---|
| TARFs (target redemption forwards) | `instruments::tarf`, `pricing::tarf` |
| Range accruals (single + dual rate) | `instruments::range_accrual`, `pricing::range_accrual` |
| Autocallables | `instruments::autocallable`, `pricing::autocallable` |
| MBS pass-through (PSA/CPR prepayment) | `instruments::mbs` |
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
