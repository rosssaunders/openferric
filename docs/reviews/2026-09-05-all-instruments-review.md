# All-instrument pricing review — 2026-09-05

## Conclusion and scope

This extends the [initial option-heavy review](2026-09-05-pricing-review.md).
Every instrument family declared by `TradeInstrument`, plus the separate rates,
credit, mortgage-strip and commodity-storage contracts, is included below.
The review examines pricing entry points, payoff/cashflow conventions, edge
cases and existing numerical oracles. It does **not** certify that every engine,
parameter combination, calibration or real-world term sheet is correct.

Substantive errors existed outside options despite the previously passing suite.
The patch corrects reproduced errors and adds independent cashflow, limiting-case,
probability and convergence regressions. Where an engine cannot model a contract
feature, it now rejects that feature instead of silently changing the contract.
There is no defensible claim of “perfect” pricing or universal production approval.

## Instrument-by-instrument coverage

“Existing” denotes source review plus the existing numerical/reference suites;
“extended” additionally denotes new regressions or corrected behavioral tests in
this pass. The table enumerates product types, not just engine module names.

| Instruments | Checks and primary evidence | Important scope |
|---|---|---|
| `VanillaOption` — European/American; `BermudanOption` | Extended: `european_quantlib`, `strata_black_scholes`, `american_approx_reference`, `bermudan_quantlib`, `audit_pricing_boundaries`, `audit_equity_extended`, PDE/LSM and cross-engine tests. Price/Greek kernels, exercise, dividends, tails and deterministic limits. | Numerical exercise/grid bias remains; different dividend models must not be mixed. |
| `FuturesOption`, `FxOption` | Initial/extended shared-kernel fixes; Black-76/Garman–Kohlhagen reference grids, zero-volatility Greeks and Python/WASM checks. | Lognormal models require nonnegative forwards; Bachelier is a separate normal-model helper. |
| `CashOrNothingOption`, `AssetOrNothingOption`, `GapOption` | Existing: `digital_reference`, `equity_exotics_exact_reference`, expiry boundary and payoff identities. | Trigger and payment strike differ for gaps; discontinuities do not have ordinary Greeks. |
| `BarrierOption`, `DoubleBarrierOption` | Extended: `barrier_quantlib`, `strata_barrier`, `audit_pricing_boundaries`, `property_invariants`; in/out, rebates, breached barriers and strikes outside the corridor. | Closed forms reject active discrete dividends; appropriate jump-aware numerical methods remain distinct. |
| `AsianOption` — arithmetic/geometric | Initial review: `asian_quantlib`, independent conditional integration, fixing/payment separation and aligned MC control variates. | Geometric cash-jump closed form unsupported; MC observation rounding has bias outside its standard error. |
| `BasketOption`, `OutperformanceBasketOption`, `QuantoBasketOption` | Existing: correlated terminal payoff loops, deterministic and Margrabe reductions, SciPy/Sobol references in `equity_exotics_exact_reference` and module tests. | Moment-matching basket formulas are approximations; MC must use the specified correlation and quanto drift. |
| `Autocallable`, `PhoenixAutocallable` | Extended: observation sequencing, initial knock-in, cashflow dates; existing stochastic/Sobol references and memory-coupon tests. | Discrete monitoring and rounded simulation observations are not continuous barrier pricing. |
| `ForwardStartOption`, `CliquetOption` | Extended finite-input checks; Rubinstein references and single-reset reductions. | `CliquetOption` is an alias for one reset, not a multi-reset ratchet product. |
| `EmployeeStockOption` | Extended finite-input validation; `vol_smile_eso_test`, exact finite-tree references, vesting, exercise multiple and dilution checks. | Expected-life truncation and global attrition discount are a simplified grant model. |
| `LookbackFloatingOption`, `LookbackFixedOption` | Extended dividend guards; existing Haug/QuantLib grids and independent discrete-monitoring Sobol checks. | Analytic monitoring is continuous; observed extrema are explicit; zero-carry evaluation uses a numerical limit. |
| `ChooserOption`, `CompoundOption`, `QuantoOption`, `ExoticOption` | Extended tail handling and dividend guards; `exotic_reference`, full-precision independent formula/quadrature tests, all four compound directions. | Simple chooser; Gaussian quadrature has finite tails; analytic implementations support continuous dividends only. `ExoticOption` dispatches these products rather than adding a payoff. |
| `PowerOption` | Extended: transformed lognormal moments and independently regenerated small-put premium. | Positive exponent/strike lognormal payoff; extreme exponent overflow is not universally guaranteed. |
| `BestOfTwoCallOption`, `WorstOfTwoCallOption`, `TwoAssetCorrelationOption`, `SpreadOption` | Extended: identical-factor rainbow reduction; `haug_rainbow_spread` and `equity_exotics_exact_reference` validate Stulz, conditional triggers, Margrabe and Kirk. | Kirk is approximate; correlation-option triggers have explicit strict inequalities. |
| `VarianceSwap`, `VolatilitySwap` | Extended duplicate-strike rejection; `variance_swap_reference`, finite-strip replication and notional/payoff identities. | Finite-strike approximation; volatility convexity adjustment is first order. Observed variance replaces the full-contract forecast, not a partial accrued slice. |
| `ConvertibleBond` | Extended unsupported-dividend guard; `audit_convertible`, straight-bond/expiry reductions and existing finite-tree references. | Continuous call/put rights, simplified uniform credit-spread discounting; not Tsiveriotis–Fernandes. Only implemented Greeks should be relied on. |
| `SwingOption` | Extended dividend guard; existing exhaustive finite-tree exercise-right and discounted-cashflow references. | Exercise-count contract, distinct from commodity volume constraints. |
| `Tarf` | Extended zero-volatility and invalid-rate checks; `audit_tarf` and independent fixing-strip/truncated-lognormal oracles. | Type-aware KO occurs before payment; target uses full uncapped gain, not final-fixing truncation. |
| `RangeAccrual`, `DualRangeAccrual` | Extended annual accrual/payment-lag identities and validation; existing exact Euler-Gaussian expectation and MC sensitivity references. | Coupon-only claims, equal fixing weights and Euler OU dynamics; explicit accrual factor now required. |
| `CallableRateNote`, `CallableRangeAccrualNote` | Extended notice settlement, simple forwards, zero-volatility reductions and curve-fitted Hull–White lattice; independent full-slice recurrence and par-floater convergence. | Reset-in-advance coupons; rate-dependent coupons earned before a call survive it. Fixed call settlement includes fixed coupons due on that date. Rate-dependent notice is explicitly unsupported. |
| `TargetRedemptionNote`, `SnowballNote`, `InverseFloaterNote`, `CmsLinkedNote` | Extended schedule/NaN validation; existing explicit coupon recursion, target, floor/cap, projected CMS and discounted redemption tests. | Direct projection methods are deterministic scenarios, not expectations of nonlinear payoffs under stochastic future rates. |
| `FixedRateBond` | Extended short-back stubs, accrued interest and difficult YTM brackets; `rates_bond_quantlib`, `strata_bond_reference`, `audit_bond_clean_price`. | Dateless forward coupon schedule; coupon-period metadata is not a full dated market bond convention. YTM is periodically compounded, not continuous. |
| `ForwardRateAgreement`, `InterestRateSwap`, `CapFloor` | Extended separate accrual/curve clocks, off-grid periods and inversion limits; `rates_fra_quantlib`, `rates_swap_test`, `rates_derivatives_test`, `audit_rates_extended`. | Curve valuation origin must match the contract; no historical fixing support for seasoned FRAs. Lognormal optionlets do not silently become normal options. |
| `Swaption` — analytic and Bermudan tree | Extended zero-forward/strike/IV limits and shared curve-fitted lattice; `rates_swaption_quantlib`, `hull_white_tree_reference`, Jamshidian and independent Gaussian dynamic programming. | Bermudan engine starts a new fixed-tenor swap at each exercise, not one co-terminal swap. |
| Zero-coupon bond primitives under `Vasicek`, `CIR`, `HullWhite` | Extended zero-reversion, zero-volatility and long-horizon limits; QuantLib cached bond grids and independent Gaussian/deterministic cashflow quadrature. | Standalone Hull–White theta sampling assumes differentiable forwards; state-price lattice fitting avoids that assumption. |
| `OvernightIndexSwap`, `BasisSwap`, dual-curve IRS pricing | Extended noninteger bootstrap/payment pillars and replacement forward curves; `rates_ois_quantlib`, `rates_xccy_inflation_ois_test`, `audit_rates_extended`. | OIS telescoping assumes the specified unfixed accrual period; separate projection and discount curves matter. |
| `XccySwap` | Existing: both currency legs, principal exchanges, FX conversion and stub/MTM identities in `rates_xccy_inflation_ois_test`. | Its explicit exchange/reset conventions, not every cross-currency term sheet. |
| `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`, `InflationIndexedBond` | Extended original-axis MTM discount ratio and redemption floor; existing CPI-fixing and full cashflow references. | Indexed bond uses deterministic TIPS-style principal protection, not a stochastic deflation-option price or a universal CPI-lag convention. |
| `CmsSpreadOption`, CMS coupon/convexity helpers | Existing: projected swap/annuity definitions, SciPy conditional-lognormal integration and QuantLib spread-basket references. | Convexity and spread distributions remain model assumptions. |
| `Future`, `InterestRateFutureQuote`, `FundingRateSwap` | Existing: cost-of-carry, quote/rate conversion, convexity sign, settlement schedules, APR/8-hour conversion and curve PV identities. | Funding projections are supplied-model expectations; futures/forward convexity helper is approximate. |
| `Cds`, `DatedCds`, `CdsOption` | Extended negative-rate ISDA integrals; initial option-expiry/stub fixes; `credit_quantlib_cds_test`, `credit_isda_quantlib`, independent Simpson cashflow integrals. | Midpoint and standard ISDA date conventions are separate APIs and must be compared like-for-like. |
| `CdsIndex`, `NthToDefaultBasket` | Extended heterogeneous common-spread repricing and default-free premium legs; independent cashflow sums, finite-name enumeration and factor quadrature. | Basket premium accrual remains the documented discrete approximation. |
| `CdoTranche`, `SyntheticCdo`, `HeterogeneousSyntheticCdo` | Extended exact degenerate loss masses; `audit_cdo_hazard`, `cdo_heterogeneous_reference`, finite-name enumeration and independent Gaussian quadrature. | LHP and finite heterogeneous portfolios are different models; base correlation does not imply arbitrary arbitrage-free calibration. |
| `CommodityForward`, `CommodityFutures`, `CommodityOption`, `CommoditySpreadOption` | Extended negative linear futures settlements; `commodity_reference`, Black-76/Kirk and two-factor spread tests. | Linear futures may be negative; positive-domain lognormal option models are not changed into normal models. |
| `CommodityStorageContract`, `VolumeConstrainedSwing` | Extended off-grid initial inventory and grid infeasibility; existing enumerated intrinsic policies and deterministic/MC reductions. | Inventory discretization and in-sample LSM regression bias remain; storage terminal target uses a penalty, not a hard terminal constraint. |
| `WeatherSwap`, `WeatherOption`, `CatastropheBond` | Extended catastrophe coupon stubs; `commodity_weather_test`, exact degree-day/burn cashflows and Poisson loss expectation. | Weather burn values are historical expectation estimates; no inferred market risk premium. Catastrophe intensity/loss assumptions are supplied inputs. |
| `MbsPassThrough`, `IoStrip`, `PoStrip` | Extended all-node spread solve and valid discount bases; existing independent Decimal amortization, prepayment, strip-sum, WAL and refinancing-scenario tests. | Dollar PV, monthly nominal-yield convention; the `oas` routine is a deterministic yield-spread solve, not stochastic mortgage OAS. |
| `DeferInvestmentOption`, `ExpandOption`, `AbandonmentOption`, `RealOptionInstrument` | Existing: source review of exercise cashflows and tree recursions; finite-tree references, abandonment put/cashflow and expansion/defer reductions. | Risk-neutral scenario valuation under the supplied project assumptions; not a corporate investment recommendation. |
| `DslProduct` | Extended rejection of discarded discrete dividends; reviewed contract validation, observation mapping, discounted PAY/REDEEM and termination. Existing DSL examples, deterministic cashflows and scalar/SIMD/parallel equivalence tests. | A programmable payoff language cannot be exhaustively validated over all possible programs; rate assets do not imply stochastic discounting of every cashflow. |

`Trade`, `Portfolio`, the instrument unions, market quotes, curves and prepayment
parameters are containers or supporting inputs, not additional payoff types.
Curve/bootstrap paths affecting the products above were reviewed; this is not an
exhaustive independent audit of all calibration, risk/XVA or numerical utilities.

## Material findings corrected in this pass

Severity denotes pricing impact, not test count. Initial-pass fixes remain in the
linked earlier report and are not repeated as newly discovered findings here.

| Severity | Defect | Correction and regression |
|---|---|---|
| P1 | Full coupons or rounded-away periods on short maturities/stubs: bonds, catastrophe bonds, IRS and curve bootstraps. | Cashflows stop at true maturity with proportional accrual. `audit_rates_extended` and `audit_other_assets` cover 0.1/1.1-year cases and noninteger calibration pillars. |
| P1 | FRA, IRS and cap curves used coupon accrual conventions as their time axis; FRA also derived end time by adding an accrual fraction. | Explicit `curve_day_count`; both curve endpoints measured from one origin. Independent mixed-clock discount/forward cashflows and Python tests. |
| P2 | Bond Newton YTM solve clamped compounding bases and could fail for admissible difficult yields. | Bracket/bisect in log-compounding space; tests include yields near minus coupon frequency and large positive yields. |
| P1 | Replacing a tenor forward curve retained the old curve; swap DV01 also rebuilt curves using a different interpolator. | Replace matching tenor, preserve interpolation settings during bumps; explicit replacement and non-flat DV01 regressions. |
| P1 | Inflation MTM discounted remaining time on an original-axis curve; indexed principal could redeem below the stated floor. | Use `DF(maturity)/DF(valuation)` and protect redemption only, not coupons. Nonflat forward discount and deflation regressions. |
| P1/P2 | Swaption and implied-volatility helpers mishandled valid zero forward/strike limits or impossible/nonfinite premiums. | Stable limiting Black cashflows; invalid inversion targets return `NaN` instead of plausible zero volatility. |
| P1 | Vasicek zero-reversion and CIR zero-volatility bond limits failed, including `NaN`; CIR's unscaled exponential form also overflowed at long horizons with representable bond PVs. | Gaussian integrated-rate moments with a small-reversion series; deterministic CIR limit and scaled log-affine pricing. Independent Simpson integration and existing QuantLib values remain separate oracles. Bermudan exercise at time zero also returns swap intrinsic rather than `NaN`. |
| P1 | CDS index “fair spread” averaged constituent spreads rather than solving the common running coupon. | Weighted protection divided by weighted risky annuity. Independently assembled heterogeneous cashflows reprice common-coupon index NPV to zero. |
| P1 | Default-free nth-to-default basket returned zero NPV even with premiums owed. | Retain the premium annuity when protection is zero; explicit discounted short-stub premium strip. |
| P1 | Underflowed zero survival nodes were discarded, turning distressed names into default-free names. | Retain their tenor at the documented log-probability floor; expected LGD regression. This does not remove that numerical floor. |
| P2 | Exact Gaussian factor loadings and LHP probability endpoints were approximated, losing degenerate probability masses. | Exact comonotonic simulation, zero/default-certain loss cases and correlation-one tranche reductions. |
| P1 | ISDA default integrals clamped negative combined discount/hazard rates; one Taylor branch treated every negative exponent as small. | Signed stable exponential moments and an absolute small-exponent test, checked by independent Simpson quadrature. |
| P1 | MBS spread solve used only the first yield-curve point; invalid monthly discount bases could produce plausible prices. | Use each monthly spot yield plus the solved spread; reject invalid inputs across pass-through and IO/PO pricing. Two-month cashflows independently determine the spread. |
| P1 | Linear commodity futures rejected zero/negative settlement or contract prices. | Permit all finite linear prices while retaining positive size. Explicit long/short marked cashflows. |
| P1 | Storage LSM rounded initial inventory to a grid point; a valid volume swing could return `Ok(-infinity)` on an infeasible grid. | Interpolate the initial storage value consistently and return an explicit grid-infeasibility error. The 150-unit deterministic storage example no longer prices as 100 units. |
| P1 | Callable notice moved payment to the decision date rather than the contractual settlement date. | Conditional PV of settlement plus intervening fixed coupons; zero/nonzero-volatility tests. Rate-dependent notice requires augmented state and now errors explicitly. |
| P1 | Floating and inverse-floating tree coupons used the instantaneous short rate, not the period's simple forward. | Project from the conditional period bond and divide by accrual; par-floater and independent coupon-branch regressions. |
| P1 | Hull–White trees differentiated discontinuous curve forwards and clipped transition moments. A nonflat par floater priced at 1,002,951.719 instead of 1,000,000. | Shared centered-OU lattice with exact one-step moments and grid-date Arrow–Debreu curve fitting. Every tested zero bond fits to roundoff; the 400-step stochastic floater residual is about 0.203 currency units and decreases with refinement. Zero-volatility PV is exact. Both notes and Bermudan swaptions use it. |
| P1 | Annualized range coupons omitted their accrual year fraction. An 8% quarterly coupon on 100 with six-month payment lag priced at 7.8808955 rather than 1.9702239. | Required explicit `accrual_factor` on single/dual contracts, distinct from payment time; Rust, Python, serialization and MCP callers updated. |
| P1/P2 | Snowball NaNs became zero through payoff flooring; schedules/model parameters and some option fields accepted nonfinite values. | Validate before payoff transforms; ordered nonoverlapping coupons, finite rate/model fields, and deterministic TARF support. |
| P1 | Multiple autocall observations rounding to one simulation step lost earlier events; initial fixing was omitted from knock-in monitoring. | Retain all contractual observations and initialize barrier state from initial spots. Exact deterministic cashflow regressions. |
| P2 | Recovering small puts via put-call parity lost meaningful relative precision, including power-option and scalar/FMA BSM paths. | Evaluate the requested tail directly; reuse the central vanilla kernel for barrier reductions and compound/chooser helpers. Independent SciPy premium, price-with-Greeks and exact breached-barrier identities. |
| P1 | Identical-volatility, perfectly correlated best/worst-of calls rejected a valid deterministic ordering. | Exact reduction to the larger/smaller discounted vanilla, with stable effective variance. |
| P2 | Interior duplicate strikes escaped variance-strip validation. | Reject every duplicate before integration; specifically test duplicates away from strip endpoints. |
| P1 | Several path-dependent engines silently smeared discrete dividends, or the DSL bridge discarded them. | Active dividends now error in analytic single/double barriers and exotics, swing/convertible trees, and the single-market DSL bridge. Cash/proportional tests also verify that post-maturity events do not cause rejection. |

### Why the previous tests missed these

- Whole-year schedules hid stub errors, and identical accrual/curve conventions
  hid time-axis mistakes.
- Homogeneous credit portfolios hid the difference between averaging spreads
  and pricing one common coupon.
- Grid-aligned inventory/observation examples missed rounding losses.
- Reimplementing a finite-grid recurrence can reproduce its *model error*.
  The Hull–White audit therefore adds exact curve-fit and continuous-model
  par-floater checks, not just an updated grid snapshot.
- Broad absolute tolerances hide lost tail premiums. Tail tests use relative
  precision against independently evaluated negative-normal-CDF formulas.

The heterogeneous CDS-index reference changes because the quoted quantity now
means a common par coupon. Floating-note references change from short-rate
coupons to contractual simple forwards. For the changed Hull–White lattice,
the multi-exercise swaption test refines from 1,200 to 2,400 steps rather than
widening its independent Gaussian-DP error budget. No barrier identity assertion
was weakened to accommodate the changed BSM arithmetic.

## API migrations

- Rust struct literals for `ForwardRateAgreement`, `InterestRateSwap` and
  `CapFloor` require `curve_day_count`. The IRS builder defaults to `Act365Fixed` and
  exposes a setter. Python constructors expose an optional explicit clock,
  also defaulting to `Act365Fixed`; they do not infer it from coupon accrual.
- Rust/Python/serialized `RangeAccrual` and `DualRangeAccrual` require
  `accrual_factor`. For an 8% annual coupon over a quarter, use `0.08` and
  `0.25`; a delayed payment does not extend the accrual. Python adds the factor
  as the final constructor argument. MCP's start-at-zero range tool uses its
  `time` as the accrual factor. No compatibility shim is added.
- The earlier change to `MonteCarloInstrument::control_variate(..., steps)`
  remains necessary; custom Rust implementations must supply the grid.
- Previously accepted unsupported dividend/notice inputs now return errors.
  Corrected stubs, common CDS spreads, annual coupons, floating coupons and
  curve-fit behavior intentionally change previously incorrect prices.

## Reference provenance

Most new tests assemble discounted cashflows or limiting distributions directly,
without calling the production pricing formula to generate the expected value.
`tests/fixtures/all_instrument_references.py` regenerates the additional Gaussian
tail oracle with SciPy; the earlier reference generator is retained separately.
The new lattice tests explicitly match OU conditional moments and price unit
zero-bond payoffs at every tested grid date. Nonlinear note branches also retain
a separately implemented full-slice recurrence; that is implementation evidence,
not an independent certification of the economic model.

- Fractional coupons and separated curve/accrual clocks were compared with
  [QuantLib fixed-rate coupons](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/cashflows/fixedratecoupon.cpp)
  and [forward-rate agreements](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/instruments/forwardrateagreement.cpp).
- The fitted-state-price approach and centered moment matching are consistent
  with [QuantLib's Hull–White lattice construction](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/models/shortrate/onefactormodels/hullwhite.cpp)
  and [trinomial tree](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/methods/lattices/trinomialtree.cpp).
  The new code is derived from the OU moments/discount identities, not a copied
  external implementation. Continuous-model Jamshidian and Gaussian-DP tests
  remain separate from finite-grid recurrence tests.
- The indexed-bond floor follows the stated
  [TreasuryDirect TIPS maturity protection](https://www.treasurydirect.gov/marketable-securities/tips/).
  This is a contract convention, not an assertion that a deterministic floor
  includes stochastic deflation-option time value.
- Negative linear futures prices are supported by
  [CME's negative-price advisory](https://www.cmegroup.com/notices/clearing/2020/04/Chadv20-160.html).
  This does not justify using a positive-domain Black formula for negative
  option underlyings.

## Validation

| Check | Final result |
|---|---|
| `cargo test --locked --workspace --features accelerated-native` | 1,826 passed; zero failures; one existing performance benchmark ignored. |
| `cargo test --locked -p openferric` | 1,696 passed; zero failures; the same benchmark ignored. |
| Five new all-instrument integration suites | 39 regressions passed in both configurations, in addition to the retained initial-pass suite. |
| Shared Hull–White lattice unit tests | Exact OU moment matching and every tested grid-date zero bond pass, including nonflat/negative-rate curves and zero volatility/reversion. |
| Fresh `openferric-python` debug extension + complete `python/tests` | 264 passed, including 12 new binding tests. Loaded the newly built shared library through `importlib`, not a previously installed wheel. |
| `wasm-pack test --node wasm --locked` | 15 WASM runtime tests passed. |
| `cargo test --locked -p openferric --features jit dsl::jit::` | 36 targeted DSL JIT tests passed. This filter is not a second full test-suite run. |
| `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` | Passed without warnings. |
| `cargo fmt --all --check` / `git diff --check` | Passed. |
| `python3 tests/fixtures/all_instrument_references.py` | Regenerated the SciPy tail reference successfully. |
| Declared-instrument inventory | All 56 `TradeInstrument` variants appear in the coverage table, including aliases/unions, plus the standalone products listed there. |

Counts overlap across configurations; they are not disjoint test counts.
Rust toolchain: `rustc 1.94.0 (4a4ef493e 2026-03-02)`. Local QuantLib reference
submodule: `ae25d2846d61c0db38bc701a3475ca5678812554`.

The first full rerun caught a bitwise barrier/vanilla identity difference after
the tail fix; the duplicated vanilla helper was replaced with the common kernel,
not a looser assertion. Deterministic barrier cashflows retain their explicit
terminal-payoff evaluation to preserve the existing roundoff-level cashflow test.

## Remaining production limitations

- This is a finite source/test audit, not a proof of perfection. No line/branch
  coverage percentage was measured, and no claim is made that all combinations
  of products, models and hardware backends were executed.
- Cached QuantLib/Strata/FinancePy and other external references are exercised;
  a fresh QuantLib-Python installation was not used to regenerate every price.
- MC standard errors exclude timestep, monitoring, inventory-grid, regression,
  calibration and model-selection bias. LSM requires out-of-sample/convergence
  analysis before using its standard error as an accuracy statement.
- Small-probability floors remain in survival/credit numerics. A stored survival
  floor of `1e-12` cannot retain an arbitrarily large underlying hazard or its
  exact default timing. Extreme floating-point overflow and all possible
  invalid mutations of public structs are not universally covered.
- Legacy `f64` APIs have differing invalid-input conventions (`NaN`, zero or
  checked wrappers). A library-wide typed-error migration is not part of this
  patch. Implied-volatility solvers retain their documented finite brackets.
- The product-specific restrictions in the coverage table matter: simplified
  mortgage OAS, projected inflation/CMS, ESO attrition, rolling-tenor Bermudans,
  single-reset cliquets, earned-coupon call semantics and historical weather
  expectations are not substitutes for richer market contracts/models.
- Release wheel packaging, ARM64 execution, GPU pricing and SIMD128/threaded
  WASM runtime coverage require separate environment-specific checks. Host
  all-feature Clippy is compilation/lint evidence, not hardware validation.
- Independent desk/model validation against actual term sheets and a larger
  generated parameter/convention matrix remains appropriate before production.
