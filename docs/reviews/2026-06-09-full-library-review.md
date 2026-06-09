# OpenFerric Full-Library Review — 2026-06-09

Scope: all of `src/` (~77k lines) — `engines`, `math`, `mc`, `greeks`, `models`, `vol`,
`rates`, `credit`, `risk`, `core`, `instruments`, `market`, `pricing`, `calibration`, `dsl`.
Each finding was verified by re-reading the surrounding code; several were verified
numerically against published reference values. Findings are listed with file:line,
severity, and the correct formula where applicable.

---

## 1. High-severity bugs

### 1.1 Sobol sequence is not a Sobol sequence beyond dimension 1
`src/math/sobol.rs:108-131` — `build_direction_numbers` builds canonical direction
numbers only for dimension 0 (van der Corput). All other dimensions are filled with
`splitmix64` hashes (masked, forced odd) — there is no primitive-polynomial recurrence,
so cross-dimensional low-discrepancy properties do not exist. The QMC engine consumes
this with `dimensions = n_steps` (`src/engines/monte_carlo/mc_qmc.rs:89`), so the
advertised QMC convergence rate is fictitious; `SOBOL_MAX_DIMENSIONS = 21_201` (the
Joe–Kuo count) is cosmetic.
**Fix:** Joe–Kuo initial values `m_1..m_s` plus the recurrence
`m_j = 2a_1 m_{j-1} ^ 2^2 a_2 m_{j-2} ^ ... ^ 2^s m_{j-s} ^ m_{j-s}`, `v_j = m_j << (64-j)`.

### 1.2 Chooser option analytic formula is wrong (~9% mispricing on the Haug reference case)
`src/engines/analytic/exotic.rs:411-444` — values the simple chooser as
`call(T2) + put(T2)·e^{-q·tau}·N(-d1_choose)`. The correct decomposition
(Rubinstein 1991 / Haug §2.7) is
`V = c(S,K,T2) + K·e^{-rT2}·N(-y2) - S·e^{-qT2}·N(-y1)` with
`y1 = [ln(S/K) + (r-q)T2 + sigma^2·t1/2]/(sigma·sqrt(t1))`, `y2 = y1 - sigma·sqrt(t1)`.
Verified numerically (S=K=50, t1=0.25, T2=0.5, r=0.08, sigma=0.25): code gives 5.5752,
correct is 6.1071. Also fails the t1 → T2 straddle limit. No unit test covers this path.

### 1.3 Wrong-way risk model is a no-op
`src/risk/wrong_way_risk.rs:134-172, 199` — `z_exposure` is a fresh independent normal
unrelated to the sampled exposure path, and the `scale` factor is passed into
`cva_contribution` as `_scale` and ignored. `cva_wwr` and `cva_independent` have
identical expectations for **any** rho. **Fix:** drive the exposure realization and the
default latent variable from the same systematic factor (e.g. rank exposure paths by a
driver Z, set `z_default = rho·Z_path + sqrt(1-rho^2)·eps`).

### 1.4 NIG Monte Carlo: wrong Inverse-Gaussian shape parameter
`src/models/nig.rs:146` — IG subordinator increment over `dt` must be
`IG(mean = delta·dt/gamma_bar, shape = (delta·dt)^2)`, but the code uses
`ig_shape = delta^2 · dt` (missing the square on `dt`). The skew/kurtosis contribution
scales as `dt^2` and vanishes with step count — simulated paths converge to plain
Brownian motion, inconsistent with the NIG characteristic function used for FFT pricing.
**Fix:** `let ig_shape = (self.delta * dt).powi(2);`

### 1.5 Autocallable knock-in redemption not capped at par
`src/pricing/autocallable.rs:552-560` — after a KI breach, redemption is
`notional * (worst_final / ki_strike)` with no cap, so a recovered worst-of pays >100%
of notional (e.g. worst=1.04, ki_strike=1.0, autocall barrier never hit → 104% paid).
**Fix:** `redemption = notional * (worst_final / ki_strike).min(1.0)`.

### 1.6 FX Malz smile interpolator anchors ATM at delta = 0 on the market-delta axis
`src/market/fx.rs:387-393, 434-440` — interpolation axis is market delta (10Δ at 0.10,
25Δ at 0.25), but the ATM anchor is placed at coordinate 0.0 instead of |δ|≈0.5. The
entire ATM region gets RR/BF linearly extrapolated from the 10Δ→25Δ slope
(`vol_at_strike(K_ATM) != atm_vol`), and deep wings interpolate back toward ATM.
**Fix:** place pillars at `0.5 - delta` (25Δ → 0.25, 10Δ → 0.40) and convert market
delta to the smile coordinate in `vol_at_strike`.

### 1.7 Ledoit–Wolf shrinkage intensity too large by a factor of T
`src/math/timeseries.rs:459-477` — `pi_hat` uses `(1/T)·sum_t ||x_t x_t' - S||_F^2`,
but Ledoit–Wolf (2004) requires `1/T^2`. Shrinkage is ~T times too big and clamps to 1
in practice, collapsing the covariance to the scaled-identity target.
**Fix:** divide `pi_hat` by `n_obs` once more.

### 1.8 DSL JIT: `price(idx)` is memory-unsafe
`src/dsl/jit.rs:620-627` — JIT lowers `price(idx)` to `fcvt_to_uint` + raw pointer load
with no bounds check. `price(-1.0)` traps with SIGILL (verified — kills the process);
a too-large index is a silent out-of-bounds read (UB). The interpreter returns a clean
`EvalError` (`src/dsl/eval.rs:1921-1929`). **Fix:** emit a bounds check, fall back to a
trap handler returning the interpreter's error.

### 1.9 DSL JIT: compile panics on any statement after `redeem`/`skip`
`src/dsl/jit.rs:600-618, 708-712` — after REDEEM/SKIP/JUMP the builder switches to a
dead block but never resets `block_terminated`, so the next instruction panics with
"you have to fill your block before switching" (verified with valid DSL:
`redeem N` followed by `pay X`).

### 1.10 DSL parser: stack-overflow abort on adversarial input
`src/dsl/parser.rs:488-706` (plus matching recursion in `compiler.rs:229`,
`eval.rs:1599`) — ~100k nested parens (a 200 KB file) aborts the process (verified).
Reachable from the public `parse_and_compile`, the LSP, WASM, and MCP surfaces.
**Fix:** depth counter in the recursive-descent parser.

---

## 2. Medium-severity bugs

### Rates / credit
- **Floating legs use the continuously-compounded forward instead of the simple forward**
  — `src/rates/swap.rs:87-89`, `src/rates/ois.rs:57-59, 203-205`,
  `src/rates/xccy_swap.rs:65-67`, `src/rates/fra.rs:43-45`. Coupon PV becomes
  `N·ln(DF1/DF2)·DF2` instead of `N·(DF1-DF2)`; float legs systematically underpriced by
  ~r²τ/2 per period (~12 bp at 5% annual), par rates biased low, telescoping identity
  fails. `multi_curve.rs:59` and `capfloor.rs:100` do it correctly — the library is
  internally inconsistent. **Fix:** `(df1/df2 - 1.0)/accrual`.
- **`FixedRateBond::discount_at` treats the continuously-compounded zero rate as an
  m-times-compounded yield** — `src/rates/bond.rs:133-140`. The bond does not reprice
  the curve's own discount factors (~38 bp at z=5%, m=2, t=10y). Contaminates
  dirty/clean price, duration, convexity. **Fix:** use `curve.discount_factor(t)`.
- **FRA cannot represent a forward-starting FRA** — `src/rates/fra.rs:28-46` computes
  `forward_rate(0, tau)` where tau is the accrual length; a 3x6 FRA is unpriceable.
  Needs `forward_rate(t_start, t_end)` and discounting to the proper date.
- **CMS convexity adjustment is not the Hagan/Hull formula** — `src/rates/cms.rs:41-61,
  189-225`. `CA = S²σ²T·duration/annuity` is dimensionally suspect and ~5x too small on
  the module's own test point. **Fix:** `CA ≈ -0.5·S²·σ²·T·G''(S)/G'(S)` with G the
  annuity function (Hull Ch. 30) or Hagan's linear-TSR form.
- **Swap bootstrap discounts pre-first-pillar coupons at DF=1 and never iterates to
  self-consistency** — `src/rates/yield_curve.rs:394-399` with `:678`. First pillar
  absorbs an undiscounted annuity (~5-10 bp error); later pillars extrapolate rather
  than solve jointly, so the curve does not reprice its inputs.
- **Dual-curve bootstrap assumes zero forwards before the first pillar** —
  `src/rates/multi_curve.rs:116-123, 155-157`. Same class of error.
- **`FundingRateCurve::cumulative_index` ignores piecewise-constant interpolation mode**
  — `src/rates/funding_rate.rs:310-342`; always uses trapezoidal areas, inconsistent
  with `forward_rate` for step curves.
- **Margin scalar uses `sqrt(σ·T)` instead of `σ·sqrt(T)`** — `src/risk/margin.rs:143-145`
  (σ=0.16, T=0.25 → 0.20 instead of 0.08); propagates into all liquidation thresholds.

### Models / vol
- **HJM swaption MC mixes risk-neutral drift with T-forward-measure discounting** —
  `src/models/hjm.rs:387-415` (drift at `:227`). Discounting with deterministic
  `P(0,T_e)` requires the T_e-forward drift
  `σ(t,T)[∫_t^T σ ds - ∫_t^{T_e} σ ds]`; otherwise discount pathwise by the bank account.
- **Same measure inconsistency in LMM swaption MC** — `src/models/lmm.rs:171-190` with
  spot-measure drift at `:217-231` but `P(0,T_e)·mean(...)` pricing. Test tolerance of
  2% absorbs the bias.
- **Dupire local vol omits the `(r-q)·K·∂C/∂K` term** — `src/vol/local_vol.rs:49-79`.
  Correct: `σ²_loc = 2[∂C/∂T + (r-q)K·∂C/∂K] / (K²·∂²C/∂K²)`. Biases SLV leverage
  (`src/models/slv.rs:328`) whenever r ≠ q on a skewed surface.
- **Cubic-spline-in-expiry interpolation of total variance can create calendar
  arbitrage** — `src/vol/surface.rs:306-337`; spline overshoot can give negative forward
  variance between knots. Use monotone interpolation in total variance.
- **Andreasen–Huge mixes three inconsistent discounting conventions** —
  `src/vol/andreasen_huge.rs:86-88, 181, 269-307`; recovered vols biased by ≈ `e^{(q-r)t}`
  whenever r ≠ q.
- **"Rough Bergomi" simulates standard fBm, not the Riemann–Liouville Volterra kernel**
  — `src/models/rough_bergomi.rs:296-299, 389-397`; autocorrelation (hence skew term
  structure) differs from the BFG model the parameters are quoted for; spot–vol
  correlation also diluted by correlating against the Cholesky innovation and adding
  independent fine-step noise (`:204-230, :395`).
- **Vanna-volga price does not reprice its own pivot quotes** — `src/vol/smile.rs:427-505`;
  ad hoc moneyness weights instead of vega/vanna/volga-matched weights, so 25Δ pivots
  are not recovered.
- **Storage LSM: double-discounted terminal value + estimate/realized comparison
  mix-up** — `src/models/commodity.rs:1199-1286`; terminal leg discounted twice;
  action choice compares a regression estimate against a realized value (order-dependent,
  not the regression argmax); extrinsic = lsm − intrinsic mixes two PV conventions.

### Engines / pricing / instruments
- **Callable convertible ignores forced conversion** — `src/engines/tree/convertible.rs:50-64`
  uses `min(max(hold, conversion, put), call)`; standard (Hull / Tsiveriotis–Fernandes)
  is `max(conversion, put, min(hold, call))`. With conversion=120, call=110 the code
  returns 110 — systematically underprices callable convertibles.
- **Digital engine reports vega/rho per-1% while all other engines report raw
  derivatives** — `src/engines/analytic/digital.rs:114, 121, 166, 173` vs
  `black_scholes.rs:271-275`, `black76.rs:61`, `fx.rs:85`, `bachelier.rs:115`. Silent
  100x unit mismatch in any cross-engine Greek aggregation.
- **TARF / range-accrual fixing times unvalidated** — `src/instruments/tarf.rs:50-74` +
  `src/pricing/tarf.rs:73-90`; `src/instruments/range_accrual.rs:51-87` +
  `src/pricing/range_accrual.rs:66-81`. Unsorted times give negative `dt`, hence
  `sqrt(dt)` = NaN and silent NaN prices.
- **NaN passes validation throughout `<= 0.0` checks** — `src/market/market.rs:367-404`
  and instrument validators in `vanilla.rs`, `barrier.rs`, `digital.rs`, `asian.rs`,
  `double_barrier.rs`, `fx.rs`, `convertible.rs`, `autocallable.rs`. (black76.rs,
  power.rs, commodity.rs, tarf.rs already use `is_finite()` — inconsistent.)
- **LM optimizer declares step-tolerance convergence on a damped, untested step** —
  `src/calibration/optimizers.rs:220-225`; after rejections, lambda growth shrinks
  ‖δ‖ mechanically and the optimizer reports `StepTolerance` far from an optimum.
  Apply the step test only to accepted steps.
- **`normal_cdf` has unbounded relative error in the tails** —
  `src/math/fast_norm.rs:28-50` (A&S 26.2.17, ~7.5e-8 absolute error → ~25% relative
  at x=-5). Backs copula uniforms (`correlation.rs:554`) and CDO tail integration
  (`src/credit/cdo.rs:275`). Use `0.5·erfc(-x/sqrt(2))` for tail-sensitive work.
- **Finite-difference Greeks divide by the nominal bump after clamping the down-bump**
  — `src/greeks/bsm.rs:172-201`; asymmetric stencil scaled as symmetric near
  sigma→0 / t→0. Use actual spacing / non-uniform 3-point formulas.
- **Zero-vol branch returns undiscounted intrinsic** — `src/math/aad.rs:715-726`,
  `src/math/simd_neon.rs:210-217`; should be `e^{-rT}(S·e^{(r-q)T} - K)^+`.
- **SIMD vs scalar divergences** — inverse-CDF boundary handling differs between
  vector body and scalar remainder (`src/math/simd_math.rs:414-428`,
  `simd_avx512.rs:427-441`): result depends on element position mod 4/8. NEON `ln`
  returns `ln(|x|)` for negatives and finite for 0 (`simd_neon.rs:102-168`) where
  AVX2/AVX-512 blend in NaN/-inf.

### DSL
- **Unbounded schedule-date generation (OOM/hang)** — `src/dsl/ast.rs:78-94`;
  `schedule 1e-9 from 0 to 30` pushes ~3e10 dates with no cap.
- **Declared underlying names and `maturity` builtin unusable in expressions** —
  `src/dsl/compiler.rs:272-291` never consults `scope.underlyings` and omits
  `"maturity"`; `asset(7)` is silently decorative.
- **NaN semantics of `min`/`max` differ across interpreter, AVX2, and JIT backends** —
  `src/dsl/eval.rs:1835-1844, 822-831` vs `jit.rs:483-492, 629-654`; same script can
  price differently per backend/lane.
- **`let` bindings leak out of `if` branches and silently read 0.0** —
  `src/dsl/compiler.rs:180-189` with zero-filled slots (`eval.rs:478`).
- **Trailing input after the product block silently ignored** — `src/dsl/parser.rs:153-156`.

---

## 3. Low-severity bugs (abbreviated)

- Knock-out rebates discounted from maturity instead of hit time —
  `src/engines/monte_carlo/mc_engine.rs:501-513`, `engines/lsm/longstaff_schwartz.rs:710-718`,
  `engines/analytic/double_barrier.rs:299-302`.
- Sample-variance cancellation unguarded → possible NaN stderr —
  `mc_engine.rs:326-330`, `longstaff_schwartz.rs:134-138`, `mc_aad.rs:195-199`
  (`mc_parallel.rs:340` already clamps).
- QMC stderr statistically meaningless without scrambling; early Sobol exhaustion
  biases mean toward zero — `mc_qmc.rs:96-131`.
- Hull-White trinomial clamps negative probabilities instead of adaptive branching —
  `engines/tree/bermudan_swaption.rs:120-144`; drift bias on outer nodes.
- Explicit-FD stability check bounds only the diagonal — `engines/pde/fd_common.rs:139-158`;
  monotonicity also needs off-diagonal coefficients ≥ 0.
- ISDA accrual-on-default in the stub period accrues from step-in instead of period
  start — `src/credit/isda.rs:291-308, 364-366`.
- SIMD `exp` overflows to +inf ~0.34 below the true threshold —
  `simd_math.rs:92-97`, `simd_avx512.rs:137-142`, `simd_neon.rs:91-95`.
- PSD Cholesky floors zero pivots at `sqrt(tol)` instead of 0 — `math/correlation.rs:377-384`.
- LCG stream seeding `base + 7919·i` admits cross-stream collisions —
  `math/fast_rng.rs:203-205`; mix with splitmix64 instead.
- Implied-vol bisection fallback can return an unconverged root as Ok —
  `src/vol/implied.rs:211-232`; same in `vol/smile.rs:237-260`.
- Fengler arbitrage reports wrong strike values (`k.exp()` missing forward;
  `(k*F).exp()` instead of `F·e^k`) — `src/vol/fengler.rs:159, 197`.
- Andreasen–Huge `interpolate_call` brackets on the wrong segment when the nearest
  grid point is above the strike — `andreasen_huge.rs:205-220`.
- MBS u32 underflow when `age > original_term` — `src/instruments/mbs.rs:99`; no
  `validate()` on `MbsPassThrough`.
- HW tree fixes floating/range-accrual coupons off the payment-date short rate
  instead of period start — `src/instruments/structured_notes.rs:1106-1128`.
- DSL: `==`/`!=` absolute-epsilon comparison degenerates to bit-equality above 2.0
  (`eval.rs:1847-1855`); u16 operand truncation on huge scripts (`eval.rs:197-215`);
  non-ASCII mangling in string literals (`lexer.rs:313-323`); LSP panic on non-ASCII
  offsets (`analysis.rs:479, 919-931`); schedule dates past maturity clamped for the
  spot but discounted at the full date (`eval.rs:288-307`).

---

## 4. Performance opportunities (ranked by expected payoff)

1. **`VolSurface::total_variance` rebuilds a `CubicSpline` per query** —
   `src/vol/surface.rs:306-337`. Allocates and runs a tridiagonal solve on every
   `vol()` call, and evaluates every SVI slice. This sits under every MC step and 5x
   per Dupire FD point. Precompute the time-interpolation structure.
2. **LM calibration recomputes the full FD Jacobian after rejected steps** —
   `src/calibration/optimizers.rs:178-188, 250-257`. Each Jacobian column is a full
   Carr-Madan FFT + implied-vol inversions for the Heston calibrator
   (`calibration/heston.rs:101-153`). Standard LM retries with increased lambda on the
   same Jacobian. Several-fold calibration speedup on damping-heavy runs.
3. **ADI Heston allocates 9 Vecs per directional sweep, 18–36 per time step** —
   `src/engines/pde/adi.rs:238-246, 328-337`. Hoist into reusable scratch as
   `crank_nicolson.rs` already does.
4. **SoA SIMD MC stores the full path matrix when only terminal values are read** —
   `src/engines/monte_carlo/mc_simd.rs:48, 180, 255`. 252 steps x 1M paths ≈ 2 GB
   allocated to use 8 MB. Ping-pong two step buffers.
5. **`price_autocallable` always runs the full Greeks ladder** —
   `src/pricing/autocallable.rs:76-85, 159-176`. ~9x simulation work for a 2-asset note
   even when the caller wants only a price. Make sensitivities opt-in.
6. **Basket/autocallable bump-and-reprice redoes PSD repair + Cholesky per bump for the
   identical correlation matrix** — `src/pricing/basket.rs:599-629` (2n+2 calls),
   `src/pricing/autocallable.rs:384-433`. Hoist the factorization out of the bump loop.
7. **HJM swaption MC: two Cholesky factorizations per path of a path-independent
   matrix** (validate + explicit) plus per-(step, maturity) Vec allocations in `drift()`
   — `src/models/hjm.rs:227, 390-397`.
8. **Distribution-fit grid searches recompute loop-invariant `ln_gamma`/`exp` per
   observation** — `src/math/timeseries.rs:558-567, 599-610, 1120-1163`; ~1e6–1e7
   redundant special-function calls; hoisting gives an order of magnitude.
9. **Curve bootstrap rebuilds a full `YieldCurve` (sort + interpolator) per coupon
   date** — `src/rates/yield_curve.rs:673-683` from `:396`; same in
   `multi_curve.rs:165`. O(pillars² x frequency) interpolator builds.
10. **`SurvivalCurve` lookups are O(n) linear scans inside CDS pricing and hazard
    bisection** — `src/credit/survival_curve.rs:103-117, 182-188`; use
    `partition_point`. Bootstrap also re-clones and reprices from t=0 per bisection
    iteration (`credit/bootstrap.rs:39-54`).
11. **`BuiltVolSurface::local_vol` deep-clones the whole surface per evaluation** —
    `src/vol/builder.rs:98-100`; borrow instead.
12. **NS/NSS curve-fit Jacobian refits the entire grid-search calibration per bumped
    node** — `src/math/interpolation.rs:223-244, 1201-1206, 1351-1356`; reuse fitted
    taus and re-solve only the linear betas.
13. **Public DSL `evaluate_product` rebuilds the execution plan per call (per path)** —
    `src/dsl/eval.rs:332-369`; the internal engine compiles once — expose that shape
    publicly.
14. **`fit_sabr` evaluates the full objective twice inside a sort comparator** —
    `src/vol/sabr.rs:245-253`; use `sort_by_cached_key` (~2000 → 150 evaluations).
15. Smaller wins: two-asset tree per-step nested-Vec allocation
    (`engines/tree/two_asset_tree.rs:147`); AVX2 MC horizontal reduction every 4 lanes
    (`mc_parallel.rs:205-237`); LSM per-step `itm`/`discounted` allocations
    (`longstaff_schwartz.rs:354-358, 551-559`); LSM path storage as `Vec<Vec<f64>>`
    (`pricing/american.rs:79-87`, `pricing/bermudan.rs:55-63`); Gauss-Legendre nodes
    recomputed per call (`math/functions.rs:381-396` — reuse the existing `GL96` cache);
    gamma ladder re-evaluating the base price per pillar (`risk/sensitivities.rs:362`);
    rolling stats O(n·window) (`math/timeseries.rs:195-260`); MBS cashflows rebuilt
    ~400x inside the OAS Newton loop (`instruments/mbs.rs:150-210`); string-keyed
    diagnostics on pricing paths vs the `DiagKey` convention
    (`pricing/autocallable.rs:88-90`, `pricing/basket.rs:151-152`); linear-scan grid
    lookups (`market/market.rs:104-124`, `risk/sensitivities.rs:304-323`,
    `vol/andreasen_huge.rs:347-353`).

---

## 5. Areas checked and found correct

Black-Scholes price/Greeks, Conze–Viswanathan and Goldman–Sosin–Gatto lookbacks,
quanto adjustment, Geske compound, Stulz two-asset, Ikeda–Kunitomo double barriers,
Reiner–Rubinstein digitals and barrier case table (incl. in/out parity), Bachelier /
Black-76 / Garman–Kohlhagen, Kirk and Margrabe, CRR/trinomial trees, Crank–Nicolson /
implicit / hopscotch stencils, Carr–Madan and FRFT conventions, Heston/VG/CGMY/NIG
characteristic functions and martingale corrections, LSM discounting chain, pathwise /
likelihood-ratio / AAD Greeks, GPU WGSL reduction; Hagan SABR expansion, Jäckel
inversion, SVI Jacobian, Gatheral butterfly density, VIX replication, Vasicek/CIR/HW
bond formulas, full-truncation Heston Euler, SLV particle timing; Black-76 swaptions /
caplets, CDS legs and `exact_flat_interval_terms`, hazard bootstrap repricing, Vasicek
LHP, Gaussian copula default times, delta-gamma VaR moments, normal ES, historical
VaR/ES indexing, CVA/DVA signs, SA-CCR maturity factor, 30/360 and Act/Act ISDA;
geometric-Asian closed forms, dividend bootstrap, cliquet forward-start, SVI
jump-wings; DSL operator precedence, INDENT/DEDENT handling, per-path scratch reuse in
the engine, SIMD lane-divergence masking; Acklam/Hart norm functions, Lanczos gamma,
PCHIP endpoints, Higham projection, Xoshiro/PCG constants, antithetic/control-variate
accumulators, `PricingArena`.
