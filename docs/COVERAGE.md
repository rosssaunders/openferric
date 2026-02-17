# Coverage

Detailed module-by-module coverage of OpenFerric's pricing and analytics library.

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
| Heston stochastic vol | `engines::analytic::heston` |
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

| Tool | Binary | Description |
|---|---|---|
| Deribit vol surface snapshot | `deribit_vol_surface` | REST fetch → SVI calibration → 3D Plotly HTML |
| Live vol surface dashboard | `vol_dashboard` | WebSocket stream → real-time recalibration → browser dashboard |
