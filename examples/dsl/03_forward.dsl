// Equity Forward
// At maturity, pays (S(T)/S(0) - 1) * notional.
// Long exposure to the underlying with no premium.

product "Equity Forward 1Y"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        let perf = worst_of(performances())
        pay notional * (perf - 1.0)
        redeem notional
