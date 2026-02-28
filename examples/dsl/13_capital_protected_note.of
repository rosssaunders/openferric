// Capital Protected Note (CPN)
// 100% capital protection at maturity plus participation in upside.
// Investor never loses principal, but upside is capped or has
// reduced participation to fund the protection.
//
// Protection: 100%
// Upside participation: 60%
// Cap: 130% of notional total return

product "Capital Protected Note 2Y"
    notional: 1_000_000
    maturity: 2.0

    underlyings
        SPX = asset(0)

    schedule annual from 2.0 to 2.0
        let perf = worst_of(performances())

        // Upside participation, capped
        let upside = max(perf - 1.0, 0.0) * 0.60
        let capped_upside = min(upside, 0.30)

        pay notional * capped_upside
        redeem notional
