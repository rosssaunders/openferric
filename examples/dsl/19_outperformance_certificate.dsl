// Outperformance Certificate
// Leveraged participation in the upside above the strike.
// Below strike, investor participates 1:1 in the downside.
//
// Strike: 100%
// Upside participation: 150%
// No cap, no capital protection

product "Outperformance Certificate 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    schedule annual from 1.0 to 1.0
        let perf = worst_of(performances())

        if perf >= 1.0 then
            // Leveraged upside
            let upside = (perf - 1.0) * 1.50
            redeem notional * (1.0 + upside)
        else
            // 1:1 downside
            redeem notional * perf
