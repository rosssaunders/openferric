// Airbag Certificate
// Provides cushioned downside exposure. The investor participates 1:1
// in the upside. On the downside, losses are reduced by the "airbag"
// ratio: the investor's entry point is effectively lowered.
//
// Buffer zone: first 30% of losses are absorbed
// Below buffer: 1:1 participation from the buffer level
// Upside: 1:1 uncapped participation
//
// Example: if perf = 0.60, investor gets 0.60/0.70 = 0.857 (not 0.60)

product "Airbag Certificate 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    schedule annual from 1.0 to 1.0
        let perf = worst_of(performances())

        if perf >= 1.0 then
            // 1:1 upside
            redeem notional * perf
        else if perf >= 0.70 then
            // Within buffer: no loss
            redeem notional
        else
            // Below buffer: airbag ratio
            redeem notional * perf / 0.70
