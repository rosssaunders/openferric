// Accumulator (TARF — Target Accrual Redemption Forward)
// Monthly observations. When the spot is within a range, the investor
// accumulates units at a discounted price. Once the total accumulated
// profit hits a target, the structure terminates.
//
// Range: 90% – 110% of initial spot
// Accumulation per period: notional * 0.0833 * (perf - 1.0)
// Target cap: 10% of notional

product "Accumulator TARF 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        accumulated_profit: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if accumulated_profit < 0.10 then
            if perf >= 0.90 and perf <= 1.10 then
                let period_pnl = 0.0833 * (perf - 1.0)
                set accumulated_profit = accumulated_profit + max(period_pnl, 0.0)
                pay notional * period_pnl

        // Target hit — knock out
        if accumulated_profit >= 0.10 and not is_final then
            redeem notional

        if is_final then
            redeem notional
