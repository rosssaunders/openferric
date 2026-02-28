// Express Certificate
// Similar to an autocallable but with a growing redemption premium.
// Each observation date, if the underlying is above par,
// the product redeems at an increasing premium.
//
// Period 1 (6m): redeem at 103%
// Period 2 (12m): redeem at 106%
// Period 3 (18m): redeem at 109%
// Period 4 (24m, final): redeem at 112% or at performance if KI hit
//
// KI barrier: 65%

product "Express Certificate 2Y"
    notional: 1_000_000
    maturity: 2.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false

    schedule semi_annual from 0.5 to 2.0
        let perf = worst_of(performances())

        if perf <= 0.65 then
            set ki_hit = true

        // Express redemption with growing premium
        let premium = 1.0 + 0.03 * observation_date / 0.5

        if perf >= 1.0 and not is_final then
            redeem notional * premium

        if is_final then
            if perf >= 1.0 then
                redeem notional * premium
            else if ki_hit then
                redeem notional * perf
            else
                redeem notional
