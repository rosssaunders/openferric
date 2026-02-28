// Twin-Win Certificate
// Profits from both upward and downward moves of the underlying.
// At maturity: pays abs(performance - 1) as a bonus.
// BUT if a knock-in barrier is breached, downside protection is lost
// and the investor participates in the full downside.
//
// KI barrier: 60%
// Participation: 100% both ways (when no KI)

product "Twin-Win 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        // Barrier monitoring
        if perf <= 0.60 then
            set ki_hit = true

        // Maturity payoff
        if is_final then
            if ki_hit then
                // KI hit: straight equity participation
                redeem notional * perf
            else
                // No KI: profit from abs move
                let move = perf - 1.0
                let abs_move = max(move, 0.0 - move)
                redeem notional * (1.0 + abs_move)
