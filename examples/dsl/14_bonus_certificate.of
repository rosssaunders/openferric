// Bonus Certificate
// Offers a minimum return (bonus level) as long as the barrier is never
// breached. If barrier is breached, the certificate tracks the underlying
// 1:1 from inception.
//
// Bonus level: 120% (i.e., min 20% return)
// Barrier: 70%
// No cap on upside

product "Bonus Certificate 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        barrier_hit: bool = false

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if perf <= 0.70 then
            set barrier_hit = true

        if is_final then
            if barrier_hit then
                // Barrier hit: 1:1 participation
                redeem notional * perf
            else
                // Barrier not hit: at least bonus level
                let payout = max(perf, 1.20)
                redeem notional * payout
