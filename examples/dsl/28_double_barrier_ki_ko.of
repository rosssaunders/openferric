// Double Barrier (Knock-In / Knock-Out)
// Combines an upper knock-out barrier with a lower knock-in barrier.
// If upper barrier is hit, the structure terminates with a rebate.
// If lower barrier is hit, capital is at risk at maturity.
// Neither hit: full notional + coupon.
//
// Upper KO: 140%
// Lower KI: 60%
// KO rebate: 5%
// Coupon: 8% p.a.

product "Double Barrier KIKO 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        // Lower knock-in
        if perf <= 0.60 then
            set ki_hit = true

        // Upper knock-out
        if perf >= 1.40 and not is_final then
            pay notional * 0.05
            redeem notional

        if is_final then
            pay notional * 0.08
            if ki_hit and perf < 1.0 then
                redeem notional * perf
            else
                redeem notional
