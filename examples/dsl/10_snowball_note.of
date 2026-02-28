// Snowball Note
// Coupon "snowballs" — each period's coupon is the previous coupon plus a
// fixed increment, but only if the underlying is above the coupon barrier.
// If below, the coupon resets to zero and starts snowballing again.
//
// Increment: 1% per quarter
// Coupon barrier: 80%
// KI barrier: 60%

product "Snowball 2Y"
    notional: 1_000_000
    maturity: 2.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false
        current_coupon: float = 0.0

    schedule quarterly from 0.25 to 2.0
        let perf = worst_of(performances())

        if perf <= 0.60 then
            set ki_hit = true

        // Snowball coupon logic
        if perf >= 0.80 then
            set current_coupon = current_coupon + 0.01
            pay notional * current_coupon
        else
            set current_coupon = 0.0

        // Maturity
        if is_final then
            if ki_hit and perf < 1.0 then
                redeem notional * perf
            else
                redeem notional
