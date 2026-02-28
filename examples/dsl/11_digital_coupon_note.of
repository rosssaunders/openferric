// Digital Coupon Note
// Pays a fixed coupon if the underlying is above the digital barrier,
// zero otherwise. No memory feature. Capital at risk if KI barrier
// is breached.
//
// Digital barrier: 80%
// Coupon: 3% per quarter (12% p.a. conditional)
// KI barrier: 55%

product "Digital Coupon Note 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false

    schedule quarterly from 0.25 to 1.0
        let perf = worst_of(performances())

        if perf <= 0.55 then
            set ki_hit = true

        // Digital coupon: all or nothing
        if perf >= 0.80 then
            pay notional * 0.03

        // Maturity
        if is_final then
            if ki_hit and perf < 1.0 then
                redeem notional * perf
            else
                redeem notional
