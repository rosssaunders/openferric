// Napoleon (Worst-of Digital Autocallable)
// Monthly observations. Each month pays a fixed coupon only if
// the worst-of performance is above the digital barrier.
// Features autocall and KI barrier.
//
// Digital barrier: 85%
// Coupon: 1% per month (12% p.a. conditional)
// Autocall: 100% (semi-annual only, every 6th observation)
// KI: 55%

product "Napoleon 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)

    state
        ki_hit: bool = false
        obs_count: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let wof = worst_of(performances())
        set obs_count = obs_count + 1.0

        if wof <= 0.55 then
            set ki_hit = true

        // Digital coupon
        if wof >= 0.85 then
            pay notional * 0.01

        // Semi-annual autocall check (obs 6)
        if obs_count == 6.0 and wof >= 1.0 then
            redeem notional

        // Maturity
        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
