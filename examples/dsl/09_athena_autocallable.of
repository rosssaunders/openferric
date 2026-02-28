// Athena (Zero-Coupon Autocallable)
// No periodic coupons — instead, a single premium is paid at redemption.
// The premium grows with time: the later the autocall, the larger the payout.
// Simpler structure favored for its clean payoff profile.
//
// Premium: 5% per period (cumulative)
// Autocall barrier: 100%
// KI barrier: 60%

product "Athena 2Y"
    notional: 1_000_000
    maturity: 2.0

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)

    state
        ki_hit: bool = false

    schedule semi_annual from 0.5 to 2.0
        let wof = worst_of(performances())

        if wof <= 0.60 then
            set ki_hit = true

        // Autocall with cumulative premium
        if wof >= 1.0 and not is_final then
            pay notional * 0.05 * observation_date / 0.5
            redeem notional

        // Maturity
        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                pay notional * 0.05 * 4.0
                redeem notional
