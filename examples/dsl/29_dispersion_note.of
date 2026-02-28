// Dispersion Note
// Pays a coupon proportional to the spread between the best and worst
// performing assets. The wider the dispersion, the higher the coupon.
// Investors profit from divergence in the basket.
//
// Coupon: notional * (best_of - worst_of) * participation
// Participation: 50%
// Capital protected

product "Dispersion Note 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)
        NKY  = asset(2)

    schedule annual from 1.0 to 1.0
        let bof = best_of(performances())
        let wof = worst_of(performances())
        let spread = bof - wof

        pay notional * spread * 0.50
        redeem notional
