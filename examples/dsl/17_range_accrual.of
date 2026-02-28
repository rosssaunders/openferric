// Equity Range Accrual
// Pays a coupon proportional to the fraction of observation dates
// where the underlying stays within a range.
//
// Range: 80% – 120% of initial spot
// Max coupon: 10% p.a. * accrual fraction
// Monthly monitoring (12 dates)

product "Range Accrual 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        days_in_range: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if perf >= 0.80 and perf <= 1.20 then
            set days_in_range = days_in_range + 1.0

        if is_final then
            let accrual_frac = days_in_range / 12.0
            pay notional * 0.10 * accrual_frac
            redeem notional
