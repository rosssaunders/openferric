// Lookback-Style Coupon Note
// Tracks the maximum performance seen across observation dates.
// At maturity, pays a coupon based on the highest watermark achieved.
// Principal always returned (capital protected).
//
// Coupon: 50% participation in max observed performance above par.

product "Lookback Coupon 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        max_perf: float = 1.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        // Track high watermark
        if perf > max_perf then
            set max_perf = perf

        if is_final then
            // Pay coupon based on best performance seen
            let upside = max(max_perf - 1.0, 0.0) * 0.50
            pay notional * upside
            redeem notional
