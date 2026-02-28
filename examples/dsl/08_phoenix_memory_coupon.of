// Phoenix with Memory Coupons
// Pays a coupon when worst-of performance is above the coupon barrier (70%).
// Missed coupons accumulate and are paid when the barrier is next breached.
// This "memory" feature is highly valued by investors.
//
// Coupon barrier: 70%
// KI barrier: 60%
// Autocall barrier: 100%
// Coupon: 2% per quarter (8% p.a.)

product "Phoenix Memory 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false
        missed_coupons: float = 0.0

    schedule quarterly from 0.25 to 1.0
        let wof = worst_of(performances())

        // Knock-in observation
        if wof <= 0.60 then
            set ki_hit = true

        // Coupon with memory
        if wof >= 0.70 then
            pay notional * (0.02 + missed_coupons)
            set missed_coupons = 0.0
        else
            set missed_coupons = missed_coupons + 0.02

        // Early autocall (not on final date)
        if wof >= 1.0 and not is_final then
            redeem notional

        // Maturity
        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
