// Wedding Cake (Multi-Barrier Coupon)
// Coupon depends on which "tier" the worst-of performance falls into.
// Higher performance tiers pay higher coupons.
//
// Tier 1: WoF >= 100% → 3% coupon
// Tier 2: WoF >= 80%  → 2% coupon
// Tier 3: WoF >= 60%  → 1% coupon
// Below 60%: no coupon + KI risk
//
// KI barrier: 50%

product "Wedding Cake 2Y"
    notional: 1_000_000
    maturity: 2.0

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)
        NKY  = asset(2)

    state
        ki_hit: bool = false

    schedule quarterly from 0.25 to 2.0
        let wof = worst_of(performances())

        if wof <= 0.50 then
            set ki_hit = true

        // Tiered coupon structure
        if wof >= 1.0 then
            pay notional * 0.03
        else if wof >= 0.80 then
            pay notional * 0.02
        else if wof >= 0.60 then
            pay notional * 0.01

        // Maturity
        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
