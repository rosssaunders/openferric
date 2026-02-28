// Zero Coupon Bond
// Simplest possible product: redeem notional at maturity, no coupons.
// PV = notional * exp(-r * T)

product "Zero Coupon"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        redeem notional
