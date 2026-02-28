// Fixed Coupon Note
// Pays a fixed 5% annual coupon and redeems at par at maturity.
// No equity linkage — pure fixed income in DSL form.

product "Fixed Coupon Note 3Y"
    notional: 1_000_000
    maturity: 3.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 3.0
        pay notional * 0.05
        if is_final then
            redeem notional
