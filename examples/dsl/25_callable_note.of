// Callable Yield Note
// Issuer has the right to call (redeem early) the note at any
// observation date. Modeled here as automatic call when the
// underlying is above a call trigger (105%).
// Pays a high fixed coupon each period as compensation for call risk.
//
// Coupon: 3% per quarter (12% p.a.)
// Call trigger: 105%

product "Callable Yield Note 2Y"
    notional: 1_000_000
    maturity: 2.0

    underlyings
        SPX = asset(0)

    schedule quarterly from 0.25 to 2.0
        let perf = worst_of(performances())

        // Fixed coupon each period
        pay notional * 0.03

        // Issuer calls if above trigger
        if perf >= 1.05 and not is_final then
            redeem notional

        if is_final then
            redeem notional
