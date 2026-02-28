// Barrier Reverse Convertible (BRC)
// Like a reverse convertible but with knock-in protection.
// Downside only applies if a barrier (60%) is ever breached.
// If barrier never touched, investor receives full notional regardless
// of final spot level.

product "Barrier Reverse Convertible 1Y"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    state
        ki_hit: bool = false
    schedule quarterly from 0.25 to 1.0
        let perf = worst_of(performances())

        // Continuous barrier observation (approximated at observation dates)
        if perf <= 0.60 then
            set ki_hit = true

        // Enhanced coupon
        pay notional * 0.02

        if is_final then
            if ki_hit and perf < 1.0 then
                redeem notional * perf
            else
                redeem notional
