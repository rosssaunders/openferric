// Reverse Convertible
// Pays an enhanced coupon (10% p.a.) in exchange for downside risk.
// At maturity: if spot >= strike (100%), redeem at par.
// If spot < strike, investor receives notional * (S(T)/S(0)),
// i.e., they "bought" the stock at strike.

product "Reverse Convertible 1Y"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule quarterly from 0.25 to 1.0
        // Enhanced coupon paid quarterly
        pay notional * 0.025

        if is_final then
            let perf = worst_of(performances())
            if perf >= 1.0 then
                redeem notional
            else
                redeem notional * perf
