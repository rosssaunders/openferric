// Worst-of Put
// At maturity, pays max(0, strike - worst_of_performance) * notional.
// The investor is long a put on the worst-performing asset.
// Strike at 100%.

product "Worst-of Put 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)

    schedule annual from 1.0 to 1.0
        let wof = worst_of(performances())
        let put_payout = max(1.0 - wof, 0.0)
        pay notional * put_payout
        redeem notional
