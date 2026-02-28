// Best-of Call on Basket
// At maturity, pays the best performance among 3 underlyings.
// Investor profits from the strongest-performing asset.
// Capital at risk (no protection).
//
// This is essentially a rainbow option (best-of call).

product "Best-of Call 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)
        NKY  = asset(2)

    schedule annual from 1.0 to 1.0
        let bof = best_of(performances())

        // Pay upside of best performer
        let payout = max(bof - 1.0, 0.0)
        pay notional * payout
        redeem notional
