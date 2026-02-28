// Shark Fin Note
// Capital protected with capped upside participation.
// If the underlying ever breaches an upper barrier (knock-out),
// the upside is capped at a rebate level instead of full participation.
//
// Upper KO barrier: 130%
// Participation (no KO): 100% of upside
// Rebate (if KO): fixed 10%
// Downside: 100% protected

product "Shark Fin 1Y"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ko_hit: bool = false

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        // Upper barrier monitoring
        if perf >= 1.30 then
            set ko_hit = true

        if is_final then
            if ko_hit then
                // Knocked out: pay rebate only
                pay notional * 0.10
                redeem notional
            else
                // No KO: participate in upside, protected on downside
                let upside = max(perf - 1.0, 0.0)
                pay notional * upside
                redeem notional
