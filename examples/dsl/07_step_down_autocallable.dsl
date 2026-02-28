// Step-Down Autocallable
// Autocall barrier decreases over time, making early redemption
// progressively easier. Common in declining-vol environments.
//
// Barriers: Q1=105%, Q2=100%, Q3=95%, Q4=90%, Q5=85%, Q6=80%
// Coupon: 8% p.a.
// KI barrier: 55%

product "Step-Down Autocall 18m"
    notional: 1_000_000
    maturity: 1.5

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)

    state
        ki_hit: bool = false

    schedule quarterly from 0.25 to 1.5
        let wof = worst_of(performances())

        if wof <= 0.55 then
            set ki_hit = true

        // Step-down autocall barriers using observation_date to determine quarter
        // Q1 (0.25): 105%, Q2 (0.50): 100%, Q3 (0.75): 95%
        // Q4 (1.00): 90%, Q5 (1.25): 85%, Q6 (1.50): 80%
        let barrier = 1.05 - observation_date * 0.2 + 0.05

        if wof >= barrier and not is_final then
            pay notional * 0.08 * observation_date
            redeem notional

        if is_final then
            pay notional * 0.08 * 1.5
            if ki_hit and wof < 0.80 then
                redeem notional * wof
            else
                redeem notional
