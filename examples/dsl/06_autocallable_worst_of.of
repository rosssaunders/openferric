// Worst-of Autocallable (3 underlyings, 18 months)
// Classic autocallable on a basket of 3 indices.
// Quarterly observations:
//   - If worst-of performance >= 100%, early redeem + coupon
//   - Knock-in barrier at 60%
//   - At maturity: coupon always paid; if KI hit and WoF < 100%,
//     redeem at WoF level; otherwise redeem at par

product "WoF Autocall 18m"
    notional: 1_000_000
    maturity: 1.5

    underlyings
        SPX  = asset(0)
        SX5E = asset(1)
        NKY  = asset(2)

    state
        ki_hit: bool = false

    schedule quarterly from 0.25 to 1.5
        let wof = worst_of(performances())

        if wof <= 0.60 then
            set ki_hit = true

        // Early autocall (not on final date)
        if wof >= 1.0 and not is_final then
            pay notional * 0.08 * observation_date
            redeem notional

        // Maturity
        if is_final then
            pay notional * 0.08 * 1.5
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
