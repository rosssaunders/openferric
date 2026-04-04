# DSL Product Examples

## Worst-of Autocallable

Three-underlying autocallable with knock-in barrier and quarterly observation.

```
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

        // Check knock-in barrier
        if wof <= 0.60 then
            set ki_hit = true

        // Early autocall if above par (excluding final date)
        if wof >= 1.0 and not is_final then
            pay notional * 0.08 * observation_date
            redeem notional

        // Maturity logic
        if is_final then
            pay notional * 0.08 * 1.5
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
```

## Phoenix with Memory Coupons

Pays a coupon when worst-of performance is above the coupon barrier. Missed
coupons accumulate and are paid when the barrier is subsequently breached.

```
product "Phoenix Memory"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        ki_hit: bool = false
        missed_coupons: float = 0.0

    schedule quarterly from 0.25 to 1.0
        let wof = worst_of(performances())

        if wof <= 0.60 then
            set ki_hit = true

        // Coupon with memory
        if wof >= 0.70 then
            pay notional * (0.02 + missed_coupons)
            set missed_coupons = 0.0
        else
            set missed_coupons = missed_coupons + 0.02

        // Early autocall
        if wof >= 1.0 and not is_final then
            redeem notional

        // Maturity
        if is_final then
            if ki_hit and wof < 1.0 then
                redeem notional * wof
            else
                redeem notional
```

## Accumulator (TARF)

Accumulates units when the spot is between two barriers, with a target cap.

```
product "Accumulator"
    notional: 100
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        accumulated: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if accumulated < 1.0 then
            if perf >= 0.90 and perf <= 1.10 then
                set accumulated = accumulated + 0.0833
                pay notional * 0.0833 * (perf - 1.0)

        if is_final then
            redeem notional
```

## Equity Range Accrual

Pays a coupon proportional to the number of days the underlying stays within a
range.

```
product "Range Accrual"
    notional: 1_000_000
    maturity: 1.0

    underlyings
        SPX = asset(0)

    state
        days_in_range: float = 0.0

    schedule monthly from 0.0833 to 1.0
        let perf = worst_of(performances())

        if perf >= 0.80 and perf <= 1.20 then
            set days_in_range = days_in_range + 1.0

        if is_final then
            let accrual_frac = days_in_range / 12.0
            pay notional * 0.10 * accrual_frac
            redeem notional
```

## Zero Coupon (Simplest Product)

```
product "Zero Coupon"
    notional: 1_000_000
    maturity: 1.0
    underlyings
        SPX = asset(0)
    schedule annual from 1.0 to 1.0
        redeem notional
```
