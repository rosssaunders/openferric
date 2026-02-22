//! Closed-form analytic pricing routines for Binary Barrier.
//!
//! This module implements formulas and sensitivities used by fast deterministic engines.

use crate::core::{BarrierDirection, BarrierStyle, OptionType, PricingError};
use crate::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;

/// Cash-or-nothing barrier variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryBarrierType {
    DownIn,
    UpIn,
    DownOut,
    UpOut,
}

fn map_barrier(kind: BinaryBarrierType) -> (BarrierStyle, BarrierDirection) {
    match kind {
        BinaryBarrierType::DownIn => (BarrierStyle::In, BarrierDirection::Down),
        BinaryBarrierType::UpIn => (BarrierStyle::In, BarrierDirection::Up),
        BinaryBarrierType::DownOut => (BarrierStyle::Out, BarrierDirection::Down),
        BinaryBarrierType::UpOut => (BarrierStyle::Out, BarrierDirection::Up),
    }
}

fn barrier_breached_at_start(spot: f64, barrier: f64, kind: BinaryBarrierType) -> bool {
    match kind {
        BinaryBarrierType::DownIn | BinaryBarrierType::DownOut => spot <= barrier,
        BinaryBarrierType::UpIn | BinaryBarrierType::UpOut => spot >= barrier,
    }
}

fn payoff_in_the_money(option_type: OptionType, spot: f64, strike: f64) -> bool {
    match option_type {
        OptionType::Call => spot > strike,
        OptionType::Put => spot < strike,
    }
}

#[allow(clippy::too_many_arguments)]
fn barrier_vanilla_price(
    option_type: OptionType,
    kind: BinaryBarrierType,
    spot: f64,
    strike: f64,
    barrier: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    let (style, direction) = map_barrier(kind);
    barrier_price_closed_form_with_carry_and_rebate(
        option_type,
        style,
        direction,
        spot,
        strike,
        barrier,
        rate,
        dividend_yield,
        vol,
        expiry,
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
fn strike_derivative_of_barrier_vanilla(
    option_type: OptionType,
    kind: BinaryBarrierType,
    spot: f64,
    strike: f64,
    barrier: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    let h = (1.0e-4 * strike.max(1.0)).max(1.0e-5);

    if strike > 2.0 * h {
        let f_m2 = barrier_vanilla_price(
            option_type,
            kind,
            spot,
            strike - 2.0 * h,
            barrier,
            rate,
            dividend_yield,
            vol,
            expiry,
        );
        let f_m1 = barrier_vanilla_price(
            option_type,
            kind,
            spot,
            strike - h,
            barrier,
            rate,
            dividend_yield,
            vol,
            expiry,
        );
        let f_p1 = barrier_vanilla_price(
            option_type,
            kind,
            spot,
            strike + h,
            barrier,
            rate,
            dividend_yield,
            vol,
            expiry,
        );
        let f_p2 = barrier_vanilla_price(
            option_type,
            kind,
            spot,
            strike + 2.0 * h,
            barrier,
            rate,
            dividend_yield,
            vol,
            expiry,
        );

        (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * h)
    } else {
        let k_lo = (strike - h).max(1.0e-8);
        let k_hi = strike + h;
        let f_lo = barrier_vanilla_price(
            option_type,
            kind,
            spot,
            k_lo,
            barrier,
            rate,
            dividend_yield,
            vol,
            expiry,
        );
        let f_hi = barrier_vanilla_price(
            option_type,
            kind,
            spot,
            k_hi,
            barrier,
            rate,
            dividend_yield,
            vol,
            expiry,
        );
        (f_hi - f_lo) / (k_hi - k_lo)
    }
}

/// Cash-or-nothing barrier option price.
#[allow(clippy::too_many_arguments)]
pub fn cash_or_nothing_barrier_price(
    option_type: OptionType,
    kind: BinaryBarrierType,
    spot: f64,
    strike: f64,
    barrier: f64,
    cash: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> Result<f64, PricingError> {
    if spot <= 0.0 {
        return Err(PricingError::InvalidInput(
            "binary barrier spot must be > 0".to_string(),
        ));
    }
    if strike <= 0.0 {
        return Err(PricingError::InvalidInput(
            "binary barrier strike must be > 0".to_string(),
        ));
    }
    if barrier <= 0.0 {
        return Err(PricingError::InvalidInput(
            "binary barrier level must be > 0".to_string(),
        ));
    }
    if cash < 0.0 {
        return Err(PricingError::InvalidInput(
            "binary barrier cash must be >= 0".to_string(),
        ));
    }
    if vol <= 0.0 && expiry > 0.0 {
        return Err(PricingError::InvalidInput(
            "binary barrier volatility must be > 0 when expiry > 0".to_string(),
        ));
    }
    if expiry < 0.0 {
        return Err(PricingError::InvalidInput(
            "binary barrier expiry must be >= 0".to_string(),
        ));
    }

    if expiry <= 0.0 {
        let hit = barrier_breached_at_start(spot, barrier, kind);
        let active = match kind {
            BinaryBarrierType::DownIn | BinaryBarrierType::UpIn => hit,
            BinaryBarrierType::DownOut | BinaryBarrierType::UpOut => !hit,
        };
        return Ok(
            if active && payoff_in_the_money(option_type, spot, strike) {
                cash
            } else {
                0.0
            },
        );
    }

    let d_v_dk = strike_derivative_of_barrier_vanilla(
        option_type,
        kind,
        spot,
        strike,
        barrier,
        rate,
        dividend_yield,
        vol,
        expiry,
    );

    let unsigned_price = match option_type {
        OptionType::Call => -d_v_dk,
        OptionType::Put => d_v_dk,
    };

    let df_r = (-rate * expiry).exp();
    let upper = cash * df_r;

    Ok((cash * unsigned_price).clamp(0.0, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_reference(
        option_type: OptionType,
        kind: BinaryBarrierType,
        spot: f64,
        strike: f64,
        barrier: f64,
        cash: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        expiry: f64,
        expected: f64,
    ) {
        let value = cash_or_nothing_barrier_price(
            option_type,
            kind,
            spot,
            strike,
            barrier,
            cash,
            rate,
            dividend_yield,
            vol,
            expiry,
        )
        .unwrap();

        assert_relative_eq!(value, expected, epsilon = 6e-2);
    }

    #[test]
    fn matches_haug_quantlib_reference_values() {
        let barrier = 100.0;
        let cash = 15.0;
        let q = 0.0;
        let r = 0.10;
        let t = 0.5;
        let vol = 0.20;

        assert_reference(
            OptionType::Call,
            BinaryBarrierType::DownIn,
            105.0,
            102.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            4.9289,
        );
        assert_reference(
            OptionType::Call,
            BinaryBarrierType::DownIn,
            105.0,
            98.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            6.2150,
        );
        assert_reference(
            OptionType::Call,
            BinaryBarrierType::UpIn,
            95.0,
            102.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            5.8926,
        );
        assert_reference(
            OptionType::Call,
            BinaryBarrierType::UpIn,
            95.0,
            98.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            7.4519,
        );
        assert_reference(
            OptionType::Put,
            BinaryBarrierType::DownIn,
            105.0,
            102.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            4.4314,
        );
        assert_reference(
            OptionType::Put,
            BinaryBarrierType::DownIn,
            105.0,
            98.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            3.1454,
        );
        assert_reference(
            OptionType::Put,
            BinaryBarrierType::UpIn,
            95.0,
            102.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            5.3297,
        );
        assert_reference(
            OptionType::Put,
            BinaryBarrierType::UpIn,
            95.0,
            98.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            3.7704,
        );
        assert_reference(
            OptionType::Call,
            BinaryBarrierType::DownOut,
            105.0,
            102.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            4.8758,
        );
        assert_reference(
            OptionType::Call,
            BinaryBarrierType::DownOut,
            105.0,
            98.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            4.9081,
        );
        assert_reference(
            OptionType::Put,
            BinaryBarrierType::UpOut,
            95.0,
            102.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            3.0461,
        );
        assert_reference(
            OptionType::Put,
            BinaryBarrierType::UpOut,
            95.0,
            98.0,
            barrier,
            cash,
            r,
            q,
            vol,
            t,
            3.0054,
        );
    }
}
