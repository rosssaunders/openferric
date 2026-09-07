//! Risk-neutral Hull-White valuation of path-dependent rate coupons.
//!
//! Historical reset fixings are contractual observations, never forwards from
//! today's curve. Payment on the valuation date is treated as already settled.
//! The curve is anchored at valuation; contract and fixing times retain their
//! original origin. Stderr measures sampling error, not model uncertainty.

use crate::core::{DiagKey, Diagnostics, PricingResult};
use crate::engines::monte_carlo::mc_engine::RunningMoments;
use crate::math::fast_rng::Xoshiro256PlusPlus;
use crate::models::{HullWhite, hull_white_paths::HullWhitePathGenerator};
use crate::rates::YieldCurve;

use super::{CouponPeriod, SnowballNote, TargetRedemptionNote};

/// Lifecycle data for a callable rate note, rates TARN or snowball. Supply
/// historical observations needed for earned coupons, targets or coupon recursion.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RateNoteHistory {
    pub valuation_time: f64,
    /// Pairs `(observation time, underlying fixing)`: simple rates for floating
    /// coupons, daily short rates for ranges, or swap rates for CMS coupons.
    pub fixings: Vec<(f64, f64)>,
}

enum Contract<'a> {
    Tarn(&'a TargetRedemptionNote),
    Snowball(&'a SnowballNote),
}

impl TargetRedemptionNote {
    /// Risk-neutral price with stochastic discounting and pathwise target redemption.
    pub fn price_hull_white_mc(
        &self,
        model: &HullWhite,
        curve: &YieldCurve,
        history: &RateNoteHistory,
        num_paths: usize,
        seed: u64,
    ) -> Result<PricingResult, String> {
        self.validate()?;
        price(
            Contract::Tarn(self),
            &self.coupon_schedule,
            model,
            curve,
            history,
            num_paths,
            seed,
        )
    }
}

impl SnowballNote {
    /// Risk-neutral price including coupon recursion along each rate path.
    pub fn price_hull_white_mc(
        &self,
        model: &HullWhite,
        curve: &YieldCurve,
        history: &RateNoteHistory,
        num_paths: usize,
        seed: u64,
    ) -> Result<PricingResult, String> {
        self.validate()?;
        price(
            Contract::Snowball(self),
            &self.coupon_schedule,
            model,
            curve,
            history,
            num_paths,
            seed,
        )
    }
}

fn price(
    contract: Contract<'_>,
    schedule: &[CouponPeriod],
    model: &HullWhite,
    curve: &YieldCurve,
    history: &RateNoteHistory,
    num_paths: usize,
    seed: u64,
) -> Result<PricingResult, String> {
    if num_paths < 2 || !history.valuation_time.is_finite() || history.valuation_time < 0.0 {
        return Err(
            "at least two paths and a non-negative finite valuation time are required".into(),
        );
    }
    if history.fixings.iter().any(|&(time, rate)| {
        !time.is_finite()
            || !rate.is_finite()
            || time > history.valuation_time
            || !schedule.iter().any(|period| period.start_time == time)
    }) || history.fixings.iter().enumerate().any(|(index, fixing)| {
        history.fixings[..index]
            .iter()
            .any(|earlier| earlier.0 == fixing.0)
    }) {
        return Err(
            "fixings must be finite, unique contractual resets at or before valuation".into(),
        );
    }
    let mut times = vec![0.0];
    for period in schedule {
        for time in [period.start_time, period.payment_time] {
            if time > history.valuation_time {
                times.push(time - history.valuation_time);
            }
        }
    }
    times.sort_by(f64::total_cmp);
    times.dedup();
    let generator = HullWhitePathGenerator::new(model, curve, &times)?;
    let index_at = |time: f64| {
        times
            .binary_search_by(|candidate| candidate.total_cmp(&time))
            .unwrap()
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut moments = RunningMoments::default();
    for _ in 0..num_paths {
        if schedule
            .last()
            .is_some_and(|period| period.payment_time <= history.valuation_time)
        {
            moments.record(0.0);
            continue;
        }
        let path = generator.sample(&mut rng)?;
        let mut present_value = 0.0;
        let mut accumulated = 0.0;
        let mut previous_coupon = match contract {
            Contract::Snowball(note) => note.initial_coupon,
            _ => 0.0,
        };
        for (index, period) in schedule.iter().enumerate() {
            let historical = history
                .fixings
                .iter()
                .find(|fixing| fixing.0 == period.start_time)
                .map(|fixing| fixing.1);
            let floating = if let Some(fixing) = historical {
                fixing
            } else if period.start_time < history.valuation_time {
                return Err(format!(
                    "missing historical fixing at {}",
                    period.start_time
                ));
            } else {
                let reset = period.start_time - history.valuation_time;
                let end = period.end_time - history.valuation_time;
                let discount =
                    model.bond_price(reset, end, path.short_rates[index_at(reset)], curve);
                (1.0 / discount - 1.0) / period.accrual()
            };
            let (notional, redemption, raw, floor, cap) = match contract {
                Contract::Tarn(note) => (
                    note.notional,
                    note.redemption,
                    floating + note.spread,
                    note.floor,
                    note.cap,
                ),
                Contract::Snowball(note) => (
                    note.notional,
                    note.redemption,
                    (previous_coupon + note.spread - floating).max(0.0),
                    note.floor,
                    note.cap,
                ),
            };
            let coupon_rate = raw
                .max(floor.unwrap_or(f64::NEG_INFINITY))
                .min(cap.unwrap_or(f64::INFINITY));
            previous_coupon = coupon_rate;
            let mut coupon = notional * period.accrual() * coupon_rate;
            let mut redeemed = false;
            if let Contract::Tarn(note) = contract {
                coupon = coupon.min((note.target_coupon - accumulated).max(0.0));
                accumulated += coupon;
                redeemed = accumulated >= note.target_coupon - 1.0e-12;
            }
            let cashflow = coupon
                + if redeemed || index + 1 == schedule.len() {
                    redemption
                } else {
                    0.0
                };
            if period.payment_time > history.valuation_time {
                let payment = period.payment_time - history.valuation_time;
                let discount = if period.start_time <= history.valuation_time {
                    curve.discount_factor(payment)
                } else {
                    path.discount_factors[index_at(payment)]
                };
                present_value += cashflow * discount;
            }
            if redeemed {
                break;
            }
        }
        if !present_value.is_finite() {
            return Err("non-finite rate-note cashflow".into());
        }
        moments.record(present_value);
    }
    let mut diagnostics = Diagnostics::new();
    diagnostics.insert_key(DiagKey::NumPaths, num_paths as f64);
    diagnostics.insert_key(DiagKey::ObservationCount, times.len() as f64);
    Ok(PricingResult {
        price: moments.mean(),
        stderr: Some((moments.sample_variance() / num_paths as f64).sqrt()),
        greeks: None,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::CouponScheduleBuilder;
    use crate::rates::Frequency;

    fn tarn() -> TargetRedemptionNote {
        TargetRedemptionNote {
            notional: 100.0,
            redemption: 100.0,
            target_coupon: 5.0,
            coupon_schedule: CouponScheduleBuilder::new(0.0, 2.0, Frequency::SemiAnnual)
                .unwrap()
                .build_floating(0.0, None, None)
                .unwrap(),
            spread: 0.0,
            floor: Some(0.0),
            cap: None,
        }
    }

    #[test]
    fn seasoned_target_is_reconstructed_and_last_coupon_is_clipped() {
        let note = tarn();
        let history = RateNoteHistory {
            valuation_time: 0.75,
            fixings: vec![(0.0, 0.06), (0.5, 0.06)],
        };
        let curve = YieldCurve::new(vec![(4.0, (-0.04_f64 * 4.0).exp())]);
        let result = note
            .price_hull_white_mc(&HullWhite::new(0.1, 0.02), &curve, &history, 10, 7)
            .unwrap();
        let expected = 102.0 * (-0.04_f64 * 0.25).exp();
        assert!((result.price - expected).abs() < 1.0e-12);
    }

    #[test]
    fn missing_fixing_fails_and_settled_target_has_no_value() {
        let note = tarn();
        let curve = YieldCurve::new(vec![(4.0, 0.8)]);
        let model = HullWhite::new(0.1, 0.0);
        assert!(
            note.price_hull_white_mc(
                &model,
                &curve,
                &RateNoteHistory {
                    valuation_time: 0.75,
                    ..Default::default()
                },
                2,
                0
            )
            .is_err()
        );
        let history = RateNoteHistory {
            valuation_time: 1.0,
            fixings: vec![(0.0, 0.06), (0.5, 0.06)],
        };
        assert_eq!(
            note.price_hull_white_mc(&model, &curve, &history, 2, 0)
                .unwrap()
                .price,
            0.0
        );
    }

    #[test]
    fn stochastic_linear_coupons_reprice_a_par_floater() {
        let mut note = tarn();
        note.target_coupon = 1.0e6;
        note.floor = None;
        let curve = YieldCurve::new(vec![(0.5, 0.99), (1.0, 0.965), (2.0, 0.90)]);
        let result = note
            .price_hull_white_mc(
                &HullWhite::new(0.1, 0.02),
                &curve,
                &RateNoteHistory::default(),
                60_000,
                42,
            )
            .unwrap();
        assert!((result.price - 100.0).abs() < 4.0 * result.stderr.unwrap());
    }
}
