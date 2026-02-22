//! Module `market::dividends`.
//!
//! Deterministic discrete-dividend modelling utilities for equity pricing.
//!
//! This module supports:
//! - cash dividends (fixed amount),
//! - proportional dividends (fraction of spot),
//! - mixed schedules,
//! - forward/prepaid-forward calculations under mixed schedules,
//! - put-call parity bootstrap of an implied dividend curve.
//!
//! References:
//! - Hull, *Options, Futures, and Other Derivatives* (11th ed.), Ch. 13.
//! - Haug, Haug, Lewis (2003): escrowed-dividend treatment for option pricing.

/// Type of deterministic discrete dividend event.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DividendKind {
    /// Fixed cash amount subtracted from spot at ex-date.
    Cash(f64),
    /// Proportional drop as a fraction of spot, i.e. `S+ = S- * (1 - p)`.
    Proportional(f64),
}

impl DividendKind {
    fn validate(self) -> Result<(), String> {
        match self {
            Self::Cash(amount) => {
                if !amount.is_finite() || amount < 0.0 {
                    return Err("cash dividend amount must be finite and >= 0".to_string());
                }
            }
            Self::Proportional(ratio) => {
                if !ratio.is_finite() || !(0.0..1.0).contains(&ratio) {
                    return Err(
                        "proportional dividend ratio must be finite and in [0, 1)".to_string()
                    );
                }
            }
        }
        Ok(())
    }
}

/// Deterministic dividend event.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DividendEvent {
    /// Ex-dividend time in years from valuation date.
    pub time: f64,
    /// Dividend amount/type.
    pub kind: DividendKind,
}

impl DividendEvent {
    /// Builds a cash dividend event.
    pub fn cash(time: f64, amount: f64) -> Result<Self, String> {
        let event = Self {
            time,
            kind: DividendKind::Cash(amount),
        };
        event.validate()?;
        Ok(event)
    }

    /// Builds a proportional dividend event.
    pub fn proportional(time: f64, ratio: f64) -> Result<Self, String> {
        let event = Self {
            time,
            kind: DividendKind::Proportional(ratio),
        };
        event.validate()?;
        Ok(event)
    }

    /// Applies the event jump to a pre-dividend spot.
    #[inline]
    pub fn apply_jump(self, pre_div_spot: f64) -> f64 {
        match self.kind {
            DividendKind::Cash(amount) => (pre_div_spot - amount).max(0.0),
            DividendKind::Proportional(ratio) => pre_div_spot * (1.0 - ratio),
        }
    }

    fn validate(self) -> Result<(), String> {
        if !self.time.is_finite() || self.time <= 0.0 {
            return Err("dividend event time must be finite and > 0".to_string());
        }
        self.kind.validate()
    }
}

/// Deterministic mixed discrete-dividend schedule.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct DividendSchedule {
    events: Vec<DividendEvent>,
}

impl DividendSchedule {
    /// Builds a schedule from events and validates ordering and values.
    pub fn new(mut events: Vec<DividendEvent>) -> Result<Self, String> {
        events.sort_by(|a, b| a.time.total_cmp(&b.time));
        let schedule = Self { events };
        schedule.validate()?;
        Ok(schedule)
    }

    /// Returns an empty schedule.
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns `true` when no events are present.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the underlying sorted event slice.
    #[inline]
    pub fn events(&self) -> &[DividendEvent] {
        &self.events
    }

    /// Returns all events up to and including `maturity`.
    #[inline]
    pub fn events_until(&self, maturity: f64) -> impl Iterator<Item = &DividendEvent> {
        self.events.iter().filter(move |ev| ev.time <= maturity)
    }

    /// Validates event values and strictly-increasing event times.
    pub fn validate(&self) -> Result<(), String> {
        let mut prev_time = 0.0_f64;
        for event in &self.events {
            event.validate()?;
            if event.time <= prev_time {
                return Err("dividend event times must be strictly increasing".to_string());
            }
            prev_time = event.time;
        }
        Ok(())
    }

    /// Prices the mixed-schedule forward at maturity `T`.
    ///
    /// Uses deterministic jump conditions:
    /// - cash: `S+ = S- - D`,
    /// - proportional: `S+ = S- * (1 - p)`.
    ///
    /// Between jumps, expectation grows at `r - q` (continuous yield `q`).
    pub fn forward_price(
        &self,
        spot: f64,
        rate: f64,
        continuous_dividend_yield: f64,
        maturity: f64,
    ) -> f64 {
        if maturity <= 0.0 {
            return spot;
        }
        if !spot.is_finite() || spot <= 0.0 {
            return 0.0;
        }

        let carry = rate - continuous_dividend_yield;
        let mut forward = spot * (carry * maturity).exp();

        for event in self.events_until(maturity) {
            match event.kind {
                DividendKind::Cash(amount) => {
                    forward -= amount * (carry * (maturity - event.time)).exp();
                }
                DividendKind::Proportional(ratio) => {
                    forward *= 1.0 - ratio;
                }
            }
        }

        forward.max(0.0)
    }

    /// Prepaid-forward spot equivalent at maturity `T`.
    ///
    /// This is the escrowed spot used in the Haug-Haug-Lewis approach.
    #[inline]
    pub fn prepaid_forward_spot(
        &self,
        spot: f64,
        rate: f64,
        continuous_dividend_yield: f64,
        maturity: f64,
    ) -> f64 {
        self.forward_price(spot, rate, continuous_dividend_yield, maturity)
            * (-rate * maturity).exp()
    }

    /// Escrowed spot adjustment (Haug-Haug-Lewis style) with no continuous yield.
    #[inline]
    pub fn escrowed_spot_adjustment(&self, spot: f64, rate: f64, maturity: f64) -> f64 {
        self.prepaid_forward_spot(spot, rate, 0.0, maturity)
    }

    /// Equivalent continuous dividend yield implied by the schedule up to `T`.
    pub fn effective_dividend_yield(
        &self,
        spot: f64,
        rate: f64,
        continuous_dividend_yield: f64,
        maturity: f64,
    ) -> f64 {
        if maturity <= 0.0 || spot <= 0.0 {
            return continuous_dividend_yield;
        }
        let forward = self.forward_price(spot, rate, continuous_dividend_yield, maturity);
        let ratio = (forward / spot).max(1.0e-16);
        rate - ratio.ln() / maturity
    }

    /// Applies all ex-date jumps up to and including `time`.
    #[inline]
    pub fn apply_jumps_until(&self, mut spot: f64, time: f64) -> f64 {
        for event in self.events_until(time) {
            spot = event.apply_jump(spot);
        }
        spot
    }
}

/// Single-strike put-call parity observation for one expiry.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PutCallParityQuote {
    /// Time to maturity in years.
    pub maturity: f64,
    /// Common strike used for call and put.
    pub strike: f64,
    /// Call premium.
    pub call_price: f64,
    /// Put premium.
    pub put_price: f64,
}

impl PutCallParityQuote {
    /// Validates quote fields.
    pub fn validate(self) -> Result<(), String> {
        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err("parity quote maturity must be finite and > 0".to_string());
        }
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err("parity quote strike must be finite and > 0".to_string());
        }
        if !self.call_price.is_finite() || self.call_price < 0.0 {
            return Err("parity quote call price must be finite and >= 0".to_string());
        }
        if !self.put_price.is_finite() || self.put_price < 0.0 {
            return Err("parity quote put price must be finite and >= 0".to_string());
        }
        Ok(())
    }

    /// Implied forward from put-call parity at this maturity.
    #[inline]
    pub fn implied_forward(self, rate: f64) -> f64 {
        (self.call_price - self.put_price) * (rate * self.maturity).exp() + self.strike
    }

    /// Implied prepaid-forward spot from put-call parity.
    #[inline]
    pub fn implied_prepaid_forward(self, rate: f64) -> f64 {
        self.call_price - self.put_price + self.strike * (-rate * self.maturity).exp()
    }
}

/// Bootstrapped point on an implied dividend curve.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BootstrappedDividendPoint {
    /// Maturity in years.
    pub maturity: f64,
    /// Forward implied by parity.
    pub forward: f64,
    /// Prepaid-forward spot implied by parity.
    pub prepaid_forward: f64,
    /// Equivalent continuous dividend yield up to maturity.
    pub implied_dividend_yield: f64,
    /// Cumulative PV dividend impact from spot to this maturity.
    pub cumulative_pv_dividends: f64,
}

/// Dividend curve bootstrapped from put-call parity quotes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DividendCurveBootstrap {
    /// Spot used for bootstrap.
    pub spot: f64,
    /// Flat discount rate used for bootstrap.
    pub rate: f64,
    /// Bootstrapped maturity points.
    pub points: Vec<BootstrappedDividendPoint>,
}

impl DividendCurveBootstrap {
    /// Bootstraps a dividend curve from put-call parity observations.
    ///
    /// Multiple strikes per maturity are supported and averaged.
    pub fn from_put_call_parity(
        spot: f64,
        rate: f64,
        quotes: &[PutCallParityQuote],
    ) -> Result<Self, String> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err("spot must be finite and > 0".to_string());
        }
        if quotes.is_empty() {
            return Err("at least one parity quote is required".to_string());
        }
        for q in quotes {
            q.validate()?;
        }

        let mut sorted = quotes.to_vec();
        sorted.sort_by(|a, b| a.maturity.total_cmp(&b.maturity));

        let mut points = Vec::new();
        let mut i = 0usize;
        while i < sorted.len() {
            let t = sorted[i].maturity;
            let mut sum_fwd = 0.0_f64;
            let mut sum_prepaid = 0.0_f64;
            let mut n = 0usize;

            while i < sorted.len() && (sorted[i].maturity - t).abs() <= 1.0e-12 {
                let fwd = sorted[i].implied_forward(rate);
                let prepaid = sorted[i].implied_prepaid_forward(rate);
                if !fwd.is_finite() || !prepaid.is_finite() || prepaid <= 0.0 {
                    return Err(
                        "invalid parity quote implies non-positive prepaid forward".to_string()
                    );
                }
                sum_fwd += fwd;
                sum_prepaid += prepaid;
                n += 1;
                i += 1;
            }

            let n_f64 = n as f64;
            let forward = sum_fwd / n_f64;
            let prepaid_forward = sum_prepaid / n_f64;
            let implied_dividend_yield = -((prepaid_forward / spot).max(1.0e-16)).ln() / t;
            let cumulative_pv_dividends = spot - prepaid_forward;

            points.push(BootstrappedDividendPoint {
                maturity: t,
                forward,
                prepaid_forward,
                implied_dividend_yield,
                cumulative_pv_dividends,
            });
        }

        Ok(Self { spot, rate, points })
    }

    /// Prepaid-forward spot from the curve at maturity `T`.
    ///
    /// Uses log-linear interpolation in prepaid-forward space.
    pub fn prepaid_forward_spot(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            return self.spot;
        }
        if self.points.is_empty() {
            return self.spot;
        }

        let first = self.points[0];
        if maturity <= first.maturity {
            let w = (maturity / first.maturity).clamp(0.0, 1.0);
            let ln_p0 = self.spot.max(1.0e-16).ln();
            let ln_p1 = first.prepaid_forward.max(1.0e-16).ln();
            return (ln_p0 + (ln_p1 - ln_p0) * w).exp();
        }

        for win in self.points.windows(2) {
            let p0 = win[0];
            let p1 = win[1];
            if maturity <= p1.maturity {
                let w = (maturity - p0.maturity) / (p1.maturity - p0.maturity);
                let ln_a = p0.prepaid_forward.max(1.0e-16).ln();
                let ln_b = p1.prepaid_forward.max(1.0e-16).ln();
                return (ln_a + (ln_b - ln_a) * w.clamp(0.0, 1.0)).exp();
            }
        }

        let last = self.points[self.points.len() - 1];
        self.spot * (-last.implied_dividend_yield * maturity).exp()
    }

    /// Forward from the curve at maturity `T`.
    #[inline]
    pub fn forward(&self, maturity: f64) -> f64 {
        self.prepaid_forward_spot(maturity) * (self.rate * maturity).exp()
    }

    /// Converts the curve into a cash-dividend schedule at bootstrap maturities.
    ///
    /// This assumes all dividend impact between neighboring maturities can be
    /// represented as one cash dividend paid at the later maturity.
    pub fn to_cash_dividend_schedule(&self) -> Result<DividendSchedule, String> {
        let mut events = Vec::new();
        let mut prev_cum_pv = 0.0_f64;

        for point in &self.points {
            if point.cumulative_pv_dividends + 1.0e-12 < prev_cum_pv {
                return Err(
                    "cumulative PV dividends must be non-decreasing to map into cash schedule"
                        .to_string(),
                );
            }
            let increment_pv = (point.cumulative_pv_dividends - prev_cum_pv).max(0.0);
            if increment_pv > 0.0 {
                let cash = increment_pv * (self.rate * point.maturity).exp();
                events.push(DividendEvent {
                    time: point.maturity,
                    kind: DividendKind::Cash(cash),
                });
            }
            prev_cum_pv = point.cumulative_pv_dividends.max(prev_cum_pv);
        }

        DividendSchedule::new(events)
    }
}

/// Convenience bootstrap wrapper.
#[inline]
pub fn bootstrap_dividend_curve_from_put_call_parity(
    spot: f64,
    rate: f64,
    quotes: &[PutCallParityQuote],
) -> Result<DividendCurveBootstrap, String> {
    DividendCurveBootstrap::from_put_call_parity(spot, rate, quotes)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn mixed_schedule_forward_and_escrowed_spot() {
        let schedule = DividendSchedule::new(vec![
            DividendEvent::cash(0.25, 1.0).expect("valid cash event"),
            DividendEvent::proportional(0.5, 0.02).expect("valid proportional event"),
            DividendEvent::cash(0.75, 0.5).expect("valid cash event"),
        ])
        .expect("valid schedule");

        let spot = 100.0;
        let r = 0.03;
        let q = 0.0;
        let t = 1.0;

        let fwd = schedule.forward_price(spot, r, q, t);
        let prepaid = schedule.prepaid_forward_spot(spot, r, q, t);
        let q_eff = schedule.effective_dividend_yield(spot, r, q, t);

        assert!(fwd < spot * (r * t).exp());
        assert!(prepaid < spot);
        assert!(q_eff > 0.0);
        assert_relative_eq!(prepaid * (r * t).exp(), fwd, epsilon = 1e-12);
    }

    #[test]
    fn put_call_parity_bootstrap_reproduces_forward_points() {
        let spot = 100.0_f64;
        let rate = 0.02_f64;
        let forwards = [
            (0.5_f64, 100.8_f64),
            (1.0_f64, 101.1_f64),
            (2.0_f64, 102.4_f64),
        ];
        let strike = 100.0_f64;

        let quotes: Vec<PutCallParityQuote> = forwards
            .iter()
            .map(|(t, fwd)| {
                let call_minus_put = (fwd - strike) * (-rate * *t).exp();
                let put = 6.0;
                let call = put + call_minus_put;
                PutCallParityQuote {
                    maturity: *t,
                    strike,
                    call_price: call,
                    put_price: put,
                }
            })
            .collect();

        let curve = bootstrap_dividend_curve_from_put_call_parity(spot, rate, &quotes)
            .expect("bootstrap should succeed");

        for (t, fwd) in forwards {
            let model_fwd = curve.forward(t);
            assert_relative_eq!(model_fwd, fwd, epsilon = 1e-10);
        }
    }
}
