use crate::rates::YieldCurve;

/// Inflation curve represented by CPI growth ratios `CPI(t)/CPI(0)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InflationCurve {
    /// Curve nodes as `(tenor_years, cpi_ratio)`.
    pub nodes: Vec<(f64, f64)>,
}

impl InflationCurve {
    /// Creates an inflation curve from unsorted CPI-ratio nodes.
    pub fn new(mut nodes: Vec<(f64, f64)>) -> Self {
        nodes.retain(|(t, ratio)| *t > 0.0 && *ratio > 0.0);
        nodes.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { nodes }
    }

    /// CPI growth ratio `CPI(t)/CPI(0)`.
    pub fn cpi_ratio(&self, t: f64) -> f64 {
        ratio_from_points(&self.nodes, t)
    }

    /// Annualized zero inflation rate at tenor `t`.
    pub fn zero_inflation_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }

        self.cpi_ratio(t).powf(1.0 / t) - 1.0
    }

    /// Forward CPI ratio `CPI(t2)/CPI(t1)`.
    pub fn forward_cpi_ratio(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 {
            return 1.0;
        }

        self.cpi_ratio(t2) / self.cpi_ratio(t1).max(1.0e-12)
    }

    /// Annualized forward inflation between `t1` and `t2`.
    pub fn forward_inflation_rate(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 {
            return 0.0;
        }

        self.forward_cpi_ratio(t1, t2).powf(1.0 / (t2 - t1)) - 1.0
    }

    /// Projects CPI level from base CPI.
    pub fn projected_cpi(&self, cpi0: f64, t: f64) -> f64 {
        cpi0 * self.cpi_ratio(t)
    }
}

/// Bootstrap helpers for inflation curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InflationCurveBuilder;

impl InflationCurveBuilder {
    /// Bootstraps an inflation curve from ZC inflation swap quotes `(tenor, fixed_rate)`.
    ///
    /// Assumes annual compounding so that `CPI(T)/CPI(0) = (1 + K)^T`.
    pub fn from_zc_swap_rates(zc_swap_rates: &[(f64, f64)]) -> InflationCurve {
        let nodes = zc_swap_rates
            .iter()
            .filter(|(tenor, rate)| *tenor > 0.0 && *rate > -1.0)
            .map(|(tenor, rate)| (*tenor, (1.0 + rate).powf(*tenor)))
            .collect();
        InflationCurve::new(nodes)
    }
}

/// Zero-coupon inflation swap (receive/pay realized inflation vs fixed).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ZeroCouponInflationSwap {
    pub notional: f64,
    pub cpi_base: f64,
    pub fixed_rate: f64,
    pub tenor: f64,
    /// If true, receives inflation and pays fixed.
    pub receive_inflation: bool,
}

impl ZeroCouponInflationSwap {
    /// NPV from a terminal CPI fixing.
    pub fn npv(&self, discount_curve: &YieldCurve, terminal_cpi: f64) -> f64 {
        if self.notional <= 0.0 || self.cpi_base <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let realized = terminal_cpi / self.cpi_base - 1.0;
        let fixed = (1.0 + self.fixed_rate).powf(self.tenor) - 1.0;
        let payoff = self.notional * (realized - fixed);
        let signed = if self.receive_inflation {
            payoff
        } else {
            -payoff
        };

        signed * discount_curve.discount_factor(self.tenor)
    }

    /// NPV from an inflation curve projection.
    pub fn npv_from_curve(
        &self,
        discount_curve: &YieldCurve,
        inflation_curve: &InflationCurve,
    ) -> f64 {
        let terminal_cpi = inflation_curve.projected_cpi(self.cpi_base, self.tenor);
        self.npv(discount_curve, terminal_cpi)
    }

    /// Mark-to-market NPV at valuation time with realized CPI to date.
    pub fn mtm(
        &self,
        valuation_time: f64,
        realized_cpi: f64,
        discount_curve: &YieldCurve,
        inflation_curve: &InflationCurve,
    ) -> f64 {
        if self.notional <= 0.0
            || self.cpi_base <= 0.0
            || realized_cpi <= 0.0
            || self.tenor <= valuation_time
        {
            return 0.0;
        }

        let forward_ratio = inflation_curve.forward_cpi_ratio(valuation_time.max(0.0), self.tenor);
        let projected_terminal_cpi = realized_cpi * forward_ratio;

        let realized = projected_terminal_cpi / self.cpi_base - 1.0;
        let fixed = (1.0 + self.fixed_rate).powf(self.tenor) - 1.0;
        let payoff = self.notional * (realized - fixed);
        let signed = if self.receive_inflation {
            payoff
        } else {
            -payoff
        };

        signed * discount_curve.discount_factor((self.tenor - valuation_time).max(0.0))
    }
}

/// Year-on-year inflation swap with annual settlements.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct YearOnYearInflationSwap {
    pub notional: f64,
    pub fixed_rate: f64,
    pub maturity_years: u32,
    /// If true, receives realized YoY inflation and pays fixed.
    pub receive_inflation: bool,
}

impl YearOnYearInflationSwap {
    /// NPV from explicit CPI fixings `CPI_0..CPI_N`.
    pub fn npv_from_fixings(&self, discount_curve: &YieldCurve, cpi_fixings: &[f64]) -> f64 {
        if self.notional <= 0.0 || self.maturity_years == 0 {
            return 0.0;
        }

        let needed = self.maturity_years as usize + 1;
        if cpi_fixings.len() < needed {
            return f64::NAN;
        }

        let mut pv = 0.0;
        for year in 1..=self.maturity_years as usize {
            let prev = cpi_fixings[year - 1];
            let curr = cpi_fixings[year];
            if prev <= 0.0 || curr <= 0.0 {
                return f64::NAN;
            }

            let yoy = curr / prev - 1.0;
            let cashflow = self.notional * (yoy - self.fixed_rate);
            let signed = if self.receive_inflation {
                cashflow
            } else {
                -cashflow
            };

            pv += signed * discount_curve.discount_factor(year as f64);
        }

        pv
    }

    /// NPV from forward inflation implied by the inflation curve.
    pub fn npv_from_curve(
        &self,
        discount_curve: &YieldCurve,
        inflation_curve: &InflationCurve,
    ) -> f64 {
        if self.notional <= 0.0 || self.maturity_years == 0 {
            return 0.0;
        }

        let mut pv = 0.0;
        for year in 1..=self.maturity_years {
            let t1 = (year - 1) as f64;
            let t2 = year as f64;
            let yoy = inflation_curve.forward_inflation_rate(t1, t2);
            let cashflow = self.notional * (yoy - self.fixed_rate);
            let signed = if self.receive_inflation {
                cashflow
            } else {
                -cashflow
            };

            pv += signed * discount_curve.discount_factor(t2);
        }

        pv
    }
}

/// Inflation-indexed bond (TIPS-style) with CPI-adjusted principal and coupons.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InflationIndexedBond {
    pub face_value: f64,
    pub coupon_rate: f64,
    pub maturity_years: u32,
    pub coupon_frequency: u32,
    pub cpi_base: f64,
}

impl InflationIndexedBond {
    /// Principal scaled by CPI ratio.
    pub fn indexed_principal(&self, cpi_level: f64) -> f64 {
        if self.cpi_base <= 0.0 {
            return f64::NAN;
        }

        self.face_value * cpi_level / self.cpi_base
    }

    /// Bond PV under nominal discounting and inflation-index projection.
    pub fn price(&self, nominal_curve: &YieldCurve, inflation_curve: &InflationCurve) -> f64 {
        if self.face_value <= 0.0
            || self.coupon_frequency == 0
            || self.maturity_years == 0
            || self.cpi_base <= 0.0
        {
            return 0.0;
        }

        let maturity = self.maturity_years as f64;
        let dt = 1.0 / self.coupon_frequency as f64;
        let mut t = dt;
        let mut pv = 0.0;

        while t < maturity - 1.0e-12 {
            pv += self.cashflow_at(t, inflation_curve) * nominal_curve.discount_factor(t);
            t += dt;
        }

        pv + self.cashflow_at(maturity, inflation_curve) * nominal_curve.discount_factor(maturity)
    }

    fn cashflow_at(&self, t: f64, inflation_curve: &InflationCurve) -> f64 {
        let cpi_t = inflation_curve.projected_cpi(self.cpi_base, t);
        let principal = self.indexed_principal(cpi_t);
        let mut cf = principal * self.coupon_rate / self.coupon_frequency as f64;

        if (t - self.maturity_years as f64).abs() <= 1.0e-10 {
            cf += principal;
        }

        cf
    }
}

fn ratio_from_points(points: &[(f64, f64)], t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    if points.is_empty() {
        return 1.0;
    }

    let first = points[0];
    if t <= first.0 {
        return log_linear_ratio(0.0, 1.0, first.0, first.1, t);
    }

    for window in points.windows(2) {
        let left = window[0];
        let right = window[1];
        if t <= right.0 {
            return log_linear_ratio(left.0, left.1, right.0, right.1, t);
        }
    }

    if points.len() == 1 {
        let (t1, ratio1) = points[0];
        let z = ratio1.ln() / t1;
        return (z * t).exp();
    }

    let left = points[points.len() - 2];
    let right = points[points.len() - 1];
    log_linear_ratio(left.0, left.1, right.0, right.1, t)
}

fn log_linear_ratio(t1: f64, r1: f64, t2: f64, r2: f64, t: f64) -> f64 {
    if (t2 - t1).abs() <= f64::EPSILON {
        return r2;
    }

    let w = (t - t1) / (t2 - t1);
    (r1.ln() + w * (r2.ln() - r1.ln())).exp()
}
