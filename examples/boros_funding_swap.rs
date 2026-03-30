//! Comprehensive Pendle Boros funding-rate swap walkthrough.
//!
//! Pendle Boros tokenizes the right to receive or pay future perpetual funding
//! into tradable units. Economically, that looks like a funding-rate swap:
//! one side pays a locked funding APR and receives the realized floating
//! funding that exchanges such as Binance or Bybit settle every 8 hours.
//!
//! This example is the end-to-end reference for OpenFerric's funding-rate
//! stack. It shows how to:
//!
//! - turn discrete exchange funding snapshots into a term structure,
//! - aggregate several venues into a Boros-style composite market view,
//! - price a long-funding swap position from historical fixings plus forward
//!   expectations,
//! - compute standard bump-and-roll sensitivities,
//! - translate the rate view into Boros margin and leverage analytics, and
//! - estimate liquidation risk under mean-reverting funding-rate dynamics.
//!
//! The mapping between Boros concepts and OpenFerric types is:
//!
//! - [`openferric::rates::FundingRateSnapshot`] stores one exchange fixing for
//!   a single 8-hour funding window.
//! - [`openferric::rates::FundingRateCurve`] interpolates those fixings into a
//!   forward funding term structure in year-fraction time.
//! - [`openferric::rates::MultiVenueFundingCurve`] combines venue curves with
//!   explicit weights, which is useful when a desk wants a composite index
//!   rather than a single-exchange view.
//! - [`openferric::instruments::FundingRateSwap`] models the discrete swap cash
//!   flows that Pendle Boros traders are effectively taking.
//! - [`openferric::risk::MarginCalculator`] and
//!   [`openferric::risk::LiquidationSimulator`] model the isolated-margin
//!   mechanics around a Boros position once the trade is on the books.
//!
//! Real-world context:
//!
//! - Binance BTCUSDT perpetuals settle funding every 8 hours at 00:00, 08:00,
//!   and 16:00 UTC.
//! - Boros traders typically monitor funding in APR terms, but exchange feeds
//!   publish a per-settlement rate. OpenFerric supports both views.
//! - Boros margin utilities use a signed-rate-exposure convention where a
//!   negative position size represents "long funding" exposure. The swap pricer
//!   instead uses positive notional for receive-floating, pay-fixed cash flows.

use std::collections::HashMap;
use std::error::Error;

use chrono::{DateTime, Duration, NaiveDate, Utc};
use openferric::instruments::FundingRateSwap;
use openferric::models::short_rate::Vasicek;
use openferric::pricing::funding_rate_swap::{
    funding_rate_swap_dv01, funding_rate_swap_mtm, funding_rate_swap_risks, funding_rate_swap_theta,
};
use openferric::rates::{
    FundingRateCurve, FundingRateSnapshot, FundingRateStats, MultiVenueFundingCurve,
};
use openferric::risk::{
    FundingRateModel, InherentLeverage, LiquidationPosition, LiquidationSimulator,
    MarginCalculator, MarginParams, StressScenario,
};

const HOURS_PER_YEAR: f64 = 8_760.0;
const BINANCE_WEIGHT: f64 = 0.65;
const BYBIT_WEIGHT: f64 = 0.35;

/// Holds the single-venue, multi-venue, and synthetic pricing curves used in
/// the rest of the example.
#[derive(Debug, Clone)]
struct MarketContext {
    binance_curve: FundingRateCurve,
    bybit_curve: FundingRateCurve,
    composite_curve: MultiVenueFundingCurve,
    pricing_curve: FundingRateCurve,
}

/// Stores the pricing inputs and interval-by-interval swap decomposition.
#[derive(Debug, Clone)]
struct SwapAnalysis {
    swap: FundingRateSwap,
    as_of: DateTime<Utc>,
    realized_fixings: Vec<(DateTime<Utc>, f64)>,
    realized_pnl: f64,
    mtm: f64,
    intervals: Vec<IntervalBreakdown>,
}

/// One funding settlement interval, tagged as historical or forward-looking.
#[derive(Debug, Clone)]
struct IntervalBreakdown {
    settlement: DateTime<Utc>,
    source: &'static str,
    floating_apr: f64,
    interval_pnl: f64,
}

/// Margin and liquidation inputs for a Boros Yield Unit position.
#[derive(Debug, Clone, Copy)]
struct YUPositionContext {
    position: LiquidationPosition,
    current_rate: f64,
    initial_margin: f64,
    maintenance_margin: f64,
    health_ratio: f64,
    liquidation_rate: f64,
    leverage: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    print_banner();

    let market = build_reference_market_curves();
    display_binance_term_structure(&market.binance_curve);
    display_multi_venue_curve(&market);

    let swap_analysis = analyze_reference_swap(&market.pricing_curve)?;
    display_swap_pricing(&swap_analysis);
    display_swap_risks(
        &swap_analysis.swap,
        &market.pricing_curve,
        swap_analysis.as_of,
    );

    let yu_context = display_margin_and_leverage(&swap_analysis, &market.pricing_curve);
    display_liquidation_simulation(yu_context);

    Ok(())
}

/// Builds deterministic Binance and Bybit funding histories, then combines them
/// into both a weighted multi-venue curve and a synthetic single curve for the
/// pricing APIs that require [`FundingRateCurve`].
///
/// Why it matters:
/// Boros markets are usually discussed as one funding view, but in practice
/// desks often blend several venues to reduce idiosyncratic exchange noise.
/// OpenFerric exposes both levels: a true multi-venue curve for analytics and a
/// blended snapshot curve for pricers that consume a single term structure.
fn build_reference_market_curves() -> MarketContext {
    let anchor = utc_datetime(2026, 3, 18, 0);
    let binance_rates = vec![
        0.00010, 0.00011, 0.00010, 0.00011, 0.00012, 0.00011, 0.00012, 0.00012, 0.00013, 0.00012,
        0.00013, 0.00012, 0.00013, 0.00014, 0.00013, 0.00014, 0.00015, 0.00014, 0.00015, 0.00015,
        0.00016, 0.00015, 0.00016, 0.00017, 0.00016, 0.00017, 0.00018, 0.00017, 0.00016, 0.00017,
    ];

    let binance_snapshots = build_snapshots("Binance", "BTCUSDT", anchor, &binance_rates);
    let bybit_snapshots = build_bybit_snapshots(anchor, &binance_rates);
    let blended_snapshots = blend_snapshots(&binance_snapshots, &bybit_snapshots);

    let binance_curve = FundingRateCurve::new(binance_snapshots);
    let bybit_curve = FundingRateCurve::new(bybit_snapshots);
    let composite_curve = MultiVenueFundingCurve::new(vec![
        (binance_curve.clone(), BINANCE_WEIGHT),
        (bybit_curve.clone(), BYBIT_WEIGHT),
    ]);
    let pricing_curve = FundingRateCurve::new(blended_snapshots);

    MarketContext {
        binance_curve,
        bybit_curve,
        composite_curve,
        pricing_curve,
    }
}

/// Displays the single-venue Binance curve, including forward rates, APR
/// conversion, cumulative funding accrual, and rolling distribution statistics.
///
/// Why it matters:
/// This is the core object behind Boros pricing. Traders quote funding in APR,
/// but settlement happens every 8 hours, so the first task is always converting
/// exchange snapshots into a smooth forward curve and a set of descriptive
/// statistics.
fn display_binance_term_structure(curve: &FundingRateCurve) {
    print_section("1. Binance BTCUSDT funding term structure");

    let anchor = curve.anchor_timestamp().expect("curve has snapshots");
    let latest_snapshot = curve.snapshots().last().expect("curve has snapshots");
    let latest_apr = FundingRateCurve::per_period_rate_to_apr(latest_snapshot.rate);
    let history_horizon = curve.nodes().last().map(|(t, _)| *t).unwrap_or(0.0);
    let cumulative_index = curve.cumulative_index(history_horizon);
    let overall_stats =
        FundingRateStats::from_rates(&curve.snapshots().iter().map(|s| s.rate).collect::<Vec<_>>());
    let rolling_stats = curve.rolling_stats(9);

    println!(
        "Anchor UTC: {} | snapshots: {:>2} | latest settlement: {}",
        fmt_datetime(anchor),
        curve.snapshots().len(),
        fmt_datetime(latest_snapshot.timestamp)
    );
    println!(
        "Latest Binance funding: {:>8.5}% APR from {:>9.6} per 8h",
        latest_apr * 100.0,
        latest_snapshot.rate
    );
    println!(
        "Cumulative funding index over observed history: {:>8.5}% | discount factor: {:>8.6}",
        cumulative_index * 100.0,
        curve.discount_factor(history_horizon)
    );
    println!();

    println!(
        "{:<10} {:>10} {:>14} {:>12}",
        "Tenor", "Days", "8h forward", "APR"
    );
    println!("{}", "-".repeat(52));
    for (label, days) in [
        ("8h", 8.0 / 24.0),
        ("1d", 1.0),
        ("3d", 3.0),
        ("7d", 7.0),
        ("10d", 10.0),
    ] {
        let t = hours_to_years(days * 24.0);
        let per_period = curve.forward_rate(t);
        let apr = FundingRateCurve::per_period_rate_to_apr(per_period);
        println!(
            "{:<10} {:>10.3} {:>14.6} {:>11.4}%",
            label,
            days,
            per_period,
            apr * 100.0
        );
    }
    println!();

    println!(
        "Overall stats on per-8h Binance rates: mean {:>9.6}, vol {:>9.6}, skew {:>8.4}, kurtosis {:>8.4}",
        overall_stats.mean, overall_stats.vol, overall_stats.skew, overall_stats.kurtosis
    );
    println!(
        "{:<18} {:>11} {:>11} {:>9} {:>10}",
        "Rolling window end", "Mean", "Vol", "Skew", "Kurtosis"
    );
    println!("{}", "-".repeat(66));
    for (timestamp, stats) in rolling_stats.iter().rev().take(5).rev() {
        println!(
            "{:<18} {:>11.6} {:>11.6} {:>9.4} {:>10.4}",
            fmt_datetime(*timestamp),
            stats.mean,
            stats.vol,
            stats.skew,
            stats.kurtosis
        );
    }
    println!();
}

/// Displays how Binance and Bybit can be combined into a weighted composite
/// market view.
///
/// Why it matters:
/// Boros desks often price off a venue basket rather than a single exchange.
/// [`MultiVenueFundingCurve`] keeps those venue weights explicit, while the
/// synthetic pricing curve below shows how to collapse the basket back into a
/// single curve for engines that need one term structure input.
fn display_multi_venue_curve(market: &MarketContext) {
    print_section("2. Multi-venue Boros market view");

    println!(
        "Weights: Binance {:>5.1}% | Bybit {:>5.1}%",
        BINANCE_WEIGHT * 100.0,
        BYBIT_WEIGHT * 100.0
    );
    println!(
        "{:<10} {:>12} {:>12} {:>12}",
        "Tenor", "Binance APR", "Bybit APR", "Composite APR"
    );
    println!("{}", "-".repeat(54));
    for (label, days) in [("1d", 1.0), ("3d", 3.0), ("7d", 7.0), ("10d", 10.0)] {
        let t = hours_to_years(days * 24.0);
        let binance_apr =
            FundingRateCurve::per_period_rate_to_apr(market.binance_curve.forward_rate(t));
        let bybit_apr =
            FundingRateCurve::per_period_rate_to_apr(market.bybit_curve.forward_rate(t));
        let composite_apr =
            FundingRateCurve::per_period_rate_to_apr(market.composite_curve.forward_rate(t));
        println!(
            "{:<10} {:>11.4}% {:>11.4}% {:>11.4}%",
            label,
            binance_apr * 100.0,
            bybit_apr * 100.0,
            composite_apr * 100.0
        );
    }

    let seven_day_index = market.composite_curve.cumulative_index(7.0 / 365.0);
    println!();
    println!(
        "Seven-day weighted cumulative funding index: {:>8.5}%",
        seven_day_index * 100.0
    );
    println!(
        "Synthetic pricing curve latest blended APR: {:>8.5}%",
        FundingRateCurve::per_period_rate_to_apr(
            market
                .pricing_curve
                .snapshots()
                .last()
                .expect("snapshots")
                .rate,
        ) * 100.0
    );
    println!();
}

/// Builds a receive-floating/pay-fixed funding-rate swap and splits its value
/// into realized historical cash flows and forward-looking MTM.
///
/// Why it matters:
/// This mirrors how a Boros trader thinks about PnL. Past settlements are
/// fixed cash flows, while future settlements depend on the funding term
/// structure. The schedule is aligned to UTC 8-hour boundaries because that is
/// how major perpetual venues settle funding in production.
fn analyze_reference_swap(curve: &FundingRateCurve) -> Result<SwapAnalysis, Box<dyn Error>> {
    let swap = FundingRateSwap::new(
        5_000_000.0,
        0.10,
        utc_datetime(2026, 3, 24, 3),
        utc_datetime(2026, 3, 29, 0),
        "Boros blended index",
        "BTCUSDT",
    );
    swap.validate()?;

    let as_of = utc_datetime(2026, 3, 27, 0);
    let fixing_lookup = curve
        .snapshots()
        .iter()
        .map(|snapshot| {
            (
                snapshot.timestamp.timestamp(),
                FundingRateCurve::per_period_rate_to_apr(snapshot.rate),
            )
        })
        .collect::<HashMap<_, _>>();

    let realized_fixings = swap
        .settlement_schedule()
        .into_iter()
        .filter(|settlement| *settlement <= as_of)
        .filter_map(|settlement| {
            fixing_lookup
                .get(&settlement.timestamp())
                .copied()
                .map(|apr| (settlement, apr))
        })
        .collect::<Vec<_>>();

    let realized_pnl = swap.realized_pnl(&realized_fixings);
    let mtm = funding_rate_swap_mtm(&swap, curve, as_of);
    let intervals = build_interval_breakdown(&swap, curve, as_of, &fixing_lookup);

    Ok(SwapAnalysis {
        swap,
        as_of,
        realized_fixings,
        realized_pnl,
        mtm,
        intervals,
    })
}

/// Prints the settlement schedule, realized fixings, and forward MTM for the
/// reference receive-floating swap.
///
/// Why it matters:
/// Pendle Boros PnL is discrete. Looking only at headline MTM hides which
/// intervals have already crystallized and which are still forward exposure.
/// The interval table below makes that distinction explicit.
fn display_swap_pricing(analysis: &SwapAnalysis) {
    print_section("3. Funding-rate swap pricing");

    println!(
        "Trade: receive floating / pay fixed | notional {:>12.0} | fixed {:>6.2}% APR",
        analysis.swap.notional,
        analysis.swap.fixed_rate * 100.0
    );
    println!(
        "Entry UTC: {} | As of UTC: {} | Maturity UTC: {}",
        fmt_datetime(analysis.swap.entry_time),
        fmt_datetime(analysis.as_of),
        fmt_datetime(analysis.swap.maturity)
    );
    println!();

    println!(
        "{:<3} {:<18} {:<18} {:<10}",
        "#", "Interval start", "Settlement", "Status"
    );
    println!("{}", "-".repeat(58));
    for (idx, settlement) in analysis.swap.settlement_schedule().iter().enumerate() {
        let status = if *settlement <= analysis.as_of {
            "Realized"
        } else {
            "Forward"
        };
        println!(
            "{:<3} {:<18} {:<18} {:<10}",
            idx + 1,
            fmt_datetime(*settlement - Duration::hours(8)),
            fmt_datetime(*settlement),
            status
        );
    }
    println!();

    println!(
        "{:<18} {:<10} {:>12} {:>14}",
        "Settlement", "Source", "Float APR", "Interval PnL"
    );
    println!("{}", "-".repeat(60));
    for row in &analysis.intervals {
        println!(
            "{:<18} {:<10} {:>11.4}% {:>14.2}",
            fmt_datetime(row.settlement),
            row.source,
            row.floating_apr * 100.0,
            row.interval_pnl
        );
    }
    println!();

    println!(
        "Known historical fixings used: {:>2} | realized PnL: {:>10.2}",
        analysis.realized_fixings.len(),
        analysis.realized_pnl
    );
    println!(
        "Remaining swap MTM from the forward curve: {:>10.2}",
        analysis.mtm
    );
    println!(
        "Total economic value (realized + MTM): {:>10.2}",
        analysis.realized_pnl + analysis.mtm
    );
    println!();
}

/// Computes and prints standard swap sensitivities from
/// [`openferric::pricing::funding_rate_swap`].
///
/// Why it matters:
/// Boros traders need to know how much PV moves for a 1bp parallel funding
/// shift and how much value decays when the next 8-hour settlement rolls off.
/// The snapshot curve has no explicit volatility state, so vega remains zero in
/// this example even though the risk report still includes it for API
/// consistency.
fn display_swap_risks(swap: &FundingRateSwap, curve: &FundingRateCurve, as_of: DateTime<Utc>) {
    print_section("4. Funding swap risk sensitivities");

    let dv01 = funding_rate_swap_dv01(swap, curve, as_of);
    let theta = funding_rate_swap_theta(swap, curve, as_of);
    let report = funding_rate_swap_risks(swap, curve, as_of);

    println!("Standalone checks:");
    println!("  DV01 (1bp parallel APR shift): {:>10.4}", dv01);
    println!("  Theta (one 8h roll-down):      {:>10.4}", theta);
    println!();
    println!(
        "{:<14} {:>12} | {:<14} {:>12}",
        "MTM",
        format!("{:.4}", report.mtm),
        "DV01",
        format!("{:.4}", report.dv01)
    );
    println!(
        "{:<14} {:>12} | {:<14} {:>12}",
        "Vega",
        format!("{:.4}", report.vega),
        "Theta",
        format!("{:.4}", report.theta)
    );
    println!();
}

/// Translates the funding view into Boros-style isolated-margin analytics for a
/// freshly opened Yield Unit position.
///
/// Why it matters:
/// The swap above is the cleanest pricing representation, but Boros margin is
/// enforced on signed rate exposure. In OpenFerric's liquidation module a
/// negative `size` is a "long funding" position, which matches the trader's
/// view of owning the Yield Unit side.
fn display_margin_and_leverage(
    analysis: &SwapAnalysis,
    curve: &FundingRateCurve,
) -> YUPositionContext {
    print_section("5. Margin and leverage analysis");

    let current_rate = curve.expected_rate(
        analysis.as_of,
        analysis.as_of,
        analysis.as_of + Duration::hours(8),
    );
    let margin_params = MarginParams {
        initial_margin_ratio: 0.18,
        maintenance_margin_ratio: 0.12,
        funding_rate_vol: 0.20,
        time_to_maturity: hours_to_years(30.0 * 24.0),
        tick_size: 0.0001,
    };

    let notional = analysis.swap.notional;
    let initial_margin = MarginCalculator::initial_margin(notional, &margin_params);
    let maintenance_margin = MarginCalculator::maintenance_margin(notional, &margin_params);
    let collateral = initial_margin * 1.20;

    let position = LiquidationPosition {
        size: -notional,
        entry_rate: current_rate,
        collateral,
        margin_params,
    };
    let health_ratio = MarginCalculator::health_ratio(collateral, notional, 0.0, &margin_params);
    let liquidation_rate = MarginCalculator::liquidation_rate(
        position.entry_rate,
        collateral,
        position.size,
        &margin_params,
    );
    let yu_cost = 125_000.0;
    let leverage = InherentLeverage::leverage(notional, yu_cost);

    println!(
        "Fresh YU entry rate: {:>8.4}% APR | signed Boros size: {:>12.0}",
        current_rate * 100.0,
        position.size
    );
    println!(
        "Initial margin: {:>11.2} | maintenance margin: {:>11.2} | collateral posted: {:>11.2}",
        initial_margin, maintenance_margin, collateral
    );
    println!(
        "Health ratio at entry: {:>8.4} | liquidation funding rate: {:>8.4}% APR",
        health_ratio,
        liquidation_rate * 100.0
    );
    println!(
        "Assumed YU upfront cost: {:>11.2} | inherent leverage: {:>8.2}x",
        yu_cost, leverage
    );
    println!(
        "A 100bp funding move implies roughly {:>7.2}% levered return on YU cost",
        InherentLeverage::leveraged_return(0.01, leverage) * 100.0
    );
    println!();

    YUPositionContext {
        position,
        current_rate,
        initial_margin,
        maintenance_margin,
        health_ratio,
        liquidation_rate,
        leverage,
    }
}

/// Runs a Monte Carlo liquidation study under a mean-reverting Vasicek funding
/// model and under several stress scenarios.
///
/// Why it matters:
/// Boros positions are path-dependent from a margin perspective. A desk can be
/// right on the long-run funding view and still get liquidated by a temporary
/// adverse move. This section estimates first-passage liquidation probability
/// under baseline and stress assumptions.
fn display_liquidation_simulation(context: YUPositionContext) {
    print_section("6. Monte Carlo liquidation simulation");

    let long_run_mean = context.current_rate - 0.005;
    let sigma = 0.06;
    let model = FundingRateModel::Vasicek(Vasicek {
        a: 8.0,
        b: long_run_mean,
        sigma,
    });
    let simulator =
        LiquidationSimulator::new(context.position, model, context.current_rate, 5_000, 90, 7);

    let mut scenarios = vec![StressScenario::Baseline];
    scenarios.extend(StressScenario::cascade_suite());
    scenarios.extend(StressScenario::mean_shift_suite(0.03));

    println!(
        "Margin state carried into simulation: init {:>10.2} | maint {:>10.2} | health {:>6.3} | leverage {:>5.2}x",
        context.initial_margin, context.maintenance_margin, context.health_ratio, context.leverage
    );
    println!(
        "Vasicek model: a = {:.2}, b = {:.2}%, sigma = {:.2}% | paths = {} | steps = {}",
        8.0,
        long_run_mean * 100.0,
        sigma * 100.0,
        simulator.num_paths,
        simulator.steps
    );
    println!(
        "{:<24} {:>12} {:>16} {:>18}",
        "Scenario", "Liq prob", "E[liq days]", "Worst funding APR"
    );
    println!("{}", "-".repeat(76));
    for result in simulator.run_stress_scenarios(&scenarios) {
        let days = result.risk.expected_time_to_liquidation.map(|t| t * 365.0);
        println!(
            "{:<24} {:>11.2}% {:>16} {:>17.4}%",
            scenario_label(result.scenario),
            result.risk.prob_liquidation * 100.0,
            format_optional(days),
            result.risk.worst_case_funding_rate * 100.0
        );
    }
    println!();
    println!(
        "Liquidation barrier from the margin model: {:>8.4}% APR | current entry rate: {:>8.4}% APR",
        context.liquidation_rate * 100.0,
        context.current_rate * 100.0
    );
    println!();
}

fn build_interval_breakdown(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: DateTime<Utc>,
    fixing_lookup: &HashMap<i64, f64>,
) -> Vec<IntervalBreakdown> {
    swap.settlement_schedule()
        .into_iter()
        .map(|settlement| {
            let interval_start =
                settlement - Duration::hours(i64::from(swap.settlement_interval_hours));
            let (source, floating_apr) = if settlement <= as_of {
                let apr = fixing_lookup
                    .get(&settlement.timestamp())
                    .copied()
                    .expect("historical fixing exists");
                ("Realized", apr)
            } else {
                (
                    "Forward",
                    curve.expected_rate(as_of, interval_start, settlement),
                )
            };

            IntervalBreakdown {
                settlement,
                source,
                floating_apr,
                interval_pnl: FundingRateSwap::interval_pnl(
                    swap.fixed_rate,
                    floating_apr,
                    swap.notional,
                ),
            }
        })
        .collect()
}

fn build_snapshots(
    venue: &str,
    asset: &str,
    anchor: DateTime<Utc>,
    rates: &[f64],
) -> Vec<FundingRateSnapshot> {
    rates
        .iter()
        .enumerate()
        .map(|(idx, rate)| FundingRateSnapshot {
            venue: venue.to_string(),
            asset: asset.to_string(),
            rate: *rate,
            timestamp: anchor + Duration::hours((idx as i64) * 8),
        })
        .collect()
}

fn build_bybit_snapshots(
    anchor: DateTime<Utc>,
    reference_rates: &[f64],
) -> Vec<FundingRateSnapshot> {
    let bybit_rates = reference_rates
        .iter()
        .enumerate()
        .map(|(idx, rate)| {
            let basis_adjustment = match idx % 6 {
                0 => -0.000004,
                1 => 0.000001,
                2 => -0.000002,
                3 => 0.000002,
                4 => -0.000001,
                _ => 0.000001,
            };
            (rate * 0.96 + basis_adjustment).max(0.000095)
        })
        .collect::<Vec<_>>();

    build_snapshots("Bybit", "BTCUSDT", anchor, &bybit_rates)
}

fn blend_snapshots(
    binance_snapshots: &[FundingRateSnapshot],
    bybit_snapshots: &[FundingRateSnapshot],
) -> Vec<FundingRateSnapshot> {
    binance_snapshots
        .iter()
        .zip(bybit_snapshots.iter())
        .map(|(binance, bybit)| FundingRateSnapshot {
            venue: "BorosComposite".to_string(),
            asset: binance.asset.clone(),
            rate: (BINANCE_WEIGHT * binance.rate + BYBIT_WEIGHT * bybit.rate)
                / (BINANCE_WEIGHT + BYBIT_WEIGHT),
            timestamp: binance.timestamp,
        })
        .collect()
}

fn scenario_label(scenario: StressScenario) -> String {
    match scenario {
        StressScenario::Baseline => "Baseline".to_string(),
        StressScenario::LiquidationCascade { vol_multiplier } => {
            format!("Cascade vol x{vol_multiplier:.0}")
        }
        StressScenario::MeanShift { shift } => format!("Mean shift {:+.1}%", shift * 100.0),
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_banner() {
    println!();
    println!("{}", "=".repeat(76));
    println!("Pendle Boros Funding Rate Swap Reference Example");
    println!("{}", "=".repeat(76));
    println!();
}

fn print_section(title: &str) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
}

fn fmt_datetime(dt: DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M").to_string()
}

fn utc_datetime(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
    DateTime::from_naive_utc_and_offset(
        NaiveDate::from_ymd_opt(year, month, day)
            .expect("valid date")
            .and_hms_opt(hour, 0, 0)
            .expect("valid time"),
        Utc,
    )
}

fn hours_to_years(hours: f64) -> f64 {
    hours / HOURS_PER_YEAR
}
