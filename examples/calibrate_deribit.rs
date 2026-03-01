//! Live Deribit SVI calibration CLI — fetches option chain and runs calibration.
//!
//! Usage:
//!   cargo run --example calibrate_deribit
//!   cargo run --example calibrate_deribit -- --currency ETH

use openferric::vol::surface::{SviParams, calibrate_svi_weighted};
use std::collections::BTreeMap;

const API_BASE: &str = "https://www.deribit.com/api/v2/public";

fn main() {
    let currency = std::env::args()
        .skip_while(|a| a != "--currency")
        .nth(1)
        .unwrap_or_else(|| "BTC".to_string());

    eprintln!("Fetching {currency} option chain from Deribit...");

    let url = format!("{API_BASE}/get_book_summary_by_currency?currency={currency}&kind=option");
    let resp: serde_json::Value = ureq::get(&url)
        .call()
        .unwrap()
        .body_mut()
        .read_json()
        .unwrap();

    let entries = resp["result"].as_array().expect("no result array");
    eprintln!("Got {} instruments", entries.len());

    // Fetch spot — Deribit index names: btc_usd, eth_usd, etc.
    let base = currency
        .split('-')
        .next()
        .unwrap_or(&currency)
        .to_lowercase();
    let index_name = if currency.contains("USDC") {
        format!("{base}_usdc")
    } else {
        format!("{base}_usd")
    };
    let spot_url = format!("{API_BASE}/get_index_price?index_name={index_name}");
    let spot_resp: serde_json::Value = ureq::get(&spot_url)
        .call()
        .unwrap()
        .body_mut()
        .read_json()
        .unwrap();
    let spot = spot_resp["result"]["index_price"].as_f64().unwrap_or(0.0);
    eprintln!("Spot: {spot:.2}\n");

    // Group by expiry
    struct Quote {
        strike: f64,
        mark_iv: f64,
        bid_iv: f64,
        ask_iv: f64,
        open_interest: f64,
        underlying_price: f64,
    }

    let mut by_expiry: BTreeMap<String, Vec<Quote>> = BTreeMap::new();
    let now_sec = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    for e in entries {
        let name = e["instrument_name"].as_str().unwrap_or("");
        let mark_iv = e["mark_iv"].as_f64().unwrap_or(0.0) / 100.0; // API returns %
        let bid_iv = e["bid_iv"].as_f64().unwrap_or(0.0) / 100.0;
        let ask_iv = e["ask_iv"].as_f64().unwrap_or(0.0) / 100.0;
        let oi = e["open_interest"].as_f64().unwrap_or(0.0);
        let underlying = e["underlying_price"].as_f64().unwrap_or(spot);

        if mark_iv <= 0.0 {
            continue;
        }

        // Parse instrument name: BTC-28MAR26-55000-C
        let parts: Vec<&str> = name.split('-').collect();
        if parts.len() < 4 {
            continue;
        }
        let expiry_code = parts[1].to_string();
        let strike: f64 = parts[2].parse().unwrap_or(0.0);
        if strike <= 0.0 {
            continue;
        }

        by_expiry.entry(expiry_code).or_default().push(Quote {
            strike,
            mark_iv,
            bid_iv,
            ask_iv,
            open_interest: oi,
            underlying_price: underlying,
        });
    }

    // Calibrate each expiry
    println!(
        "{:<12} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Expiry", "T(d)", "nPts", "a", "b", "rho", "m", "sigma", "RMSE(%)"
    );
    println!("{}", "-".repeat(90));

    for (expiry_code, quotes) in &by_expiry {
        if quotes.len() < 5 {
            continue;
        }

        // Parse expiry date
        let expiry_ts = parse_deribit_expiry(&expiry_code);
        if expiry_ts <= 0.0 {
            continue;
        }
        let t = (expiry_ts - now_sec) / (365.25 * 24.0 * 3600.0);
        if t <= 0.0 {
            continue;
        }

        let forward = quotes.iter().map(|q| q.underlying_price).sum::<f64>() / quotes.len() as f64;
        if forward <= 0.0 {
            continue;
        }

        // Quality filter (same as calibrate.rs)
        let mut filtered: Vec<usize> = Vec::new();
        for (i, q) in quotes.iter().enumerate() {
            if q.mark_iv <= 0.0 {
                continue;
            }
            if q.bid_iv <= 0.0 && q.open_interest <= 0.0 {
                continue;
            }
            if q.bid_iv > 0.0 && q.ask_iv > 0.0 {
                let spread = q.ask_iv - q.bid_iv;
                let mid = (q.ask_iv + q.bid_iv) * 0.5;
                if mid > 0.0 && spread / mid > 0.5 {
                    continue;
                }
            }
            filtered.push(i);
        }

        if filtered.len() < 5 {
            continue;
        }

        // ATM vol
        let mut atm_vol = 0.0_f64;
        let mut min_abs_k = f64::INFINITY;
        for &idx in &filtered {
            let q = &quotes[idx];
            let k = (q.strike / forward).ln();
            if k.abs() < min_abs_k {
                min_abs_k = k.abs();
                atm_vol = q.mark_iv;
            }
        }

        // Build weighted points (same logic as calibrate.rs)
        let decay_scale = 0.08_f64.max(0.5 * t.sqrt());
        let mut points: Vec<(f64, f64)> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();

        for &idx in &filtered {
            let q = &quotes[idx];
            let k = (q.strike / forward).ln();
            let iv2 = q.mark_iv * q.mark_iv;
            points.push((k, iv2));

            let spread_quality = if q.bid_iv > 0.0 && q.ask_iv > 0.0 {
                let rel_spread = (q.ask_iv - q.bid_iv) / ((q.ask_iv + q.bid_iv) * 0.5).max(1e-8);
                1.0 / (1.0 + rel_spread * 2.0)
            } else {
                0.5
            };
            let moneyness_decay = (-0.5 * (k / decay_scale).powi(2)).exp();
            let iv_ratio = q.mark_iv / atm_vol.max(0.01);
            let iv_penalty = 1.0 / (1.0 + (iv_ratio - 1.0).max(0.0).powi(2) * 16.0);
            weights.push(spread_quality * moneyness_decay * iv_penalty);
        }

        let atm_iv2 = (atm_vol * atm_vol).max(1e-4);
        let init = SviParams {
            a: atm_iv2 * 0.5,
            b: atm_iv2 * 1.5,
            rho: -0.1,
            m: 0.0,
            sigma: 0.15,
        };

        let result = calibrate_svi_weighted(&points, &weights, init, 150);

        // Compute RMSE in IV% (weighted)
        let mut sum_sq = 0.0;
        let mut sum_w = 0.0;
        for (i, (k, iv2)) in points.iter().enumerate() {
            let fitted_iv2 = result.total_variance(*k);
            let market_iv = iv2.sqrt() * 100.0;
            let fitted_iv = fitted_iv2.max(0.0).sqrt() * 100.0;
            let w = weights[i];
            sum_sq += w * (fitted_iv - market_iv).powi(2);
            sum_w += w;
        }
        let rmse = if sum_w > 0.0 {
            (sum_sq / sum_w).sqrt()
        } else {
            f64::NAN
        };

        let days = t * 365.25;
        println!(
            "{:<12} {:>6.1} {:>8} {:>8.5} {:>8.5} {:>8.4} {:>8.4} {:>8.4} {:>10.3}",
            expiry_code,
            days,
            filtered.len(),
            result.a,
            result.b,
            result.rho,
            result.m,
            result.sigma,
            rmse
        );

        // Show per-point detail for short-dated slices
        if days < 20.0 {
            println!(
                "  {:>10} {:>10} {:>10} {:>10} {:>10}",
                "strike", "k", "mkt_iv%", "fit_iv%", "weight"
            );
            let mut idxs: Vec<usize> = (0..points.len()).collect();
            idxs.sort_by(|a, b| points[*a].0.partial_cmp(&points[*b].0).unwrap());
            for &i in &idxs {
                let (k, iv2) = points[i];
                let mkt_iv = iv2.sqrt() * 100.0;
                let fit_iv = result.total_variance(k).max(0.0).sqrt() * 100.0;
                let w = weights[i];
                if w > 0.001 {
                    println!(
                        "  {:>10.0} {:>10.4} {:>10.2} {:>10.2} {:>10.4}",
                        quotes[filtered[i]].strike, k, mkt_iv, fit_iv, w
                    );
                }
            }
            println!();
        }
    }
}

fn parse_deribit_expiry(code: &str) -> f64 {
    // Parse "28MAR26" -> unix timestamp at 08:00 UTC on that date
    if code.len() < 5 {
        return 0.0;
    }
    let day: u32 = code[..code.len() - 5].parse().unwrap_or(0);
    let month_str = &code[code.len() - 5..code.len() - 2];
    let year_suffix: u32 = code[code.len() - 2..].parse().unwrap_or(0);
    let year = 2000 + year_suffix;
    let month = match month_str {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return 0.0,
    };

    // Rough unix timestamp (good enough for year fraction)
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
    let date = NaiveDate::from_ymd_opt(year as i32, month, day);
    let Some(date) = date else { return 0.0 };
    let time = NaiveTime::from_hms_opt(8, 0, 0).unwrap();
    let dt = NaiveDateTime::new(date, time);
    dt.and_utc().timestamp() as f64
}
