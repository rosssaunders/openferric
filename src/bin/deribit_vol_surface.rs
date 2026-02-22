//! Command-line entry point for Deribit Vol Surface workflows.
//!
//! This binary wires OpenFerric models and engines into an executable utility.

#[cfg(feature = "deribit")]
mod app {
    use chrono::NaiveDate;
    use openferric::vol::surface::{SviParams, calibrate_svi};
    use reqwest::Client;
    use serde_json::Value;
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::error::Error;
    use std::fmt::Write as _;
    use std::fs;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    const BOOK_SUMMARY_URL: &str =
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency";
    const INDEX_PRICE_URL: &str = "https://www.deribit.com/api/v2/public/get_index_price";

    type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum OptionType {
        Call,
        Put,
    }

    #[derive(Debug, Clone)]
    struct OptionQuote {
        expiry: NaiveDate,
        strike: f64,
        option_type: OptionType,
        mark_iv: f64,
        underlying_price: f64,
        time_to_expiry: f64,
        log_moneyness: f64,
    }

    #[derive(Debug, Clone)]
    struct CalibratedSlice {
        expiry: NaiveDate,
        time_to_expiry: f64,
        options_count: usize,
        atm_vol: f64,
        rr25: Option<f64>,
        rmse_vol_points: f64,
        params: SviParams,
        quotes: Vec<OptionQuote>,
    }

    #[derive(Debug, Clone)]
    struct VolMatrix {
        expiries: Vec<NaiveDate>,
        times: Vec<f64>,
        strikes: Vec<f64>,
        vols: Vec<Vec<Option<f64>>>,
        log_moneyness: Vec<Vec<Option<f64>>>,
    }

    async fn fetch_book_summaries(client: &Client) -> AppResult<Vec<Value>> {
        let payload = client
            .get(BOOK_SUMMARY_URL)
            .query(&[("currency", "BTC"), ("kind", "option")])
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        if let Some(err) = payload.get("error") {
            return Err(format!("Deribit returned error for book summary: {err}").into());
        }

        let entries = payload
            .get("result")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                "missing or invalid `result` array in book summary response".to_string()
            })?;

        Ok(entries.clone())
    }

    async fn fetch_spot_price(client: &Client) -> AppResult<f64> {
        let payload = client
            .get(INDEX_PRICE_URL)
            .query(&[("index_name", "btc_usd")])
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        if let Some(err) = payload.get("error") {
            return Err(format!("Deribit returned error for index price: {err}").into());
        }

        payload
            .get("result")
            .and_then(|v| v.get("index_price"))
            .and_then(Value::as_f64)
            .ok_or_else(|| "missing `result.index_price` in index response".into())
    }

    fn parse_instrument_name(name: &str) -> Option<(NaiveDate, f64, OptionType)> {
        let parts: Vec<&str> = name.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let expiry = parse_deribit_expiry(parts[1])?;
        let strike = parts[2].parse::<f64>().ok()?;
        let option_type = match parts[3] {
            "C" => OptionType::Call,
            "P" => OptionType::Put,
            _ => return None,
        };
        Some((expiry, strike, option_type))
    }

    fn parse_deribit_expiry(code: &str) -> Option<NaiveDate> {
        if code.len() != 7 {
            return None;
        }

        let day = code[0..2].parse::<u32>().ok()?;
        let month = match &code[2..5] {
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
            _ => return None,
        };
        let yy = code[5..7].parse::<i32>().ok()?;
        let year = 2000 + yy;

        NaiveDate::from_ymd_opt(year, month, day)
    }

    fn field_f64(v: &Value, key: &str) -> Option<f64> {
        v.get(key).and_then(Value::as_f64)
    }

    fn now_unix_seconds() -> AppResult<f64> {
        Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64())
    }

    fn build_quotes(entries: &[Value], spot_price: f64, now_secs: f64) -> Vec<OptionQuote> {
        let mut quotes = Vec::with_capacity(entries.len());

        for entry in entries {
            let Some(instrument_name) = entry.get("instrument_name").and_then(Value::as_str) else {
                continue;
            };
            let Some((expiry, strike, option_type)) = parse_instrument_name(instrument_name) else {
                continue;
            };

            let Some(mark_iv_pct) = field_f64(entry, "mark_iv") else {
                continue;
            };
            let Some(bid_price) = field_f64(entry, "bid_price") else {
                continue;
            };
            let Some(ask_price) = field_f64(entry, "ask_price") else {
                continue;
            };
            let Some(open_interest) = field_f64(entry, "open_interest") else {
                continue;
            };

            if bid_price <= 0.0 || ask_price <= 0.0 || open_interest <= 0.0 || mark_iv_pct <= 0.0 {
                continue;
            }

            let underlying_price = field_f64(entry, "underlying_price").unwrap_or(spot_price);
            if underlying_price <= 0.0 || strike <= 0.0 {
                continue;
            }

            let Some(expiry_ts) = expiry
                .and_hms_opt(8, 0, 0)
                .map(|ts| ts.and_utc().timestamp() as f64)
            else {
                continue;
            };

            let t = (expiry_ts - now_secs) / (365.25 * 24.0 * 3600.0);
            if t <= 0.0 {
                continue;
            }

            let mark_iv = mark_iv_pct / 100.0;
            let log_moneyness = (strike / underlying_price).ln();
            let _mark_price = field_f64(entry, "mark_price").unwrap_or(0.0);
            let _interest_rate = field_f64(entry, "interest_rate").unwrap_or(0.0);

            quotes.push(OptionQuote {
                expiry,
                strike,
                option_type,
                mark_iv,
                underlying_price,
                time_to_expiry: t,
                log_moneyness,
            });
        }

        quotes
    }

    fn group_by_expiry(quotes: Vec<OptionQuote>) -> BTreeMap<NaiveDate, Vec<OptionQuote>> {
        let mut grouped: BTreeMap<NaiveDate, Vec<OptionQuote>> = BTreeMap::new();
        for quote in quotes {
            grouped.entry(quote.expiry).or_default().push(quote);
        }
        grouped
    }

    fn calibrate_slice(expiry: NaiveDate, mut quotes: Vec<OptionQuote>) -> Option<CalibratedSlice> {
        if quotes.len() < 5 {
            return None;
        }

        quotes.sort_by(|a, b| a.strike.total_cmp(&b.strike));
        let t = quotes[0].time_to_expiry;
        if t <= 0.0 {
            return None;
        }

        let points: Vec<(f64, f64)> = quotes
            .iter()
            .map(|q| (q.log_moneyness, q.mark_iv * q.mark_iv * t))
            .collect();
        if points.len() < 5 {
            return None;
        }

        let atm_vol = quotes
            .iter()
            .min_by(|a, b| a.log_moneyness.abs().total_cmp(&b.log_moneyness.abs()))
            .map(|q| q.mark_iv)
            .unwrap_or(0.0);

        let atm_w = (atm_vol * atm_vol * t).max(1e-6);
        let init = SviParams {
            a: atm_w * 0.4,
            b: atm_w * 0.8,
            rho: -0.2,
            m: 0.0,
            sigma: 0.25,
        };
        let params = calibrate_svi(&points, init, 600, 0.01);
        let rmse_vol_points = rmse_vol_points(&quotes, params, t);
        let rr25 = risk_reversal_25d(&quotes);

        Some(CalibratedSlice {
            expiry,
            time_to_expiry: t,
            options_count: quotes.len(),
            atm_vol,
            rr25,
            rmse_vol_points,
            params,
            quotes,
        })
    }

    fn rmse_vol_points(quotes: &[OptionQuote], params: SviParams, t: f64) -> f64 {
        if quotes.is_empty() || t <= 0.0 {
            return 0.0;
        }

        let mse = quotes
            .iter()
            .map(|q| {
                let fitted_vol = (params.total_variance(q.log_moneyness).max(1e-12) / t).sqrt();
                let err = (fitted_vol - q.mark_iv) * 100.0;
                err * err
            })
            .sum::<f64>()
            / quotes.len() as f64;
        mse.sqrt()
    }

    fn risk_reversal_25d(quotes: &[OptionQuote]) -> Option<f64> {
        let mut call_choice: Option<(f64, f64)> = None;
        let mut put_choice: Option<(f64, f64)> = None;

        for q in quotes {
            let delta = bs_delta(
                q.underlying_price,
                q.strike,
                q.mark_iv,
                q.time_to_expiry,
                q.option_type,
            );
            match q.option_type {
                OptionType::Call => {
                    let dist = (delta - 0.25).abs();
                    if call_choice.is_none_or(|(best_dist, _)| dist < best_dist) {
                        call_choice = Some((dist, q.mark_iv));
                    }
                }
                OptionType::Put => {
                    let abs_put_delta = (-delta).abs();
                    let dist = (abs_put_delta - 0.25).abs();
                    if put_choice.is_none_or(|(best_dist, _)| dist < best_dist) {
                        put_choice = Some((dist, q.mark_iv));
                    }
                }
            }
        }

        match (call_choice, put_choice) {
            (Some((_, call_vol)), Some((_, put_vol))) => Some((call_vol - put_vol) * 100.0),
            _ => None,
        }
    }

    fn bs_delta(forward: f64, strike: f64, vol: f64, t: f64, option_type: OptionType) -> f64 {
        if forward <= 0.0 || strike <= 0.0 || vol <= 0.0 || t <= 0.0 {
            return 0.0;
        }
        let sqrt_t = t.sqrt();
        let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
        let call_delta = norm_cdf(d1);
        match option_type {
            OptionType::Call => call_delta,
            OptionType::Put => call_delta - 1.0,
        }
    }

    fn norm_cdf(x: f64) -> f64 {
        let z = x.abs();
        let t = 1.0 / (1.0 + 0.231_641_9 * z);
        let poly =
            (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t - 0.356_563_782) * t)
                + 0.319_381_530;
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let cdf = 1.0 - pdf * poly * t;
        if x >= 0.0 { cdf } else { 1.0 - cdf }
    }

    fn build_vol_matrix(grouped: &BTreeMap<NaiveDate, Vec<OptionQuote>>) -> VolMatrix {
        let mut global_buckets = BTreeSet::new();
        let mut strike_avg: HashMap<i64, (f64, usize)> = HashMap::new();
        let mut rows: Vec<(NaiveDate, f64, HashMap<i64, (f64, f64, usize)>)> = Vec::new();

        for (expiry, quotes) in grouped {
            if quotes.is_empty() {
                continue;
            }

            let t = quotes[0].time_to_expiry;
            let mut per_strike: HashMap<i64, (f64, f64, usize)> = HashMap::new();
            for q in quotes {
                let bucket = strike_bucket(q.strike);
                global_buckets.insert(bucket);

                let agg = per_strike.entry(bucket).or_insert((0.0, 0.0, 0));
                agg.0 += q.mark_iv;
                agg.1 += q.log_moneyness;
                agg.2 += 1;

                let global = strike_avg.entry(bucket).or_insert((0.0, 0));
                global.0 += q.strike;
                global.1 += 1;
            }
            rows.push((*expiry, t, per_strike));
        }

        let ordered_buckets: Vec<i64> = global_buckets.into_iter().collect();
        let strikes: Vec<f64> = ordered_buckets
            .iter()
            .map(|bucket| {
                strike_avg
                    .get(bucket)
                    .map(|(sum, n)| sum / *n as f64)
                    .unwrap_or(*bucket as f64 / 100.0)
            })
            .collect();

        let mut expiries = Vec::with_capacity(rows.len());
        let mut times = Vec::with_capacity(rows.len());
        let mut vols = Vec::with_capacity(rows.len());
        let mut log_moneyness = Vec::with_capacity(rows.len());

        for (expiry, t, row_map) in rows {
            expiries.push(expiry);
            times.push(t);
            let mut vol_row = Vec::with_capacity(ordered_buckets.len());
            let mut k_row = Vec::with_capacity(ordered_buckets.len());
            for bucket in &ordered_buckets {
                if let Some((sum_vol, sum_k, n)) = row_map.get(bucket) {
                    vol_row.push(Some(sum_vol / *n as f64));
                    k_row.push(Some(sum_k / *n as f64));
                } else {
                    vol_row.push(None);
                    k_row.push(None);
                }
            }
            vols.push(vol_row);
            log_moneyness.push(k_row);
        }

        VolMatrix {
            expiries,
            times,
            strikes,
            vols,
            log_moneyness,
        }
    }

    fn strike_bucket(strike: f64) -> i64 {
        (strike * 100.0).round() as i64
    }

    fn print_summary(
        spot: f64,
        total_quotes: usize,
        matrix: &VolMatrix,
        slices: &[CalibratedSlice],
    ) -> AppResult<()> {
        let ts = chrono::DateTime::<chrono::Utc>::from(SystemTime::now())
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string();
        let populated_cells = matrix
            .vols
            .iter()
            .flat_map(|row| row.iter())
            .filter(|v| v.is_some())
            .count();
        let k_min = matrix
            .log_moneyness
            .iter()
            .flat_map(|row| row.iter().flatten())
            .fold(f64::INFINITY, |acc, &k| acc.min(k));
        let k_max = matrix
            .log_moneyness
            .iter()
            .flat_map(|row| row.iter().flatten())
            .fold(f64::NEG_INFINITY, |acc, &k| acc.max(k));
        let t_min = matrix.times.first().copied().unwrap_or(0.0);
        let t_max = matrix.times.last().copied().unwrap_or(0.0);

        println!("Deribit BTC Vol Surface");
        println!(
            "timestamp={} spot={spot:.2} total_options={} expiries={} matrix={}x{}",
            ts,
            total_quotes,
            matrix.expiries.len(),
            matrix.expiries.len(),
            matrix.strikes.len()
        );
        println!(
            "matrix_points={} t_range=[{:.4},{:.4}]y k_range=[{:.4},{:.4}]",
            populated_cells, t_min, t_max, k_min, k_max
        );
        println!();
        println!(
            "{:<12} {:>7} {:>7} {:>8} {:>9} {:>11} {:>9} {:>8} {:>9} {:>9} {:>9}",
            "expiry",
            "days",
            "count",
            "atm(%)",
            "rr25(%)",
            "rmse(vp)",
            "a",
            "b",
            "rho",
            "m",
            "sigma"
        );

        for s in slices {
            let rr25 = s
                .rr25
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "{:<12} {:>7.2} {:>7} {:>8.2} {:>9} {:>11.3} {:>9.6} {:>8.6} {:>9.4} {:>9.4} {:>9.4}",
                s.expiry,
                s.time_to_expiry * 365.25,
                s.options_count,
                s.atm_vol * 100.0,
                rr25,
                s.rmse_vol_points,
                s.params.a,
                s.params.b,
                s.params.rho,
                s.params.m,
                s.params.sigma
            );
        }
        println!();

        Ok(())
    }

    fn write_html(matrix: &VolMatrix, slices: &[CalibratedSlice]) -> AppResult<()> {
        if slices.is_empty() {
            return Ok(());
        }

        let mut market_x = Vec::new();
        let mut market_y = Vec::new();
        let mut market_z = Vec::new();
        let mut k_min = f64::INFINITY;
        let mut k_max = f64::NEG_INFINITY;

        for slice in slices {
            for q in &slice.quotes {
                market_x.push(q.log_moneyness);
                market_y.push(slice.time_to_expiry);
                market_z.push(q.mark_iv * 100.0);
                k_min = k_min.min(q.log_moneyness);
                k_max = k_max.max(q.log_moneyness);
            }
        }

        if !k_min.is_finite() || !k_max.is_finite() || (k_max - k_min).abs() < 1e-8 {
            return Ok(());
        }

        let grid_n = 41usize;
        let mut k_grid = Vec::with_capacity(grid_n);
        for i in 0..grid_n {
            let u = i as f64 / (grid_n.saturating_sub(1)) as f64;
            k_grid.push(k_min + u * (k_max - k_min));
        }

        let t_grid: Vec<f64> = slices.iter().map(|s| s.time_to_expiry).collect();
        let mut z_grid = Vec::with_capacity(slices.len());
        for slice in slices {
            let row: Vec<f64> = k_grid
                .iter()
                .map(|k| {
                    (slice.params.total_variance(*k).max(1e-12) / slice.time_to_expiry).sqrt()
                        * 100.0
                })
                .collect();
            z_grid.push(row);
        }

        let mut slice_traces = String::new();
        for slice in slices {
            let mut mkt = slice.quotes.clone();
            mkt.sort_by(|a, b| a.log_moneyness.total_cmp(&b.log_moneyness));
            let mkt_x: Vec<f64> = mkt.iter().map(|q| q.log_moneyness).collect();
            let mkt_y: Vec<f64> = mkt.iter().map(|q| q.mark_iv * 100.0).collect();
            let fit_x = k_grid.clone();
            let fit_y: Vec<f64> = fit_x
                .iter()
                .map(|k| {
                    (slice.params.total_variance(*k).max(1e-12) / slice.time_to_expiry).sqrt()
                        * 100.0
                })
                .collect();

            writeln!(
                &mut slice_traces,
                "sliceTraces.push({{x:{},y:{},mode:'markers',type:'scatter',name:'{} mkt'}});",
                serde_json::to_string(&mkt_x)?,
                serde_json::to_string(&mkt_y)?,
                slice.expiry.format("%Y-%m-%d"),
            )?;
            writeln!(
                &mut slice_traces,
                "sliceTraces.push({{x:{},y:{},mode:'lines',type:'scatter',name:'{} svi'}});",
                serde_json::to_string(&fit_x)?,
                serde_json::to_string(&fit_y)?,
                slice.expiry.format("%Y-%m-%d"),
            )?;
        }

        let html = format!(
            r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deribit BTC Vol Surface</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 10% 10%, #1f2937 0%, var(--bg) 55%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      padding: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 85%, black);
      border: 1px solid #374151;
      border-radius: 12px;
      padding: 12px;
    }}
    .title {{
      font-size: 18px;
      font-weight: 700;
      margin: 0 0 8px 4px;
      color: var(--text);
    }}
    .subtitle {{
      margin: 0 0 8px 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    #surface, #slices {{
      width: 100%;
      min-height: 420px;
    }}
    @media (min-width: 980px) {{
      .grid {{
        grid-template-columns: 1fr 1fr;
      }}
      #surface, #slices {{
        min-height: 620px;
      }}
    }}
  </style>
</head>
<body>
  <div class="grid">
    <section class="panel">
      <h2 class="title">SVI Vol Surface</h2>
      <p class="subtitle">Market points and calibrated per-expiry SVI mesh</p>
      <div id="surface"></div>
    </section>
    <section class="panel">
      <h2 class="title">Smile Slices</h2>
      <p class="subtitle">Each expiry: market smiles vs fitted SVI curves</p>
      <div id="slices"></div>
    </section>
  </div>
  <script>
    const market = {{
      x: {market_x},
      y: {market_y},
      z: {market_z},
      mode: 'markers',
      type: 'scatter3d',
      name: 'market',
      marker: {{size: 3, color: '#22d3ee', opacity: 0.8}}
    }};

    const sviSurface = {{
      x: {k_grid},
      y: {t_grid},
      z: {z_grid},
      type: 'surface',
      name: 'svi fit',
      colorscale: 'Viridis',
      opacity: 0.75
    }};

    Plotly.newPlot('surface', [sviSurface, market], {{
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{color: '#e5e7eb'}},
      scene: {{
        xaxis: {{title: 'log-moneyness k = ln(K/F)'}},
        yaxis: {{title: 'time to expiry (years)'}},
        zaxis: {{title: 'implied vol (%)'}}
      }},
      margin: {{l: 0, r: 0, b: 0, t: 0}}
    }}, {{responsive: true}});

    const sliceTraces = [];
    {slice_traces}

    Plotly.newPlot('slices', sliceTraces, {{
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{color: '#e5e7eb'}},
      xaxis: {{title: 'log-moneyness'}},
      yaxis: {{title: 'implied vol (%)'}},
      margin: {{l: 50, r: 20, b: 50, t: 20}},
      legend: {{orientation: 'h'}}
    }}, {{responsive: true}});
  </script>
</body>
</html>
"#,
            market_x = serde_json::to_string(&market_x)?,
            market_y = serde_json::to_string(&market_y)?,
            market_z = serde_json::to_string(&market_z)?,
            k_grid = serde_json::to_string(&k_grid)?,
            t_grid = serde_json::to_string(&t_grid)?,
            z_grid = serde_json::to_string(&z_grid)?,
            slice_traces = slice_traces
        );

        fs::write("vol_surface.html", html)?;
        println!(
            "wrote vol_surface.html (expiries={}, strikes={}, rows={})",
            matrix.expiries.len(),
            matrix.strikes.len(),
            matrix.vols.len()
        );
        Ok(())
    }

    pub async fn run() -> AppResult<()> {
        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .user_agent("openferric-deribit-vol-surface/0.1")
            .build()?;

        let (entries, spot_price) =
            tokio::try_join!(fetch_book_summaries(&client), fetch_spot_price(&client))?;

        let now_secs = now_unix_seconds()?;
        let quotes = build_quotes(&entries, spot_price, now_secs);
        if quotes.is_empty() {
            println!("No liquid BTC options available from Deribit with current filters.");
            return Ok(());
        }

        let grouped = group_by_expiry(quotes);
        let matrix = build_vol_matrix(&grouped);

        let mut slices = Vec::new();
        let mut total_quotes = 0usize;
        for (expiry, quotes_for_expiry) in grouped {
            total_quotes += quotes_for_expiry.len();
            if let Some(slice) = calibrate_slice(expiry, quotes_for_expiry) {
                slices.push(slice);
            }
        }
        slices.sort_by(|a, b| a.time_to_expiry.total_cmp(&b.time_to_expiry));

        if slices.is_empty() {
            println!("No expiry slices had enough data for SVI calibration.");
            return Ok(());
        }

        print_summary(spot_price, total_quotes, &matrix, &slices)?;
        write_html(&matrix, &slices)?;
        Ok(())
    }
}

#[cfg(feature = "deribit")]
#[tokio::main]
async fn main() {
    if let Err(err) = app::run().await {
        eprintln!("deribit_vol_surface failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "deribit"))]
fn main() {
    eprintln!(
        "Enable the `deribit` feature: cargo run --features deribit --bin deribit_vol_surface"
    );
}
