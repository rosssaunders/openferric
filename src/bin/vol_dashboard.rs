//! Command-line entry point for Vol Dashboard workflows.
//!
//! This binary wires OpenFerric models and engines into an executable utility.

#[cfg(feature = "deribit")]
mod app {
    use axum::Router;
    use axum::extract::State;
    use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
    use axum::response::{Html, IntoResponse};
    use axum::routing::get;
    use chrono::NaiveDate;
    use futures_util::{SinkExt, StreamExt};
    use openferric::vol::surface::{SviParams, calibrate_svi};
    use reqwest::Client;
    use serde::Serialize;
    use serde_json::{Value, json};
    use std::collections::{BTreeMap, HashMap};
    use std::error::Error;
    use std::sync::Arc;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::net::TcpListener;
    use tokio::sync::{RwLock, watch};
    use tokio::time::sleep;
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message as DeribitMessage;
    use tower_http::cors::CorsLayer;

    const DERIBIT_WS_URL: &str = "wss://www.deribit.com/ws/api/v2";
    const BOOK_SUMMARY_URL: &str =
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency";
    const INDEX_PRICE_URL: &str = "https://www.deribit.com/api/v2/public/get_index_price";
    const MARKPRICE_CHANNEL: &str = "markprice.options.btc_usd";

    const MEANINGFUL_IV_CHANGE: f64 = 0.001;
    const CALIBRATION_DEBOUNCE_MS: u64 = 200;

    type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

    #[derive(Debug, Clone)]
    struct OptionQuote {
        strike: f64,
        expiry: NaiveDate,
        is_call: bool,
        mark_price: f64,
        mark_iv: f64,
        underlying_price: f64,
        bid_iv: f64,
        ask_iv: f64,
        open_interest: f64,
        timestamp: u64,
    }

    #[derive(Debug, Clone)]
    struct QuoteUpdate {
        instrument_name: String,
        strike: f64,
        expiry: NaiveDate,
        is_call: bool,
        mark_price: Option<f64>,
        mark_iv: Option<f64>,
        underlying_price: Option<f64>,
        bid_iv: Option<f64>,
        ask_iv: Option<f64>,
        open_interest: Option<f64>,
        timestamp: Option<u64>,
    }

    #[derive(Debug)]
    struct MarketState {
        chain: HashMap<String, OptionQuote>,
        spot: f64,
        dirty: bool,
        last_calibration_at: Instant,
        tick_window_start: Instant,
        tick_counter: u64,
        ticks_per_sec: f64,
    }

    impl MarketState {
        fn new() -> Self {
            let now = Instant::now();
            Self {
                chain: HashMap::new(),
                spot: 0.0,
                dirty: false,
                last_calibration_at: now - Duration::from_millis(CALIBRATION_DEBOUNCE_MS),
                tick_window_start: now,
                tick_counter: 0,
                ticks_per_sec: 0.0,
            }
        }

        fn note_tick(&mut self) {
            self.tick_counter += 1;
        }

        fn refresh_ticks_per_sec(&mut self) {
            let elapsed = self.tick_window_start.elapsed().as_secs_f64();
            if elapsed >= 1.0 {
                self.ticks_per_sec = self.tick_counter as f64 / elapsed;
                self.tick_counter = 0;
                self.tick_window_start = Instant::now();
            }
        }
    }

    #[derive(Clone)]
    struct HttpState {
        tx: watch::Sender<String>,
    }

    #[derive(Debug, Clone)]
    struct SviSlice {
        expiry: NaiveDate,
        t: f64,
        params: SviParams,
        rmse: f64,
        n_options: usize,
        atm_vol: f64,
        points: Vec<SlicePoint>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SlicePoint {
        strike: f64,
        market_iv: f64,
        fitted_iv: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SviJson {
        a: f64,
        b: f64,
        rho: f64,
        m: f64,
        sigma: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SurfaceSliceJson {
        expiry: String,
        #[serde(rename = "T")]
        t: f64,
        svi: SviJson,
        rmse: f64,
        n_options: usize,
        atm_vol: f64,
        points: Vec<SlicePoint>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct MetricsJson {
        calibration_time_us: u64,
        n_options: usize,
        n_expiries: usize,
        ticks_per_sec: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SurfaceUpdateJson {
        timestamp: u64,
        spot: f64,
        surfaces: Vec<SurfaceSliceJson>,
        metrics: MetricsJson,
    }

    fn parse_port() -> u16 {
        let mut port = 3000_u16;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--port" {
                if let Some(value) = args.next() {
                    match value.parse::<u16>() {
                        Ok(parsed) => port = parsed,
                        Err(_) => eprintln!("ignoring invalid --port value: {value}"),
                    }
                }
            } else if let Some(value) = arg.strip_prefix("--port=") {
                match value.parse::<u16>() {
                    Ok(parsed) => port = parsed,
                    Err(_) => eprintln!("ignoring invalid --port value: {value}"),
                }
            }
        }
        port
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

    fn parse_instrument_name(name: &str) -> Option<(NaiveDate, f64, bool)> {
        let parts: Vec<&str> = name.split('-').collect();
        if parts.len() != 4 {
            return None;
        }
        let expiry = parse_deribit_expiry(parts[1])?;
        let strike = parts[2].parse::<f64>().ok()?;
        let is_call = match parts[3] {
            "C" => true,
            "P" => false,
            _ => return None,
        };
        Some((expiry, strike, is_call))
    }

    fn field_f64(v: &Value, key: &str) -> Option<f64> {
        v.get(key).and_then(Value::as_f64)
    }

    fn field_u64(v: &Value, key: &str) -> Option<u64> {
        v.get(key).and_then(Value::as_u64)
    }

    fn first_f64(v: &Value, keys: &[&str]) -> Option<f64> {
        keys.iter().find_map(|key| field_f64(v, key))
    }

    fn first_u64(v: &Value, keys: &[&str]) -> Option<u64> {
        keys.iter().find_map(|key| field_u64(v, key))
    }

    fn normalize_iv(raw: f64) -> Option<f64> {
        if !raw.is_finite() || raw <= 0.0 {
            return None;
        }
        if raw > 3.0 {
            Some(raw / 100.0)
        } else {
            Some(raw)
        }
    }

    fn unix_time_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    fn unix_time_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    fn expiry_time_secs(expiry: NaiveDate) -> Option<f64> {
        expiry
            .and_hms_opt(8, 0, 0)
            .map(|dt| dt.and_utc().timestamp() as f64)
    }

    fn time_to_expiry(expiry: NaiveDate, now_secs: f64) -> Option<f64> {
        let expiry_secs = expiry_time_secs(expiry)?;
        let t = (expiry_secs - now_secs) / (365.25 * 24.0 * 3600.0);
        if t > 0.0 { Some(t) } else { None }
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

    fn quote_from_book_entry(entry: &Value, fallback_spot: f64) -> Option<(String, OptionQuote)> {
        let instrument_name = entry.get("instrument_name")?.as_str()?.to_string();
        let (expiry, strike, is_call) = parse_instrument_name(&instrument_name)?;
        let mark_iv = normalize_iv(first_f64(entry, &["mark_iv", "iv"])?)?;

        let underlying_price = first_f64(entry, &["underlying_price", "index_price"])
            .filter(|v| *v > 0.0)
            .unwrap_or(fallback_spot.max(1e-6));

        let mark_price = first_f64(entry, &["mark_price"]).unwrap_or(0.0).max(0.0);
        let bid_iv = first_f64(entry, &["bid_iv"])
            .and_then(normalize_iv)
            .unwrap_or(0.0);
        let ask_iv = first_f64(entry, &["ask_iv"])
            .and_then(normalize_iv)
            .unwrap_or(0.0);
        let open_interest = first_f64(entry, &["open_interest"]).unwrap_or(0.0).max(0.0);
        let timestamp =
            first_u64(entry, &["creation_timestamp", "timestamp"]).unwrap_or_else(unix_time_millis);

        Some((
            instrument_name,
            OptionQuote {
                strike,
                expiry,
                is_call,
                mark_price,
                mark_iv,
                underlying_price,
                bid_iv,
                ask_iv,
                open_interest,
                timestamp,
            },
        ))
    }

    async fn seed_state(client: &Client, market: &Arc<RwLock<MarketState>>) -> AppResult<()> {
        let (entries, spot) =
            tokio::try_join!(fetch_book_summaries(client), fetch_spot_price(client))?;
        let mut seeded = HashMap::new();
        for entry in &entries {
            if let Some((instrument, quote)) = quote_from_book_entry(entry, spot) {
                seeded.insert(instrument, quote);
            }
        }

        let mut state = market.write().await;
        if !seeded.is_empty() {
            state.chain = seeded;
            state.spot = spot.max(0.0);
            state.dirty = true;
        }
        Ok(())
    }

    fn parse_quote_update(data: &Value) -> Option<QuoteUpdate> {
        let instrument_name = data
            .get("instrument_name")
            .and_then(Value::as_str)
            .or_else(|| data.get("instrument").and_then(Value::as_str))?
            .to_string();

        let (expiry, strike, is_call) = parse_instrument_name(&instrument_name)?;

        Some(QuoteUpdate {
            instrument_name,
            strike,
            expiry,
            is_call,
            mark_price: first_f64(data, &["mark_price"]).map(|v| v.max(0.0)),
            mark_iv: first_f64(data, &["mark_iv", "iv"]).and_then(normalize_iv),
            underlying_price: first_f64(data, &["underlying_price", "index_price"])
                .filter(|v| *v > 0.0),
            bid_iv: first_f64(data, &["bid_iv"]).and_then(normalize_iv),
            ask_iv: first_f64(data, &["ask_iv"]).and_then(normalize_iv),
            open_interest: first_f64(data, &["open_interest"]).map(|v| v.max(0.0)),
            timestamp: first_u64(data, &["timestamp", "creation_timestamp"]),
        })
    }

    fn collect_quote_nodes<'a>(value: &'a Value, out: &mut Vec<&'a Value>) {
        match value {
            Value::Object(map) => {
                if map.contains_key("instrument_name") {
                    out.push(value);
                    return;
                }
                for child in map.values() {
                    if child.is_array() || child.is_object() {
                        collect_quote_nodes(child, out);
                    }
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    collect_quote_nodes(item, out);
                }
            }
            _ => {}
        }
    }

    fn apply_update(state: &mut MarketState, update: QuoteUpdate) -> bool {
        if !state.chain.contains_key(&update.instrument_name) && update.mark_iv.is_none() {
            return false;
        }

        let instrument = update.instrument_name.clone();
        let ts = update.timestamp.unwrap_or_else(unix_time_millis);
        let fallback_spot = if state.spot > 0.0 { state.spot } else { 1.0 };
        let incoming_iv = update.mark_iv.unwrap_or(0.0);

        let entry = state
            .chain
            .entry(instrument)
            .or_insert_with(|| OptionQuote {
                strike: update.strike,
                expiry: update.expiry,
                is_call: update.is_call,
                mark_price: update.mark_price.unwrap_or(0.0),
                mark_iv: incoming_iv,
                underlying_price: update.underlying_price.unwrap_or(fallback_spot),
                bid_iv: update.bid_iv.unwrap_or(0.0),
                ask_iv: update.ask_iv.unwrap_or(0.0),
                open_interest: update.open_interest.unwrap_or(0.0),
                timestamp: ts,
            });

        let meaningful = {
            let previous_iv = entry.mark_iv;
            entry.strike = update.strike;
            entry.expiry = update.expiry;
            entry.is_call = update.is_call;
            if let Some(v) = update.mark_price {
                entry.mark_price = v;
            }
            if let Some(v) = update.mark_iv {
                entry.mark_iv = v;
            }
            if let Some(v) = update.underlying_price {
                entry.underlying_price = v;
                state.spot = v;
            }
            if let Some(v) = update.bid_iv {
                entry.bid_iv = v;
            }
            if let Some(v) = update.ask_iv {
                entry.ask_iv = v;
            }
            if let Some(v) = update.open_interest {
                entry.open_interest = v;
            }
            entry.timestamp = ts;

            if update.mark_iv.is_some() {
                (entry.mark_iv - previous_iv).abs() > MEANINGFUL_IV_CHANGE
            } else {
                false
            }
        };
        if meaningful {
            state.dirty = true;
        }

        state.note_tick();
        meaningful
    }

    async fn handle_deribit_text(text: &str, market: &Arc<RwLock<MarketState>>) -> AppResult<()> {
        let payload: Value = serde_json::from_str(text)?;
        let Some(params) = payload.get("params") else {
            return Ok(());
        };
        if params.get("channel").and_then(Value::as_str) != Some(MARKPRICE_CHANNEL) {
            return Ok(());
        }
        let Some(data) = params.get("data") else {
            return Ok(());
        };

        let mut nodes = Vec::new();
        collect_quote_nodes(data, &mut nodes);
        if nodes.is_empty() {
            return Ok(());
        }

        let updates: Vec<QuoteUpdate> = nodes.into_iter().filter_map(parse_quote_update).collect();
        if updates.is_empty() {
            return Ok(());
        }

        let mut state = market.write().await;
        for update in updates {
            apply_update(&mut state, update);
        }
        Ok(())
    }

    async fn deribit_feed_loop(client: Client, market: Arc<RwLock<MarketState>>) {
        loop {
            if let Err(err) = seed_state(&client, &market).await {
                eprintln!("seed state failed: {err}");
            }

            match connect_async(DERIBIT_WS_URL).await {
                Ok((mut stream, _)) => {
                    let subscribe = json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "public/subscribe",
                        "params": { "channels": [MARKPRICE_CHANNEL] }
                    })
                    .to_string();

                    if stream
                        .send(DeribitMessage::Text(subscribe.into()))
                        .await
                        .is_err()
                    {
                        sleep(Duration::from_secs(1)).await;
                        continue;
                    }

                    while let Some(msg) = stream.next().await {
                        match msg {
                            Ok(DeribitMessage::Text(text)) => {
                                if let Err(err) = handle_deribit_text(&text, &market).await {
                                    eprintln!("ws message parse failed: {err}");
                                }
                            }
                            Ok(DeribitMessage::Binary(bytes)) => {
                                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                    if let Err(err) = handle_deribit_text(&text, &market).await {
                                        eprintln!("ws message parse failed: {err}");
                                    }
                                }
                            }
                            Ok(DeribitMessage::Ping(payload)) => {
                                let _ = stream.send(DeribitMessage::Pong(payload)).await;
                            }
                            Ok(DeribitMessage::Close(_)) => break,
                            Ok(_) => {}
                            Err(err) => {
                                eprintln!("deribit ws disconnected: {err}");
                                break;
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!("deribit ws connect failed: {err}");
                }
            }

            sleep(Duration::from_secs(1)).await;
        }
    }

    fn calibrate_surface(quotes: Vec<OptionQuote>) -> Vec<SviSlice> {
        let now_secs = unix_time_secs();
        let mut grouped: BTreeMap<NaiveDate, Vec<OptionQuote>> = BTreeMap::new();
        for quote in quotes {
            if quote.mark_iv > 0.0 && quote.strike > 0.0 {
                grouped.entry(quote.expiry).or_default().push(quote);
            }
        }

        let mut slices = Vec::new();
        for (expiry, mut expiry_quotes) in grouped {
            let Some(t) = time_to_expiry(expiry, now_secs) else {
                continue;
            };

            expiry_quotes.retain(|q| q.mark_iv > 0.0 && q.strike > 0.0);
            if expiry_quotes.len() < 5 {
                continue;
            }

            let forward = expiry_quotes
                .iter()
                .map(|q| q.underlying_price)
                .filter(|v| *v > 0.0)
                .sum::<f64>()
                / expiry_quotes.len() as f64;
            let forward = if forward.is_finite() && forward > 0.0 {
                forward
            } else {
                continue;
            };

            let mut points = Vec::with_capacity(expiry_quotes.len());
            let mut k_with_quotes = Vec::with_capacity(expiry_quotes.len());
            for q in &expiry_quotes {
                let k = (q.strike / forward).ln();
                points.push((k, q.mark_iv * q.mark_iv * t));
                k_with_quotes.push((k, q));
            }
            if points.len() < 5 {
                continue;
            }

            let atm_vol = k_with_quotes
                .iter()
                .min_by(|(ka, _), (kb, _)| ka.abs().total_cmp(&kb.abs()))
                .map(|(_, q)| q.mark_iv)
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

            let mut err_sq = 0.0;
            let mut smile_points = Vec::with_capacity(k_with_quotes.len());
            for (k, q) in k_with_quotes {
                let fitted_iv = (params.total_variance(k).max(1e-12) / t).sqrt();
                let err = (fitted_iv - q.mark_iv) * 100.0;
                err_sq += err * err;
                smile_points.push(SlicePoint {
                    strike: q.strike,
                    market_iv: q.mark_iv * 100.0,
                    fitted_iv: fitted_iv * 100.0,
                });
            }
            smile_points.sort_by(|a, b| a.strike.total_cmp(&b.strike));

            let rmse = (err_sq / points.len() as f64).sqrt();

            slices.push(SviSlice {
                expiry,
                t,
                params,
                rmse,
                n_options: points.len(),
                atm_vol: atm_vol * 100.0,
                points: smile_points,
            });
        }

        slices.sort_by(|a, b| a.t.total_cmp(&b.t));
        slices
    }

    fn to_payload(
        spot: f64,
        slices: Vec<SviSlice>,
        n_options: usize,
        ticks_per_sec: f64,
        calibration_time_us: u64,
    ) -> SurfaceUpdateJson {
        let surfaces = slices
            .iter()
            .map(|slice| SurfaceSliceJson {
                expiry: slice.expiry.format("%Y-%m-%d").to_string(),
                t: slice.t,
                svi: SviJson {
                    a: slice.params.a,
                    b: slice.params.b,
                    rho: slice.params.rho,
                    m: slice.params.m,
                    sigma: slice.params.sigma,
                },
                rmse: slice.rmse,
                n_options: slice.n_options,
                atm_vol: slice.atm_vol,
                points: slice.points.clone(),
            })
            .collect::<Vec<_>>();

        SurfaceUpdateJson {
            timestamp: unix_time_millis(),
            spot,
            metrics: MetricsJson {
                calibration_time_us,
                n_options,
                n_expiries: surfaces.len(),
                ticks_per_sec,
            },
            surfaces,
        }
    }

    async fn calibration_loop(market: Arc<RwLock<MarketState>>, tx: watch::Sender<String>) {
        loop {
            sleep(Duration::from_millis(40)).await;

            let (ready, snapshot, spot, ticks_per_sec) = {
                let mut state = market.write().await;
                state.refresh_ticks_per_sec();

                let ready = state.dirty
                    && state.last_calibration_at.elapsed()
                        >= Duration::from_millis(CALIBRATION_DEBOUNCE_MS);
                if !ready {
                    (false, Vec::new(), state.spot, state.ticks_per_sec)
                } else {
                    state.dirty = false;
                    state.last_calibration_at = Instant::now();
                    let quotes = state.chain.values().cloned().collect::<Vec<_>>();
                    (true, quotes, state.spot, state.ticks_per_sec)
                }
            };

            if !ready {
                continue;
            }

            let n_options = snapshot.len();
            let started = Instant::now();
            let slices =
                match tokio::task::spawn_blocking(move || calibrate_surface(snapshot)).await {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("calibration worker failed: {err}");
                        continue;
                    }
                };
            let calibration_time_us = started.elapsed().as_micros() as u64;

            let payload = to_payload(spot, slices, n_options, ticks_per_sec, calibration_time_us);
            if let Ok(text) = serde_json::to_string(&payload) {
                let _ = tx.send(text);
            }
        }
    }

    async fn index_handler() -> Html<&'static str> {
        Html(DASHBOARD_HTML)
    }

    async fn ws_handler(ws: WebSocketUpgrade, State(state): State<HttpState>) -> impl IntoResponse {
        ws.on_upgrade(move |socket| client_ws(socket, state.tx.subscribe()))
    }

    async fn client_ws(socket: WebSocket, mut rx: watch::Receiver<String>) {
        let (mut sender, mut receiver) = socket.split();

        let initial = rx.borrow().clone();
        if !initial.is_empty() && sender.send(Message::Text(initial.into())).await.is_err() {
            return;
        }

        loop {
            tokio::select! {
                changed = rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let msg = rx.borrow().clone();
                    if !msg.is_empty() && sender.send(Message::Text(msg.into())).await.is_err() {
                        break;
                    }
                }
                incoming = receiver.next() => {
                    match incoming {
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Ok(_)) => {}
                        Some(Err(_)) => break,
                    }
                }
            }
        }
    }

    pub async fn run() -> AppResult<()> {
        let port = parse_port();
        let client = Client::builder()
            .timeout(Duration::from_secs(3))
            .user_agent("openferric-vol-dashboard/0.1")
            .build()?;

        let market = Arc::new(RwLock::new(MarketState::new()));
        if let Err(err) = seed_state(&client, &market).await {
            eprintln!("initial REST seed failed: {err}");
        }

        let (tx, _rx) = watch::channel(String::new());

        tokio::spawn(deribit_feed_loop(client.clone(), Arc::clone(&market)));
        tokio::spawn(calibration_loop(Arc::clone(&market), tx.clone()));

        let app = Router::new()
            .route("/", get(index_handler))
            .route("/ws", get(ws_handler))
            .layer(CorsLayer::permissive())
            .with_state(HttpState { tx });

        let listener = TcpListener::bind(("0.0.0.0", port)).await?;
        println!("🌋 Vol Surface Dashboard live at http://localhost:{port}");
        axum::serve(listener, app).await?;
        Ok(())
    }

    const DASHBOARD_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deribit Vol Surface Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #070a0f;
      --panel: #11161f;
      --border: #263241;
      --fg: #d8e0ea;
      --muted: #8ea1b8;
      --accent: #2fb2ff;
      --accent2: #19d8a8;
      --danger: #ff7a7a;
      --mono: "JetBrains Mono", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: "IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--fg);
      font-family: var(--sans);
      background:
        radial-gradient(1000px 700px at 12% -20%, #15314f 0%, rgba(21,49,79,0) 60%),
        radial-gradient(900px 650px at 100% 0%, #0f2d25 0%, rgba(15,45,37,0) 58%),
        linear-gradient(180deg, #06080c 0%, #070a0f 100%);
      min-height: 100vh;
      padding: 10px;
    }
    .wrap {
      max-width: 1600px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.7fr 1fr;
      grid-template-rows: auto 1fr;
      gap: 10px;
      height: calc(100vh - 20px);
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 92%, black);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      overflow: hidden;
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 6px;
    }
    .title {
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .mono { font-family: var(--mono); }
    #surface { height: 56vh; min-height: 360px; }
    #term { height: 20vh; min-height: 160px; }
    #smile { height: 28vh; min-height: 210px; }
    .right {
      display: grid;
      grid-template-rows: min-content 1fr;
      gap: 10px;
      min-height: 0;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 10px;
      font-size: 12px;
    }
    .metrics-grid .k { color: var(--muted); }
    .metrics-grid .v { text-align: right; font-family: var(--mono); }
    .rmse-wrap {
      margin-top: 8px;
      max-height: 220px;
      overflow: auto;
      border-top: 1px solid var(--border);
      padding-top: 6px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 3px 0;
      border-bottom: 1px solid rgba(142, 161, 184, 0.14);
      text-align: right;
      font-family: var(--mono);
    }
    th:first-child, td:first-child { text-align: left; font-family: var(--sans); color: var(--muted); }
    .toolbar {
      display: flex;
      gap: 6px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }
    select {
      background: #0d141d;
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 3px 7px;
      font-family: var(--mono);
      font-size: 12px;
    }
    @media (max-width: 1060px) {
      .wrap {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto auto;
        height: auto;
      }
      #surface { height: 45vh; }
      #term { height: 26vh; }
      #smile { height: 34vh; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel" style="grid-row: 1 / span 2;">
      <div class="head">
        <div class="title">3D Surface · BTC Options</div>
        <div class="mono" id="last-ts" style="font-size:12px;color:var(--muted)">waiting for feed...</div>
      </div>
      <div id="surface"></div>
      <div class="head" style="margin-top:4px;">
        <div class="title">Term Structure (ATM Vol)</div>
      </div>
      <div id="term"></div>
    </section>

    <div class="right">
      <section class="panel">
        <div class="head">
          <div class="title">Smile Slice</div>
          <div class="toolbar">
            <label for="expiry-select">expiry</label>
            <select id="expiry-select"></select>
          </div>
        </div>
        <div id="smile"></div>
      </section>

      <section class="panel">
        <div class="head">
          <div class="title">Metrics</div>
        </div>
        <div class="metrics-grid">
          <div class="k">spot</div><div class="v" id="m-spot">-</div>
          <div class="k">calib time (µs)</div><div class="v" id="m-calib">-</div>
          <div class="k">options</div><div class="v" id="m-options">-</div>
          <div class="k">expiries</div><div class="v" id="m-expiries">-</div>
          <div class="k">ticks/sec</div><div class="v" id="m-tps">-</div>
        </div>
        <div class="rmse-wrap">
          <table id="rmse-table">
            <thead>
              <tr><th>expiry</th><th>rmse(vp)</th><th>n</th></tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script>
    const surfaceDiv = document.getElementById('surface');
    const termDiv = document.getElementById('term');
    const smileDiv = document.getElementById('smile');
    const expirySelect = document.getElementById('expiry-select');
    const rmseBody = document.querySelector('#rmse-table tbody');
    const priceFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
    const numFmt2 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
    const numFmt1 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });

    let latest = null;
    let activeExpiry = '';

    function sviIvPct(svi, k, t) {
      if (!svi || !Number.isFinite(t) || t <= 0) return NaN;
      const x = k - svi.m;
      const w = svi.a + svi.b * (svi.rho * x + Math.sqrt(x * x + svi.sigma * svi.sigma));
      return Math.sqrt(Math.max(w, 1e-12) / t) * 100.0;
    }

    function buildSurfaceTraces(payload) {
      const surfaces = payload.surfaces || [];
      const marketX = [];
      const marketY = [];
      const marketZ = [];

      for (const slice of surfaces) {
        for (const p of (slice.points || [])) {
          if (!Number.isFinite(p.strike) || p.strike <= 0 || payload.spot <= 0) continue;
          marketX.push(Math.log(p.strike / payload.spot));
          marketY.push(slice.T);
          marketZ.push(p.market_iv);
        }
      }

      let kMin = -0.45;
      let kMax = 0.45;
      if (marketX.length > 0) {
        kMin = Math.min(...marketX);
        kMax = Math.max(...marketX);
        if (Math.abs(kMax - kMin) < 1e-6) {
          kMin -= 0.25;
          kMax += 0.25;
        }
      }

      const gridN = 37;
      const kGrid = [];
      for (let i = 0; i < gridN; i++) {
        const u = i / (gridN - 1);
        kGrid.push(kMin + (kMax - kMin) * u);
      }
      const tGrid = surfaces.map(s => s.T);
      const zGrid = surfaces.map(s => kGrid.map(k => sviIvPct(s.svi, k, s.T)));

      return [
        {
          type: 'surface',
          name: 'SVI fit',
          x: kGrid,
          y: tGrid,
          z: zGrid,
          opacity: 0.82,
          showscale: false,
          colorscale: [
            [0.0, '#144f83'],
            [0.5, '#1e8ac8'],
            [1.0, '#39dba7']
          ]
        },
        {
          type: 'scatter3d',
          mode: 'markers',
          name: 'Market',
          x: marketX,
          y: marketY,
          z: marketZ,
          marker: { size: 2.5, color: '#d8e0ea', opacity: 0.85 }
        }
      ];
    }

    function updateSurface(payload) {
      const traces = buildSurfaceTraces(payload);
      Plotly.react(surfaceDiv, traces, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d8e0ea', family: 'IBM Plex Sans, sans-serif', size: 11 },
        margin: { l: 0, r: 0, b: 0, t: 0 },
        scene: {
          xaxis: { title: 'ln(K/F)', gridcolor: '#2a3746', zerolinecolor: '#2a3746', color: '#9cb0c6' },
          yaxis: { title: 'T (years)', gridcolor: '#2a3746', zerolinecolor: '#2a3746', color: '#9cb0c6' },
          zaxis: { title: 'IV (%)', gridcolor: '#2a3746', zerolinecolor: '#2a3746', color: '#9cb0c6' },
          camera: { eye: { x: 1.65, y: 1.25, z: 0.8 } }
        },
        showlegend: false,
        transition: { duration: 140, easing: 'cubic-in-out' }
      }, { responsive: true, displayModeBar: false });
    }

    function updateTerm(payload) {
      const surfaces = payload.surfaces || [];
      Plotly.react(termDiv, [{
        type: 'scatter',
        mode: 'lines+markers',
        name: 'ATM',
        x: surfaces.map(s => s.expiry),
        y: surfaces.map(s => s.atm_vol),
        line: { color: '#2fb2ff', width: 2 },
        marker: { color: '#39dba7', size: 6 }
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d8e0ea', family: 'IBM Plex Sans, sans-serif', size: 11 },
        margin: { l: 42, r: 8, b: 32, t: 6 },
        xaxis: { title: '', gridcolor: '#223040', color: '#8ea1b8' },
        yaxis: { title: 'ATM IV (%)', gridcolor: '#223040', color: '#8ea1b8' },
        transition: { duration: 120, easing: 'linear' }
      }, { responsive: true, displayModeBar: false });
    }

    function ensureExpirySelection(payload) {
      const expiries = (payload.surfaces || []).map(s => s.expiry);
      const prev = activeExpiry;
      expirySelect.innerHTML = '';
      for (const e of expiries) {
        const opt = document.createElement('option');
        opt.value = e;
        opt.textContent = e;
        expirySelect.appendChild(opt);
      }
      if (expiries.length === 0) {
        activeExpiry = '';
        return;
      }
      if (prev && expiries.includes(prev)) {
        activeExpiry = prev;
      } else {
        activeExpiry = expiries[0];
      }
      expirySelect.value = activeExpiry;
    }

    function updateSmile(payload) {
      if (!activeExpiry) {
        Plotly.react(smileDiv, [], { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' });
        return;
      }
      const slice = (payload.surfaces || []).find(s => s.expiry === activeExpiry);
      if (!slice) return;

      const pts = [...(slice.points || [])].sort((a, b) => a.strike - b.strike);
      const x = pts.map(p => Math.log(p.strike / payload.spot));
      const market = pts.map(p => p.market_iv);
      const fitted = pts.map(p => p.fitted_iv);

      Plotly.react(smileDiv, [
        {
          type: 'scatter',
          mode: 'markers',
          name: 'market',
          x,
          y: market,
          marker: { color: '#d8e0ea', size: 5 }
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'fitted',
          x,
          y: fitted,
          line: { color: '#19d8a8', width: 2 }
        }
      ], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#d8e0ea', family: 'IBM Plex Sans, sans-serif', size: 11 },
        margin: { l: 42, r: 10, b: 34, t: 6 },
        xaxis: { title: 'ln(K/F)', gridcolor: '#223040', color: '#8ea1b8' },
        yaxis: { title: 'IV (%)', gridcolor: '#223040', color: '#8ea1b8' },
        legend: { orientation: 'h', y: 1.02, x: 0.01 }
      }, { responsive: true, displayModeBar: false });
    }

    function updateMetrics(payload) {
      const m = payload.metrics || {};
      document.getElementById('m-spot').textContent = payload.spot > 0 ? priceFmt.format(payload.spot) : '-';
      document.getElementById('m-calib').textContent = Number.isFinite(m.calibration_time_us) ? Math.round(m.calibration_time_us).toString() : '-';
      document.getElementById('m-options').textContent = Number.isFinite(m.n_options) ? m.n_options.toString() : '-';
      document.getElementById('m-expiries').textContent = Number.isFinite(m.n_expiries) ? m.n_expiries.toString() : '-';
      document.getElementById('m-tps').textContent = Number.isFinite(m.ticks_per_sec) ? numFmt1.format(m.ticks_per_sec) : '-';

      const tsDate = new Date(payload.timestamp || 0);
      document.getElementById('last-ts').textContent = Number.isFinite(tsDate.getTime())
        ? tsDate.toISOString().replace('T', ' ').replace('Z', ' UTC')
        : 'n/a';

      rmseBody.innerHTML = '';
      for (const s of (payload.surfaces || [])) {
        const tr = document.createElement('tr');
        const c1 = document.createElement('td');
        const c2 = document.createElement('td');
        const c3 = document.createElement('td');
        c1.textContent = s.expiry;
        c2.textContent = Number.isFinite(s.rmse) ? numFmt2.format(s.rmse) : '-';
        c3.textContent = Number.isFinite(s.n_options) ? String(s.n_options) : '-';
        tr.appendChild(c1);
        tr.appendChild(c2);
        tr.appendChild(c3);
        rmseBody.appendChild(tr);
      }
    }

    function render(payload) {
      latest = payload;
      updateSurface(payload);
      updateTerm(payload);
      ensureExpirySelection(payload);
      updateSmile(payload);
      updateMetrics(payload);
    }

    expirySelect.addEventListener('change', () => {
      activeExpiry = expirySelect.value;
      if (latest) updateSmile(latest);
    });

    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${location.host}/ws`);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload && Array.isArray(payload.surfaces)) {
          render(payload);
        }
      } catch (_) {}
    };
    socket.onerror = () => {
      document.getElementById('last-ts').textContent = 'ws error';
    };
    socket.onclose = () => {
      document.getElementById('last-ts').textContent = 'ws disconnected';
    };
  </script>
</body>
</html>
"#;
}

#[cfg(feature = "deribit")]
#[tokio::main]
async fn main() {
    if let Err(err) = app::run().await {
        eprintln!("vol_dashboard failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "deribit"))]
fn main() {
    eprintln!("Enable the `deribit` feature: cargo run --features deribit --bin vol_dashboard");
}
