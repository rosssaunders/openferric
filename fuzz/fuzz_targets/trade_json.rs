#![no_main]

use libfuzzer_sys::fuzz_target;
use openferric::instruments::{Portfolio, TradeInstrument};

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<TradeInstrument>(data);
    let _ = serde_json::from_slice::<Portfolio>(data);
});
