#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(source) = std::str::from_utf8(data) else {
        return;
    };

    // The diagnostic pipeline intentionally accepts malformed and incomplete
    // editor input. A successful parse is then passed through the complete
    // compiler pipeline.
    let _ = openferric::dsl::analysis::parse_and_diagnose(source);
    let _ = openferric::dsl::parse_and_compile(source);
});
