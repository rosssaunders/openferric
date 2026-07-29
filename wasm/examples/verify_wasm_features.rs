use std::{env, fs, process};

use wasmparser::{Validator, WasmFeatures};

fn validate(bytes: &[u8], features: WasmFeatures) -> Result<(), String> {
    Validator::new_with_features(features)
        .validate_all(bytes)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn portable_features() -> WasmFeatures {
    let mut features = WasmFeatures::default();
    features.remove(
        WasmFeatures::SIMD
            | WasmFeatures::RELAXED_SIMD
            | WasmFeatures::THREADS
            | WasmFeatures::SHARED_EVERYTHING_THREADS,
    );
    features
}

fn verify(bytes: &[u8], profile: &str) -> Result<(), String> {
    let portable = portable_features();

    match profile {
        "baseline" => validate(bytes, portable).map_err(|error| {
            format!("portable package unexpectedly requires SIMD or threads: {error}")
        }),
        "simd" => {
            let mut features = portable;
            features.insert(WasmFeatures::SIMD);
            validate(bytes, features).map_err(|error| {
                format!("SIMD package is invalid with SIMD128 enabled: {error}")
            })?;
            if validate(bytes, portable).is_ok() {
                return Err("SIMD package contains no SIMD128 operations".to_owned());
            }
            Ok(())
        }
        "threads" => {
            let mut features = portable;
            features.insert(WasmFeatures::THREADS);
            validate(bytes, features).map_err(|error| {
                format!("threaded package is invalid with WebAssembly threads enabled: {error}")
            })?;
            if validate(bytes, portable).is_ok() {
                return Err(
                    "threaded package contains no atomic/shared-memory operations".to_owned(),
                );
            }
            Ok(())
        }
        _ => Err(format!(
            "unknown profile '{profile}'; expected baseline, simd, or threads"
        )),
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: verify_wasm_features <module.wasm> <baseline|simd|threads>");
        process::exit(2);
    };
    let Some(profile) = args.next() else {
        eprintln!("usage: verify_wasm_features <module.wasm> <baseline|simd|threads>");
        process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: verify_wasm_features <module.wasm> <baseline|simd|threads>");
        process::exit(2);
    }

    let bytes = fs::read(&path).unwrap_or_else(|error| {
        eprintln!("failed to read '{path}': {error}");
        process::exit(1);
    });
    if let Err(error) = verify(&bytes, &profile) {
        eprintln!("{path}: {error}");
        process::exit(1);
    }

    println!("Verified {profile} WebAssembly features in {path}");
}
