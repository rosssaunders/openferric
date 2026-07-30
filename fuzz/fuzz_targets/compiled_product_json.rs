#![no_main]

use libfuzzer_sys::fuzz_target;
use openferric::dsl::{CompiledProduct, ProductEvaluator};

fuzz_target!(|data: &[u8]| {
    // Keep this target focused on structural correctness rather than expected
    // resource exhaustion from enormous products.
    if data.len() > 64 * 1024 {
        return;
    }
    let Ok(product) = serde_json::from_slice::<CompiledProduct>(data) else {
        return;
    };
    if product.num_underlyings > 8
        || product.state_vars.len() > 64
        || product.schedules.len() > 16
        || product
            .schedules
            .iter()
            .map(|schedule| schedule.dates.len())
            .sum::<usize>()
            > 64
    {
        return;
    }
    if product.validate().is_ok() {
        let num_steps = 8;
        if let Ok(mut evaluator) = ProductEvaluator::new(&product, num_steps, 0.0) {
            let initial_spots = vec![100.0; product.num_underlyings];
            let path_spots = vec![initial_spots.clone(); num_steps + 1];
            let _ = evaluator.evaluate(&path_spots, &initial_spots);
        }
    }
});
