//! XVA calculation demo: CVA, FVA, MVA, KVA.

use openferric::credit::SurvivalCurve;
use openferric::rates::YieldCurve;
use openferric::risk::xva::XvaCalculator;
use openferric::risk::fva::{CsaTerms, fva_from_profile, funding_exposure_profile};
use openferric::risk::mva::{SimmMargin, SimmRiskClass, mva_from_profile};
use openferric::risk::kva::{SaCcrAssetClass, sa_ccr_ead, regulatory_capital, kva_from_profile};

fn main() {
    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let notional = 10_000_000.0;

    // Build curves
    let discount_curve = YieldCurve::new(
        times.iter().map(|&t| (t, (-0.03_f64 * t).exp())).collect(),
    );
    let hazard_rate = 0.02;
    let hazards = vec![hazard_rate; times.len()];
    let cpty_survival = SurvivalCurve::from_piecewise_hazard(&times, &hazards);
    let own_survival = SurvivalCurve::from_piecewise_hazard(&times, &vec![0.01; times.len()]);

    // CVA
    let calc = XvaCalculator::new(
        discount_curve.clone(), cpty_survival, own_survival, 0.6, 0.6,
    );
    let ee_profile = vec![0.03 * notional; times.len()]; // 3% EE
    let cva = calc.cva_from_expected_exposure(&times, &ee_profile);
    println!("CVA = {cva:.0} (notional = {notional:.0})");

    // FVA
    let paths = vec![vec![0.03 * notional; times.len()]; 100]; // Simplified
    let csa = CsaTerms { collateralized: false, ..Default::default() };
    let funding_exp = funding_exposure_profile(&paths, &csa);
    let funding_spread = vec![0.005; times.len()];
    let fva = fva_from_profile(&times, &funding_exp, &funding_spread, &discount_curve);
    println!("FVA = {fva:.0}");

    // MVA (SIMM)
    let simm = SimmMargin {
        risk_class: SimmRiskClass::InterestRate,
        sensitivities: vec![notional * 0.01], // DV01
        risk_weights: vec![0.017],
        intra_corr: 1.0,
    };
    let im = simm.compute();
    let expected_im = vec![im; times.len()];
    let mva = mva_from_profile(&times, &expected_im, &funding_spread, &discount_curve);
    println!("MVA = {mva:.0} (IM = {im:.0})");

    // KVA (SA-CCR)
    let ead = sa_ccr_ead(0.03 * notional, notional, 5.0, SaCcrAssetClass::InterestRate);
    let reg_capital = regulatory_capital(ead, 0.50);
    let expected_capital = vec![reg_capital; times.len()];
    let kva = kva_from_profile(&times, &expected_capital, 0.10, &discount_curve);
    println!("KVA = {kva:.0} (EAD = {ead:.0}, RegCap = {reg_capital:.0})");

    println!("\nTotal XVA = {:.0}", cva + fva + mva + kva);
}
