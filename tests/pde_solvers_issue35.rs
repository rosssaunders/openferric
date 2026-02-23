use openferric::core::{OptionType, PricingEngine, PricingError};
use openferric::engines::analytic::black_scholes::bs_price;
use openferric::engines::fft::heston_price_fft;
use openferric::engines::pde::{
    AdiHestonEngine, AdiScheme, ExplicitFdEngine, HopscotchEngine, ImplicitFdEngine,
};
use openferric::instruments::vanilla::VanillaOption;
use openferric::market::Market;
use openferric::models::stochastic::Heston;

fn rel_err(x: f64, y: f64) -> f64 {
    let denom = y.abs().max(1.0e-8);
    (x - y).abs() / denom
}

fn vanilla_market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.20)
        .build()
        .expect("valid market")
}

#[test]
fn european_call_put_match_black_scholes_within_half_percent() {
    let market = vanilla_market();
    let call = VanillaOption::european_call(100.0, 1.0);
    let put = VanillaOption::european_put(100.0, 1.0);

    let bs_call = bs_price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
    let bs_put = bs_price(OptionType::Put, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

    let explicit = ExplicitFdEngine::new(3_000, 120)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&call, &market)
        .expect("explicit call price");
    let explicit_put = ExplicitFdEngine::new(3_000, 120)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&put, &market)
        .expect("explicit put price");

    let implicit = ImplicitFdEngine::new(320, 160)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&call, &market)
        .expect("implicit call price");
    let implicit_put = ImplicitFdEngine::new(320, 160)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&put, &market)
        .expect("implicit put price");

    let hopscotch = HopscotchEngine::new(320, 160)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&call, &market)
        .expect("hopscotch call price");
    let hopscotch_put = HopscotchEngine::new(320, 160)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&put, &market)
        .expect("hopscotch put price");

    assert!(rel_err(explicit.price, bs_call) <= 0.005);
    assert!(rel_err(explicit_put.price, bs_put) <= 0.005);
    assert!(rel_err(implicit.price, bs_call) <= 0.005);
    assert!(rel_err(implicit_put.price, bs_put) <= 0.005);
    assert!(rel_err(hopscotch.price, bs_call) <= 0.005);
    assert!(rel_err(hopscotch_put.price, bs_put) <= 0.005);
}

#[test]
fn american_put_exceeds_european_put_for_1d_solvers() {
    let option_eu = VanillaOption::european_put(100.0, 1.0);
    let option_am = VanillaOption::american_put(100.0, 1.0);
    let market = Market::builder()
        .spot(100.0)
        .rate(0.06)
        .dividend_yield(0.0)
        .flat_vol(0.25)
        .build()
        .expect("valid market");

    let ex_engine = ExplicitFdEngine::new(1_400, 90)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.15);
    let eu_ex = ex_engine
        .price(&option_eu, &market)
        .expect("explicit eu put");
    let am_ex = ex_engine
        .price(&option_am, &market)
        .expect("explicit am put");
    assert!(
        am_ex.price > eu_ex.price,
        "explicit American put should exceed European put"
    );

    let im_engine = ImplicitFdEngine::new(260, 140)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.15);
    let eu_im = im_engine
        .price(&option_eu, &market)
        .expect("implicit eu put");
    let am_im = im_engine
        .price(&option_am, &market)
        .expect("implicit am put");
    assert!(
        am_im.price > eu_im.price,
        "implicit American put should exceed European put"
    );

    let hs_engine = HopscotchEngine::new(260, 140)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.15);
    let eu_hs = hs_engine
        .price(&option_eu, &market)
        .expect("hopscotch eu put");
    let am_hs = hs_engine
        .price(&option_am, &market)
        .expect("hopscotch am put");
    assert!(
        am_hs.price > eu_hs.price,
        "hopscotch American put should exceed European put"
    );
}

#[test]
fn implicit_grid_refinement_improves_accuracy() {
    let market = vanilla_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let bs = bs_price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

    let coarse = ImplicitFdEngine::new(120, 90)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&option, &market)
        .expect("coarse implicit");
    let fine = ImplicitFdEngine::new(360, 180)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&option, &market)
        .expect("fine implicit");

    let coarse_err = (coarse.price - bs).abs();
    let fine_err = (fine.price - bs).abs();
    assert!(
        fine_err <= coarse_err,
        "finer grid should not increase BS error: coarse={coarse_err} fine={fine_err}"
    );
}

#[test]
fn explicit_solver_detects_cfl_violation() {
    let market = vanilla_market();
    let option = VanillaOption::european_call(100.0, 1.0);

    let err = ExplicitFdEngine::new(60, 180)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.15)
        .price(&option, &market)
        .expect_err("expected CFL violation to be detected");

    match err {
        PricingError::ConvergenceFailure(msg) => {
            assert!(msg.contains("CFL"), "expected CFL message, got {msg}");
        }
        other => panic!("unexpected error variant: {other}"),
    }
}

#[test]
fn hopscotch_and_implicit_have_comparable_accuracy() {
    let market = vanilla_market();
    let option = VanillaOption::european_put(100.0, 1.0);
    let bs = bs_price(OptionType::Put, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

    let implicit = ImplicitFdEngine::new(280, 160)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&option, &market)
        .expect("implicit put");
    let hopscotch = HopscotchEngine::new(280, 160)
        .with_s_max_multiplier(4.0)
        .with_grid_stretch(0.18)
        .price(&option, &market)
        .expect("hopscotch put");

    let implicit_err = (implicit.price - bs).abs();
    let hopscotch_err = (hopscotch.price - bs).abs();

    assert!(
        hopscotch_err <= 8.0 * implicit_err + 1.0e-4,
        "hopscotch should be reasonably close to implicit accuracy: hop={hopscotch_err} imp={implicit_err}"
    );
}

#[test]
fn adi_heston_matches_fft_reference_values() {
    let model = Heston {
        mu: 0.0,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.25,
        rho: -0.6,
        v0: 0.04,
    };

    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.20)
        .build()
        .expect("valid market");
    let option = VanillaOption::european_call(100.0, 1.0);

    let ref_call = heston_price_fft(
        market.spot,
        &[option.strike],
        market.rate,
        market.dividend_yield,
        model.v0,
        model.kappa,
        model.theta,
        model.xi,
        model.rho,
        option.expiry,
    )[0]
    .1;

    let dr = AdiHestonEngine::new(model, 120, 90, 60)
        .with_scheme(AdiScheme::DouglasRachford)
        .with_s_max_multiplier(4.0)
        .with_v_max_multiplier(5.0)
        .price(&option, &market)
        .expect("Douglas-Rachford price");
    let cs = AdiHestonEngine::new(model, 120, 90, 60)
        .with_scheme(AdiScheme::CraigSneyd)
        .with_s_max_multiplier(4.0)
        .with_v_max_multiplier(5.0)
        .price(&option, &market)
        .expect("Craig-Sneyd price");

    let dr_err = rel_err(dr.price, ref_call);
    let cs_err = rel_err(cs.price, ref_call);

    assert!(dr_err <= 0.04, "DR relative error too high: {dr_err}");
    assert!(cs_err <= 0.03, "CS relative error too high: {cs_err}");
}

#[test]
fn adi_enforces_feller_condition_when_requested() {
    let violating = Heston {
        mu: 0.0,
        kappa: 0.5,
        theta: 0.02,
        xi: 0.6,
        rho: -0.4,
        v0: 0.02,
    };

    let market = Market::builder()
        .spot(100.0)
        .rate(0.02)
        .dividend_yield(0.0)
        .flat_vol(0.25)
        .build()
        .expect("valid market");
    let option = VanillaOption::european_call(100.0, 1.0);

    let err = AdiHestonEngine::new(violating, 80, 60, 40)
        .with_enforce_feller(true)
        .price(&option, &market)
        .expect_err("expected Feller-condition validation error");

    match err {
        PricingError::InvalidInput(msg) => {
            assert!(msg.contains("Feller"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error variant: {other}"),
    }
}
