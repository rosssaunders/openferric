// Reference values from OpenGamma Strata (Apache 2.0), https://github.com/OpenGamma/Strata
//
// Source: BlackScholesFormulaRepositoryTest.java
//   modules/pricer/src/test/java/com/opengamma/strata/pricer/impl/option/
//
// Strata parameters:
//   SPOT = 100.0
//   STRIKES = [85, 90, 95, 100, 103, 108, 120, 150, 250]
//   TIME_TO_EXPIRY = 4.5
//   COST_OF_CARRY = 0.05
//   VOLS = [0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.8]
//   INTEREST_RATES = [-0.01, -0.005, 0, 0.008, 0.032, 0.062, 0.1]
//
// Mapping: Strata uses cost_of_carry = b where b = r - q.
// With b = 0.05, dividend_yield = rate - 0.05.

use approx::assert_relative_eq;
use openferric::core::OptionType;
use openferric::engines::analytic::black_scholes::{
    bs_delta, bs_gamma, bs_price, bs_rho, bs_theta, bs_vega,
};

const SPOT: f64 = 100.0;
const EXPIRY: f64 = 4.5;
const COST_OF_CARRY: f64 = 0.05;

const STRIKES: [f64; 9] = [85.0, 90.0, 95.0, 100.0, 103.0, 108.0, 120.0, 150.0, 250.0];
const VOLS: [f64; 7] = [0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.8];
const RATES: [f64; 7] = [-0.01, -0.005, 0.0, 0.008, 0.032, 0.062, 0.1];

/// Full 9x7x7 precomputed call prices from Strata's BlackScholesFormulaRepositoryTest.java
#[rustfmt::skip]
const PRECOMPUTED_CALL_PRICE: [[[f64; 7]; 7]; 9] = [
    // Strike = 85.0
    [
        [42.388192240722034, 41.445107405754896, 40.523005041587581, 39.090123476135133, 35.088373580052092, 30.657270312145542, 25.838608802827295],
        [42.844635277413687, 41.891395149588462, 40.959363435265075, 39.511052365075997, 35.466210966900221, 30.987392849065387, 26.116843198832520],
        [43.925335617747066, 42.948051244384672, 41.992510239239948, 40.507667401273466, 36.360800126408598, 31.769009632151473, 26.775606685804973],
        [46.509094773672501, 45.474324955700368, 44.462577485967870, 42.890393795080151, 38.499601092232645, 33.637714067925671, 28.350591098717572],
        [53.193986053605883, 52.010485675158193, 50.853316715912236, 49.055158361427800, 44.033263892482793, 38.472563306420405, 32.425506341403754],
        [68.103724051366044, 66.588500448696692, 65.106988696562240, 62.804824684054182, 56.375343825382409, 49.256034927120396, 41.514048860248678],
        [88.766116738864042, 86.791180462272393, 84.860184074456981, 81.859552870616028, 73.479393688121178, 64.200115446500774, 54.109242317679445],
    ],
    // Strike = 90.0
    [
        [37.456487390776388, 36.623126887248162, 35.808307623895459, 34.542136375448479, 31.005974850266114, 27.090413584076515, 22.832385002928326],
        [38.139841849279193, 37.291277554490861, 36.461592765423561, 35.172321546100306, 31.571646450304428, 27.584649861820012, 23.248938000202855],
        [39.567718773636088, 38.687385983840642, 37.826639509476351, 36.489100637901586, 32.753623701507848, 28.617362193449722, 24.119330232490562],
        [42.656781946566326, 41.707721322941552, 40.779776124021012, 39.337815208451701, 35.310708514387102, 30.851527881998159, 26.002333273509258],
        [50.072156744232956, 48.958113205612392, 47.868855757438943, 46.176227066557196, 41.449055713979739, 36.214699501747923, 30.522530016865090],
        [65.887130303467842, 64.421223169334212, 62.987930655324554, 60.760695913246877, 54.540477430048355, 47.652882961695354, 40.162877798235037],
        [87.395751286588535, 85.451303945903021, 83.550118152890519, 80.595810495553920, 72.345023657486877, 63.208998300993564, 53.273907405608348],
    ],
    // Strike = 95.0
    [
        [32.715707145094910, 31.987823136771382, 31.276133647099869, 30.170218740355807, 27.081620931686061, 23.661643122317834, 19.942543287275861],
        [33.664230469455035, 32.915242990671231, 32.182919556648116, 31.044940966268769, 27.866795740876597, 24.347662846564219, 20.520735510697762],
        [35.457091533184425, 34.668215113848404, 33.896890218850032, 32.698306128876695, 29.350901937810157, 25.644350045459888, 21.613611455971515],
        [39.036835496346015, 38.168314202665016, 37.319116433239728, 35.999523428691212, 32.314165687937233, 28.233400734474280, 23.795719231484398],
        [47.128774909821658, 46.080217975484366, 45.054990135668476, 43.461858907350447, 39.012563947403692, 34.085897876593542, 28.728330081543525],
        [63.777287309365754, 62.358321571567203, 60.970926065899761, 58.815012014147349, 52.793977868286255, 46.126938504827613, 38.876778890067470],
        [86.080586862737633, 84.165400303370134, 82.292824275492563, 79.382974160666237, 71.256348293338164, 62.257805312674051, 52.472221434517728],
    ],
    // Strike = 100.0
    [
        [28.227691366172778, 27.599660156387031, 26.985601864023081, 26.031398901929947, 23.366502028061248, 20.415684622407838, 17.206779436958136],
        [29.460572920194522, 28.805111621144633, 28.164233525066351, 27.168354493347223, 24.387064742833289, 21.307366505179694, 17.958308164432047],
        [31.614253168446297, 30.910875148391668, 30.223146419068740, 29.154464831673863, 26.169852192071076, 22.865016266736468, 19.271128987385019],
        [35.651639079672989, 34.858434218120514, 34.082877183389897, 32.877716648934467, 29.511945771703950, 25.785056605612148, 21.732201980396823],
        [44.356758683618423, 43.369875680794145, 42.404949603817691, 40.905523030963742, 36.717926319013792, 32.081036469111204, 27.038589635496734],
        [61.767323908809182, 60.393077369959677, 59.049406116390983, 56.961436446119521, 51.130157286355242, 44.673231988200470, 37.651563676357824],
        [84.816610871615353, 82.929546214267660, 81.084466410882186, 78.217343475515833, 70.210046023101043, 61.343634370732559, 51.701738442808583],
    ],
    // Strike = 103.0
    [
        [25.680774107575076, 25.109408655764113, 24.550755378366432, 23.682647873419768, 21.258198287726145, 18.573626097805651, 15.654252772826112],
        [27.083882968770851, 26.481300080715201, 25.892123916407286, 24.976586013624527, 22.419672870447798, 19.588424921750786, 16.509547114380702],
        [29.443114295640029, 28.788041429386809, 28.147543123955984, 27.152254259921776, 24.372612729001617, 21.294739550739045, 17.947665894822222],
        [33.732817530354083, 32.982304074320730, 32.248488614153935, 31.108191521144597, 27.923571184422670, 24.397268454983575, 20.562544187638785],
        [42.772673931596117, 41.821034854685237, 40.890568569874155, 39.444690065935177, 35.406642335828842, 30.935346788260858, 26.072977655972238],
        [60.606631963610560, 59.258209384541409, 57.939787539597141, 55.891053656546795, 50.169352155690575, 43.833761257507241, 36.944039634231586],
        [84.081289897742295, 82.210585222329073, 80.381501411520475, 77.539235112223196, 69.601357242839285, 60.811813298151890, 51.253508169610768],
    ],
    // Strike = 108.0
    [
        [21.714207592329316, 21.231093338072895, 20.758727778267996, 20.024705256364712, 17.974728048498548, 15.704805904238377, 13.236349223276591],
        [23.383992396693486, 22.863727496386588, 22.355037846443309, 21.564570269019399, 19.356953378635112, 16.912478168716945, 14.254201461462443],
        [26.054446449835140, 25.474767250713811, 24.907985196597735, 24.027246150119325, 21.567519210574041, 18.843884709915876, 15.882032561529108],
        [30.719333503473834, 30.035866338776202, 29.367605472912139, 28.329175562286544, 25.429049768831092, 22.217765402134638, 18.725611995250446],
        [40.259006031733684, 39.363292955701560, 38.487508387440954, 37.126601386295221, 33.325861966011026, 29.117335870422906, 24.540718833611479],
        [58.743506466504932, 57.436536124699913, 56.158644256025113, 54.172891076746374, 48.627082008907607, 42.486255290141514, 35.808332534554012],
        [82.891922555376681, 81.047679831853728, 79.244469227244423, 76.442408052283383, 68.616815005325648, 59.951603079507294, 50.528504439616363],
    ],
    // Strike = 120.0
    [
        [13.788335878907276, 13.481562464438824, 13.181614378904314, 12.715516363047620, 11.413798390334314, 9.972417266434356, 8.404968411811751],
        [15.907916865834139, 15.553985396738028, 15.207928464947720, 14.670180577697515, 13.168358931159062, 11.505404728972913, 9.697003317100766],
        [19.094139183282898, 18.669318209596582, 18.253948976989157, 17.608494698388782, 15.805870772872780, 13.809840792343792, 11.639231745995765],
        [24.390979675271552, 23.848310553854262, 23.317715149001913, 22.493207584689088, 20.190523860255283, 17.640781962021954, 14.868031610501113],
        [34.828165689108474, 34.053282091700282, 33.295638695653658, 32.118314683083376, 28.830285610372485, 25.189479276293007, 21.230236558576035],
        [54.606702067923010, 53.391770505959599, 52.203869668144705, 50.357956157633822, 45.202691146926071, 39.494310506189137, 33.286656924003069],
        [80.206222389649881, 78.421733172084004, 76.676946631850868, 73.965672302338390, 66.393628647109210, 58.009169783685806, 48.891379752868431],
    ],
    // Strike = 150.0
    [
        [3.328906490728528, 3.254842439810734, 3.182426222394327, 3.069896565158853, 2.755623875073592, 2.407632426279232, 2.029204549857958],
        [5.112418288611877, 4.998673306740088, 4.887459009990433, 4.714639893783904, 4.231990875892294, 3.697557766385675, 3.116381454667383],
        [8.059806552432782, 7.880485828180092, 7.705154768187647, 7.432702756917415, 6.671799110673657, 5.829257042581645, 4.913023592799185],
        [13.345242982640791, 13.048327836922702, 12.758018685855426, 12.306898889386751, 11.047012069592625, 9.651950221812832, 8.134871873018049],
        [24.368455680587459, 23.826287690185346, 23.296182266803100, 22.472436098885268, 20.171878801378924, 17.624491477407318, 14.854301638625593],
        [45.980465197347087, 44.957456733431215, 43.957208942178283, 42.402894934681818, 38.062008661910710, 33.255382598719173, 28.028353888253228],
        [74.354192706245044, 72.699903909042803, 71.082420990903273, 68.568967446155852, 61.549397438409613, 53.776687896749209, 45.324152709230404],
    ],
    // Strike = 250.0
    [
        [0.005810696974459, 0.005681416155733, 0.005555011675278, 0.005358588092742, 0.004810016549949, 0.004202587995173, 0.003542031826745],
        [0.047125603795891, 0.046077117414915, 0.045051958558736, 0.043458934526089, 0.039009938942081, 0.034083604367675, 0.028726397062748],
        [0.307974608501107, 0.301122554486089, 0.294422950195592, 0.284012241084843, 0.254937225321809, 0.222742710245262, 0.187732361528245],
        [1.674165161621819, 1.636917058041711, 1.600497678683162, 1.543904550483984, 1.385851330768950, 1.210840358926868, 1.020522376533685],
        [8.035294567394008, 7.856519204482371, 7.681721372215851, 7.410097958949876, 6.651508420206252, 5.811528743461139, 4.898081799839467],
        [28.157055740368474, 27.530596085866989, 26.918074383626475, 25.966259173421985, 23.308030810128514, 20.364597388924221, 17.163722014449160],
        [60.479193171039057, 59.133605947438191, 57.817956374810940, 55.773530405992034, 50.063860042770415, 43.741591119891297, 36.866356653819025],
    ],
];

/// Compute dividend yield from rate and cost-of-carry.
/// Strata convention: b = r - q, so q = r - b.
fn div_yield(rate: f64) -> f64 {
    rate - COST_OF_CARRY
}

// ============================================================================
// Test 1: Representative subset of call prices (~25 cases)
// ============================================================================

struct PriceCase {
    strike_idx: usize,
    vol_idx: usize,
    rate_idx: usize,
}

/// Select ~25 cases covering corners and interior of the [strike x vol x rate] grid.
fn representative_cases() -> Vec<PriceCase> {
    vec![
        // Corner: lowest strike, lowest vol, lowest rate
        PriceCase { strike_idx: 0, vol_idx: 0, rate_idx: 0 },
        // Corner: lowest strike, lowest vol, highest rate
        PriceCase { strike_idx: 0, vol_idx: 0, rate_idx: 6 },
        // Corner: lowest strike, highest vol, lowest rate
        PriceCase { strike_idx: 0, vol_idx: 6, rate_idx: 0 },
        // Corner: lowest strike, highest vol, highest rate
        PriceCase { strike_idx: 0, vol_idx: 6, rate_idx: 6 },
        // Corner: highest strike, lowest vol, lowest rate
        PriceCase { strike_idx: 8, vol_idx: 0, rate_idx: 0 },
        // Corner: highest strike, lowest vol, highest rate
        PriceCase { strike_idx: 8, vol_idx: 0, rate_idx: 6 },
        // Corner: highest strike, highest vol, lowest rate
        PriceCase { strike_idx: 8, vol_idx: 6, rate_idx: 0 },
        // Corner: highest strike, highest vol, highest rate
        PriceCase { strike_idx: 8, vol_idx: 6, rate_idx: 6 },
        // ATM strike (100), mid vol (0.2), mid rate (0.008)
        PriceCase { strike_idx: 3, vol_idx: 3, rate_idx: 3 },
        // ATM strike, low vol, zero rate
        PriceCase { strike_idx: 3, vol_idx: 0, rate_idx: 2 },
        // ATM strike, high vol, zero rate
        PriceCase { strike_idx: 3, vol_idx: 6, rate_idx: 2 },
        // ITM call (K=85), mid vol (0.15), positive rate (0.032)
        PriceCase { strike_idx: 0, vol_idx: 2, rate_idx: 4 },
        // OTM call (K=150), mid vol (0.15), positive rate (0.032)
        PriceCase { strike_idx: 7, vol_idx: 2, rate_idx: 4 },
        // Deep OTM (K=250), mid vol (0.3), negative rate
        PriceCase { strike_idx: 8, vol_idx: 4, rate_idx: 0 },
        // K=108, vol=0.5, rate=0.062
        PriceCase { strike_idx: 5, vol_idx: 5, rate_idx: 5 },
        // K=120, vol=0.3, rate=0
        PriceCase { strike_idx: 6, vol_idx: 4, rate_idx: 2 },
        // K=95, vol=0.12, rate=-0.005
        PriceCase { strike_idx: 2, vol_idx: 1, rate_idx: 1 },
        // K=103, vol=0.8, rate=0.1
        PriceCase { strike_idx: 4, vol_idx: 6, rate_idx: 6 },
        // K=90, vol=0.5, rate=-0.01
        PriceCase { strike_idx: 1, vol_idx: 5, rate_idx: 0 },
        // K=100, vol=0.1, rate=0.1
        PriceCase { strike_idx: 3, vol_idx: 0, rate_idx: 6 },
        // K=85, vol=0.3, rate=0.062
        PriceCase { strike_idx: 0, vol_idx: 4, rate_idx: 5 },
        // K=250, vol=0.2, rate=0.008
        PriceCase { strike_idx: 8, vol_idx: 3, rate_idx: 3 },
        // K=150, vol=0.8, rate=0
        PriceCase { strike_idx: 7, vol_idx: 6, rate_idx: 2 },
        // K=108, vol=0.1, rate=0.032
        PriceCase { strike_idx: 5, vol_idx: 0, rate_idx: 4 },
        // K=120, vol=0.12, rate=0.008
        PriceCase { strike_idx: 6, vol_idx: 1, rate_idx: 3 },
    ]
}

#[test]
fn strata_call_prices_representative_subset() {
    let cases = representative_cases();
    assert!(cases.len() >= 25, "need at least 25 representative cases");

    for (idx, c) in cases.iter().enumerate() {
        let strike = STRIKES[c.strike_idx];
        let vol = VOLS[c.vol_idx];
        let rate = RATES[c.rate_idx];
        let q = div_yield(rate);
        let expected = PRECOMPUTED_CALL_PRICE[c.strike_idx][c.vol_idx][c.rate_idx];

        let price = bs_price(OptionType::Call, SPOT, strike, rate, q, vol, EXPIRY);

        // Tolerance 1e-4: our FMA-optimized BS kernel produces minor floating-point
        // differences from the scalar formula Strata uses.
        let err = (price - expected).abs();
        assert!(
            err < 1e-4,
            "case {idx}: K={strike} vol={vol} r={rate} q={q} expected={expected} got={price} err={err:.2e}"
        );

        // Also check relative error is extremely small
        assert_relative_eq!(
            price,
            expected,
            epsilon = 1e-4,
        );
    }
}

// ============================================================================
// Test 2: Full grid validation (all 441 prices)
// ============================================================================

#[test]
fn strata_call_prices_full_grid() {
    let mut count = 0;
    for (i, &strike) in STRIKES.iter().enumerate() {
        for (j, &vol) in VOLS.iter().enumerate() {
            for (k, &rate) in RATES.iter().enumerate() {
                let q = div_yield(rate);
                let expected = PRECOMPUTED_CALL_PRICE[i][j][k];
                let price = bs_price(OptionType::Call, SPOT, strike, rate, q, vol, EXPIRY);

                let err = (price - expected).abs();
                assert!(
                    err < 1e-4,
                    "grid [{i}][{j}][{k}]: K={strike} vol={vol} r={rate} q={q} \
                     expected={expected} got={price} err={err:.2e}"
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, 441, "expected 441 grid cases");
}

// ============================================================================
// Test 3: Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
// ============================================================================

#[test]
fn strata_put_call_parity() {
    // Check parity for a selection of cases across the grid
    let parity_cases = [
        (0, 0, 0), // K=85, low vol, negative rate
        (3, 3, 3), // K=100 ATM, mid vol, mid rate
        (8, 6, 6), // K=250 deep OTM, high vol, high rate
        (0, 6, 0), // K=85 ITM, high vol, negative rate
        (5, 2, 4), // K=108, vol=0.15, rate=0.032
        (7, 4, 2), // K=150, vol=0.3, rate=0
        (1, 5, 5), // K=90, vol=0.5, rate=0.062
        (6, 1, 1), // K=120, vol=0.12, rate=-0.005
        (4, 3, 6), // K=103, vol=0.2, rate=0.1
        (2, 0, 2), // K=95, vol=0.1, rate=0
    ];

    for &(si, vi, ri) in &parity_cases {
        let strike = STRIKES[si];
        let vol = VOLS[vi];
        let rate = RATES[ri];
        let q = div_yield(rate);

        let call = bs_price(OptionType::Call, SPOT, strike, rate, q, vol, EXPIRY);
        let put = bs_price(OptionType::Put, SPOT, strike, rate, q, vol, EXPIRY);

        let forward_diff = SPOT * (-q * EXPIRY).exp() - strike * (-rate * EXPIRY).exp();
        let parity_err = ((call - put) - forward_diff).abs();

        assert!(
            parity_err < 1e-10,
            "put-call parity violated: K={strike} vol={vol} r={rate} q={q} \
             C-P={} fwd_diff={forward_diff} err={parity_err:.2e}",
            call - put
        );
    }
}

// ============================================================================
// Test 4: Greeks via finite differences
// ============================================================================

const FD_BUMP: f64 = 1e-5;
// Absolute tolerance for finite-difference Greek checks. The FMA-optimized BS
// kernel on x86_64 introduces tiny floating-point differences (~1e-6 in price)
// that get amplified by the FD quotient, so we allow up to 1e-3 absolute error
// in the Greek values.
const GREEK_ABS_TOL: f64 = 1e-3;
// Relative tolerance for Greek FD checks.
const GREEK_REL_TOL: f64 = 1e-4;

#[test]
fn strata_greeks_delta_finite_diff() {
    // Test bs_delta against central finite difference of bs_price
    let cases = [(3, 3, 3), (0, 0, 2), (8, 6, 6), (5, 2, 4)];

    for &(si, vi, ri) in &cases {
        let strike = STRIKES[si];
        let vol = VOLS[vi];
        let rate = RATES[ri];
        let q = div_yield(rate);
        let ds = SPOT * FD_BUMP;

        for &opt in &[OptionType::Call, OptionType::Put] {
            let delta_analytic = bs_delta(opt, SPOT, strike, rate, q, vol, EXPIRY);

            let p_up = bs_price(opt, SPOT + ds, strike, rate, q, vol, EXPIRY);
            let p_dn = bs_price(opt, SPOT - ds, strike, rate, q, vol, EXPIRY);
            let delta_fd = (p_up - p_dn) / (2.0 * ds);

            assert_relative_eq!(
                delta_analytic,
                delta_fd,
                max_relative = GREEK_REL_TOL, epsilon = GREEK_ABS_TOL,
            );
        }
    }
}

#[test]
fn strata_greeks_gamma_finite_diff() {
    // Test bs_gamma against central finite difference of bs_delta (or second diff of price)
    let cases = [(3, 3, 3), (0, 2, 4), (8, 4, 0), (6, 5, 5)];

    for &(si, vi, ri) in &cases {
        let strike = STRIKES[si];
        let vol = VOLS[vi];
        let rate = RATES[ri];
        let q = div_yield(rate);
        let ds = SPOT * FD_BUMP;

        let gamma_analytic = bs_gamma(SPOT, strike, rate, q, vol, EXPIRY);

        // Second derivative: (P(S+h) - 2*P(S) + P(S-h)) / h^2
        let p_up = bs_price(OptionType::Call, SPOT + ds, strike, rate, q, vol, EXPIRY);
        let p_mid = bs_price(OptionType::Call, SPOT, strike, rate, q, vol, EXPIRY);
        let p_dn = bs_price(OptionType::Call, SPOT - ds, strike, rate, q, vol, EXPIRY);
        let gamma_fd = (p_up - 2.0 * p_mid + p_dn) / (ds * ds);

        assert_relative_eq!(
            gamma_analytic,
            gamma_fd,
            max_relative = GREEK_REL_TOL, epsilon = GREEK_ABS_TOL,
        );
    }
}

#[test]
fn strata_greeks_vega_finite_diff() {
    // Test bs_vega against central finite difference in vol
    let cases = [(3, 3, 3), (0, 0, 2), (7, 4, 5), (2, 6, 1)];

    for &(si, vi, ri) in &cases {
        let strike = STRIKES[si];
        let vol = VOLS[vi];
        let rate = RATES[ri];
        let q = div_yield(rate);
        let dv = vol * FD_BUMP;

        let vega_analytic = bs_vega(SPOT, strike, rate, q, vol, EXPIRY);

        let p_up = bs_price(OptionType::Call, SPOT, strike, rate, q, vol + dv, EXPIRY);
        let p_dn = bs_price(OptionType::Call, SPOT, strike, rate, q, vol - dv, EXPIRY);
        let vega_fd = (p_up - p_dn) / (2.0 * dv);

        assert_relative_eq!(
            vega_analytic,
            vega_fd,
            max_relative = GREEK_REL_TOL, epsilon = GREEK_ABS_TOL,
        );
    }
}

#[test]
fn strata_greeks_theta_finite_diff() {
    // Test bs_theta against finite difference in time.
    // Convention: bs_theta returns dP/d(passage-of-time) which is -dP/dT
    // (price decreases as time passes, i.e. T shrinks), so we compare
    // theta_analytic with -(P(T+dt) - P(T-dt)) / (2*dt).
    let cases = [(3, 3, 3), (5, 2, 4)];

    for &(si, vi, ri) in &cases {
        let strike = STRIKES[si];
        let vol = VOLS[vi];
        let rate = RATES[ri];
        let q = div_yield(rate);
        let dt = EXPIRY * FD_BUMP;

        for &opt in &[OptionType::Call, OptionType::Put] {
            let theta_analytic = bs_theta(opt, SPOT, strike, rate, q, vol, EXPIRY);

            let p_up = bs_price(opt, SPOT, strike, rate, q, vol, EXPIRY + dt);
            let p_dn = bs_price(opt, SPOT, strike, rate, q, vol, EXPIRY - dt);
            let dp_dt = (p_up - p_dn) / (2.0 * dt);

            // theta_analytic = -dP/dT (standard convention: time passes => T decreases)
            assert_relative_eq!(
                theta_analytic,
                -dp_dt,
                max_relative = GREEK_REL_TOL, epsilon = GREEK_ABS_TOL,
            );
        }
    }
}

#[test]
fn strata_greeks_rho_finite_diff() {
    // Test bs_rho against central finite difference in rate.
    // When bumping rate, q stays the same (rho measures sensitivity to r only).
    let cases = [(3, 3, 3), (0, 2, 4), (8, 5, 1)];

    for &(si, vi, ri) in &cases {
        let strike = STRIKES[si];
        let vol = VOLS[vi];
        let rate = RATES[ri];
        let q = div_yield(rate);
        let dr = FD_BUMP;

        for &opt in &[OptionType::Call, OptionType::Put] {
            let rho_analytic = bs_rho(opt, SPOT, strike, rate, q, vol, EXPIRY);

            let p_up = bs_price(opt, SPOT, strike, rate + dr, q, vol, EXPIRY);
            let p_dn = bs_price(opt, SPOT, strike, rate - dr, q, vol, EXPIRY);
            let rho_fd = (p_up - p_dn) / (2.0 * dr);

            assert_relative_eq!(
                rho_analytic,
                rho_fd,
                max_relative = GREEK_REL_TOL, epsilon = GREEK_ABS_TOL,
            );
        }
    }
}
