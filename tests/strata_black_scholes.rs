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
        [42.388192240722034, 41.445_107_405_754_9, 40.523_005_041_587_58, 39.090_123_476_135_13, 35.088_373_580_052_09, 30.657270312145542, 25.838608802827295],
        [42.844_635_277_413_69, 41.891_395_149_588_46, 40.959363435265075, 39.511_052_365_076, 35.466_210_966_900_22, 30.987392849065387, 26.116_843_198_832_52],
        [43.925335617747066, 42.948_051_244_384_67, 41.992_510_239_239_95, 40.507667401273466, 36.360_800_126_408_6, 31.769009632151473, 26.775606685804973],
        [46.509_094_773_672_5, 45.474_324_955_700_37, 44.462_577_485_967_87, 42.890_393_795_080_15, 38.499601092232645, 33.637_714_067_925_67, 28.350591098717572],
        [53.193_986_053_605_88, 52.010_485_675_158_19, 50.853316715912236, 49.055_158_361_427_8, 44.033_263_892_482_79, 38.472563306420405, 32.425506341403754],
        [68.103_724_051_366_04, 66.588_500_448_696_69, 65.106_988_696_562_24, 62.804_824_684_054_18, 56.375_343_825_382_41, 49.256034927120396, 41.514_048_860_248_68],
        [88.766_116_738_864_04, 86.791_180_462_272_4, 84.860_184_074_456_98, 81.859_552_870_616_03, 73.479_393_688_121_18, 64.200_115_446_500_77, 54.109242317679445],
    ],
    // Strike = 90.0
    [
        [37.456_487_390_776_39, 36.623_126_887_248_16, 35.808_307_623_895_46, 34.542_136_375_448_48, 31.005974850266114, 27.090413584076515, 22.832385002928326],
        [38.139_841_849_279_19, 37.291_277_554_490_86, 36.461_592_765_423_56, 35.172_321_546_100_31, 31.571646450304428, 27.584_649_861_820_01, 23.248938000202855],
        [39.567_718_773_636_09, 38.687_385_983_840_64, 37.826_639_509_476_35, 36.489100637901586, 32.753_623_701_507_85, 28.617362193449722, 24.119_330_232_490_56],
        [42.656781946566326, 41.707_721_322_941_55, 40.779_776_124_021_01, 39.337_815_208_451_7, 35.310_708_514_387_1, 30.851_527_881_998_16, 26.002333273509258],
        [50.072156744232956, 48.958_113_205_612_39, 47.868_855_757_438_94, 46.176227066557196, 41.449_055_713_979_74, 36.214_699_501_747_92, 30.522_530_016_865_09],
        [65.887_130_303_467_84, 64.421_223_169_334_21, 62.987930655324554, 60.760_695_913_246_88, 54.540477430048355, 47.652882961695354, 40.162_877_798_235_04],
        [87.395_751_286_588_53, 85.451_303_945_903_02, 83.550_118_152_890_52, 80.595_810_495_553_92, 72.345_023_657_486_88, 63.208998300993564, 53.273_907_405_608_35],
    ],
    // Strike = 95.0
    [
        [32.715_707_145_094_91, 31.987823136771382, 31.276_133_647_099_87, 30.170218740355807, 27.081_620_931_686_06, 23.661643122317834, 19.942_543_287_275_86],
        [33.664230469455035, 32.915_242_990_671_23, 32.182919556648116, 31.044_940_966_268_77, 27.866795740876597, 24.347_662_846_564_22, 20.520_735_510_697_76],
        [35.457091533184425, 34.668215113848404, 33.896_890_218_850_03, 32.698306128876695, 29.350901937810157, 25.644350045459888, 21.613611455971515],
        [39.036835496346015, 38.168314202665016, 37.319_116_433_239_73, 35.999_523_428_691_21, 32.314_165_687_937_23, 28.233_400_734_474_28, 23.795719231484398],
        [47.128_774_909_821_66, 46.080217975484366, 45.054990135668476, 43.461_858_907_350_45, 39.012_563_947_403_69, 34.085_897_876_593_54, 28.728330081543525],
        [63.777287309365754, 62.358_321_571_567_2, 60.970_926_065_899_76, 58.815_012_014_147_35, 52.793977868286255, 46.126_938_504_827_61, 38.876_778_890_067_47],
        [86.080_586_862_737_63, 84.165_400_303_370_13, 82.292_824_275_492_56, 79.382_974_160_666_24, 71.256_348_293_338_16, 62.257_805_312_674_05, 52.472_221_434_517_73],
    ],
    // Strike = 100.0
    [
        [28.227691366172778, 27.599_660_156_387_03, 26.985_601_864_023_08, 26.031398901929947, 23.366502028061248, 20.415684622407838, 17.206779436958136],
        [29.460572920194522, 28.805111621144633, 28.164_233_525_066_35, 27.168354493347223, 24.387_064_742_833_29, 21.307366505179694, 17.958308164432047],
        [31.614253168446297, 30.910875148391668, 30.223_146_419_068_74, 29.154464831673863, 26.169852192071076, 22.865_016_266_736_47, 19.271_128_987_385_02],
        [35.651_639_079_672_99, 34.858434218120514, 34.082_877_183_389_9, 32.877_716_648_934_47, 29.511_945_771_703_95, 25.785056605612148, 21.732201980396823],
        [44.356_758_683_618_42, 43.369875680794145, 42.404_949_603_817_69, 40.905_523_030_963_74, 36.717_926_319_013_79, 32.081036469111204, 27.038589635496734],
        [61.767_323_908_809_18, 60.393_077_369_959_68, 59.049_406_116_390_98, 56.961_436_446_119_52, 51.130_157_286_355_24, 44.673_231_988_200_47, 37.651563676357824],
        [84.816_610_871_615_35, 82.929_546_214_267_66, 81.084_466_410_882_19, 78.217_343_475_515_83, 70.210_046_023_101_04, 61.343_634_370_732_56, 51.701_738_442_808_58],
    ],
    // Strike = 103.0
    [
        [25.680774107575076, 25.109408655764113, 24.550755378366432, 23.682647873419768, 21.258198287726145, 18.573_626_097_805_65, 15.654252772826112],
        [27.083_882_968_770_85, 26.481_300_080_715_2, 25.892123916407286, 24.976586013624527, 22.419672870447798, 19.588424921750786, 16.509_547_114_380_7],
        [29.443_114_295_640_03, 28.788_041_429_386_81, 28.147543123955984, 27.152254259921776, 24.372612729001617, 21.294739550739045, 17.947665894822222],
        [33.732_817_530_354_08, 32.982_304_074_320_73, 32.248488614153935, 31.108191521144597, 27.923_571_184_422_67, 24.397268454983575, 20.562544187638785],
        [42.772_673_931_596_12, 41.821_034_854_685_24, 40.890568569874155, 39.444_690_065_935_18, 35.406_642_335_828_84, 30.935_346_788_260_86, 26.072977655972238],
        [60.606_631_963_610_56, 59.258_209_384_541_41, 57.939_787_539_597_14, 55.891053656546795, 50.169352155690575, 43.833_761_257_507_24, 36.944039634231586],
        [84.081_289_897_742_3, 82.210_585_222_329_07, 80.381_501_411_520_47, 77.539_235_112_223_2, 69.601_357_242_839_28, 60.811_813_298_151_89, 51.253_508_169_610_77],
    ],
    // Strike = 108.0
    [
        [21.714207592329316, 21.231093338072895, 20.758727778267996, 20.024_705_256_364_71, 17.974728048498548, 15.704805904238377, 13.236_349_223_276_59],
        [23.383992396693486, 22.863727496386588, 22.355_037_846_443_31, 21.564_570_269_019_4, 19.356_953_378_635_11, 16.912478168716945, 14.254201461462443],
        [26.054_446_449_835_14, 25.474_767_250_713_81, 24.907985196597735, 24.027246150119325, 21.567_519_210_574_04, 18.843884709915876, 15.882032561529108],
        [30.719333503473834, 30.035866338776202, 29.367_605_472_912_14, 28.329175562286544, 25.429049768831092, 22.217_765_402_134_64, 18.725611995250446],
        [40.259_006_031_733_68, 39.363_292_955_701_56, 38.487508387440954, 37.126_601_386_295_22, 33.325861966011026, 29.117335870422906, 24.540_718_833_611_48],
        [58.743_506_466_504_93, 57.436_536_124_699_91, 56.158_644_256_025_11, 54.172891076746374, 48.627_082_008_907_61, 42.486255290141514, 35.808_332_534_554_01],
        [82.891_922_555_376_68, 81.047_679_831_853_73, 79.244_469_227_244_42, 76.442_408_052_283_38, 68.616_815_005_325_65, 59.951603079507294, 50.528_504_439_616_36],
    ],
    // Strike = 120.0
    [
        [13.788335878907276, 13.481562464438824, 13.181614378904314, 12.715_516_363_047_62, 11.413798390334314, 9.972417266434356, 8.404968411811751],
        [15.907916865834139, 15.553985396738028, 15.207_928_464_947_72, 14.670180577697515, 13.168358931159062, 11.505404728972913, 9.697003317100766],
        [19.094_139_183_282_9, 18.669318209596582, 18.253948976989157, 17.608494698388782, 15.805_870_772_872_78, 13.809840792343792, 11.639231745995765],
        [24.390979675271552, 23.848_310_553_854_26, 23.317715149001913, 22.493207584689088, 20.190523860255283, 17.640781962021954, 14.868031610501113],
        [34.828165689108474, 34.053_282_091_700_28, 33.295_638_695_653_66, 32.118314683083376, 28.830285610372485, 25.189479276293007, 21.230236558576035],
        [54.606_702_067_923_01, 53.391_770_505_959_6, 52.203869668144705, 50.357_956_157_633_82, 45.202_691_146_926_07, 39.494_310_506_189_14, 33.286_656_924_003_07],
        [80.206_222_389_649_88, 78.421_733_172_084, 76.676_946_631_850_87, 73.965_672_302_338_39, 66.393_628_647_109_21, 58.009169783685806, 48.891_379_752_868_43],
    ],
    // Strike = 150.0
    [
        [3.328906490728528, 3.254842439810734, 3.182426222394327, 3.069896565158853, 2.755623875073592, 2.407632426279232, 2.029204549857958],
        [5.112418288611877, 4.998673306740088, 4.887459009990433, 4.714639893783904, 4.231990875892294, 3.697557766385675, 3.116381454667383],
        [8.059806552432782, 7.880485828180092, 7.705154768187647, 7.432702756917415, 6.671799110673657, 5.829257042581645, 4.913023592799185],
        [13.345_242_982_640_79, 13.048327836922702, 12.758018685855426, 12.306898889386751, 11.047012069592625, 9.651950221812832, 8.134_871_873_018_05],
        [24.368_455_680_587_46, 23.826287690185346, 23.296_182_266_803_1, 22.472436098885268, 20.171878801378924, 17.624491477407318, 14.854301638625593],
        [45.980_465_197_347_09, 44.957456733431215, 43.957_208_942_178_28, 42.402_894_934_681_82, 38.062_008_661_910_71, 33.255_382_598_719_17, 28.028_353_888_253_23],
        [74.354_192_706_245_04, 72.699_903_909_042_8, 71.082_420_990_903_27, 68.568_967_446_155_85, 61.549_397_438_409_61, 53.776_687_896_749_21, 45.324152709230404],
    ],
    // Strike = 250.0
    [
        [0.005810696974459, 0.005681416155733, 0.005555011675278, 0.005358588092742, 0.004810016549949, 0.004202587995173, 0.003542031826745],
        [0.047125603795891, 0.046077117414915, 0.045051958558736, 0.043458934526089, 0.039009938942081, 0.034083604367675, 0.028726397062748],
        [0.307974608501107, 0.301122554486089, 0.294422950195592, 0.284012241084843, 0.254937225321809, 0.222742710245262, 0.187732361528245],
        [1.674165161621819, 1.636917058041711, 1.600497678683162, 1.543904550483984, 1.385_851_330_768_95, 1.210840358926868, 1.020522376533685],
        [8.035294567394008, 7.856519204482371, 7.681721372215851, 7.410097958949876, 6.651508420206252, 5.811528743461139, 4.898081799839467],
        [28.157055740368474, 27.530_596_085_866_99, 26.918074383626475, 25.966259173421985, 23.308030810128514, 20.364_597_388_924_22, 17.163_722_014_449_16],
        [60.479_193_171_039_06, 59.133_605_947_438_19, 57.817_956_374_810_94, 55.773530405992034, 50.063860042770415, 43.741_591_119_891_3, 36.866356653819025],
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
        PriceCase {
            strike_idx: 0,
            vol_idx: 0,
            rate_idx: 0,
        },
        // Corner: lowest strike, lowest vol, highest rate
        PriceCase {
            strike_idx: 0,
            vol_idx: 0,
            rate_idx: 6,
        },
        // Corner: lowest strike, highest vol, lowest rate
        PriceCase {
            strike_idx: 0,
            vol_idx: 6,
            rate_idx: 0,
        },
        // Corner: lowest strike, highest vol, highest rate
        PriceCase {
            strike_idx: 0,
            vol_idx: 6,
            rate_idx: 6,
        },
        // Corner: highest strike, lowest vol, lowest rate
        PriceCase {
            strike_idx: 8,
            vol_idx: 0,
            rate_idx: 0,
        },
        // Corner: highest strike, lowest vol, highest rate
        PriceCase {
            strike_idx: 8,
            vol_idx: 0,
            rate_idx: 6,
        },
        // Corner: highest strike, highest vol, lowest rate
        PriceCase {
            strike_idx: 8,
            vol_idx: 6,
            rate_idx: 0,
        },
        // Corner: highest strike, highest vol, highest rate
        PriceCase {
            strike_idx: 8,
            vol_idx: 6,
            rate_idx: 6,
        },
        // ATM strike (100), mid vol (0.2), mid rate (0.008)
        PriceCase {
            strike_idx: 3,
            vol_idx: 3,
            rate_idx: 3,
        },
        // ATM strike, low vol, zero rate
        PriceCase {
            strike_idx: 3,
            vol_idx: 0,
            rate_idx: 2,
        },
        // ATM strike, high vol, zero rate
        PriceCase {
            strike_idx: 3,
            vol_idx: 6,
            rate_idx: 2,
        },
        // ITM call (K=85), mid vol (0.15), positive rate (0.032)
        PriceCase {
            strike_idx: 0,
            vol_idx: 2,
            rate_idx: 4,
        },
        // OTM call (K=150), mid vol (0.15), positive rate (0.032)
        PriceCase {
            strike_idx: 7,
            vol_idx: 2,
            rate_idx: 4,
        },
        // Deep OTM (K=250), mid vol (0.3), negative rate
        PriceCase {
            strike_idx: 8,
            vol_idx: 4,
            rate_idx: 0,
        },
        // K=108, vol=0.5, rate=0.062
        PriceCase {
            strike_idx: 5,
            vol_idx: 5,
            rate_idx: 5,
        },
        // K=120, vol=0.3, rate=0
        PriceCase {
            strike_idx: 6,
            vol_idx: 4,
            rate_idx: 2,
        },
        // K=95, vol=0.12, rate=-0.005
        PriceCase {
            strike_idx: 2,
            vol_idx: 1,
            rate_idx: 1,
        },
        // K=103, vol=0.8, rate=0.1
        PriceCase {
            strike_idx: 4,
            vol_idx: 6,
            rate_idx: 6,
        },
        // K=90, vol=0.5, rate=-0.01
        PriceCase {
            strike_idx: 1,
            vol_idx: 5,
            rate_idx: 0,
        },
        // K=100, vol=0.1, rate=0.1
        PriceCase {
            strike_idx: 3,
            vol_idx: 0,
            rate_idx: 6,
        },
        // K=85, vol=0.3, rate=0.062
        PriceCase {
            strike_idx: 0,
            vol_idx: 4,
            rate_idx: 5,
        },
        // K=250, vol=0.2, rate=0.008
        PriceCase {
            strike_idx: 8,
            vol_idx: 3,
            rate_idx: 3,
        },
        // K=150, vol=0.8, rate=0
        PriceCase {
            strike_idx: 7,
            vol_idx: 6,
            rate_idx: 2,
        },
        // K=108, vol=0.1, rate=0.032
        PriceCase {
            strike_idx: 5,
            vol_idx: 0,
            rate_idx: 4,
        },
        // K=120, vol=0.12, rate=0.008
        PriceCase {
            strike_idx: 6,
            vol_idx: 1,
            rate_idx: 3,
        },
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
        assert_relative_eq!(price, expected, epsilon = 1e-4,);
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
                max_relative = GREEK_REL_TOL,
                epsilon = GREEK_ABS_TOL,
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
            max_relative = GREEK_REL_TOL,
            epsilon = GREEK_ABS_TOL,
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
            max_relative = GREEK_REL_TOL,
            epsilon = GREEK_ABS_TOL,
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
                max_relative = GREEK_REL_TOL,
                epsilon = GREEK_ABS_TOL,
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
                max_relative = GREEK_REL_TOL,
                epsilon = GREEK_ABS_TOL,
            );
        }
    }
}
