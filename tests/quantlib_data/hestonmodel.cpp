// Minimal fixture extracted from QuantLib test-suite/hestonmodel.cpp.
// Source context: BOOST_AUTO_TEST_CASE(testPiecewiseTimeDependentChFAsymtotic).
// The expected table below is also reproduced by:
// "Option Pricing Formulae using Fourier Transform" (Alan L. Lewis).

const Real spot = 100.0;
const Rate riskFreeRate = 0.01;
const Rate dividendRate = 0.02;
const Time maturity = 1.0;

// Lewis notation: dv_t = (omega - theta_speed * v_t) dt + xi sqrt(v_t) dW_t.
// QuantLib/OpenFerric Heston notation: dv_t = kappa(theta - v_t) dt + sigma_v sqrt(v_t) dW_t.
// Mapping used in QuantLib tests:
//   omega = 1.0
//   theta_speed = 4.0
// Therefore:
const Volatility v0 = 0.04;
const Real kappa = 4.0;
const Volatility theta = 0.25;
const Volatility sigma_v = 1.0;
const Real rho = -0.5;

const Real strikes[] = {80.0, 90.0, 100.0, 110.0, 120.0};

// columns copied verbatim from QuantLib expectedResults[][2]
const Real expectedResults[][2] = {
    {7.958878113256768285, 26.774758743998854958},
    {10.641309158052722944, 21.416638084984698542},
    {14.050704070679939120, 17.055270632522355568},
    {18.235417170836522467, 13.678708969543181513},
    {23.084213651877347073, 11.167487369935069554}
};
