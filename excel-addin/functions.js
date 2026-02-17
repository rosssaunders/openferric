/**
 * OpenFerric Excel Custom Functions
 *
 * Loads the WASM module and registers Excel custom functions.
 * All computation runs client-side — no server calls.
 */

let wasm = null;

async function ensureWasm() {
  if (wasm) return wasm;
  // Try local pkg first (dev), then GH Pages (prod)
  try {
    const mod = await import('./pkg/openferric.js');
    await mod.default();
    wasm = mod;
  } catch {
    const mod = await import('https://rosssaunders.github.io/openferric/pkg/openferric.js');
    await mod.default();
    wasm = mod;
  }
  return wasm;
}

// =OPENFERRIC.BS_PRICE(spot, strike, rate, div_yield, vol, maturity, is_call)
async function bsPrice(spot, strike, rate, divYield, vol, maturity, isCall) {
  const w = await ensureWasm();
  return w.bs_price(spot, strike, rate, divYield, vol, maturity, isCall);
}

// =OPENFERRIC.BS_IMPLIED_VOL(price, spot, strike, rate, div_yield, maturity, is_call)
async function bsImpliedVol(price, spot, strike, rate, divYield, maturity, isCall) {
  const w = await ensureWasm();
  return w.bs_implied_vol(price, spot, strike, rate, divYield, maturity, isCall);
}

// =OPENFERRIC.BS_GREEKS(spot, strike, rate, div_yield, vol, maturity, is_call)
async function bsGreeks(spot, strike, rate, divYield, vol, maturity, isCall) {
  const w = await ensureWasm();
  const arr = w.bsm_greeks_wasm(spot, strike, rate, divYield, vol, maturity, isCall);
  // Returns: delta, gamma, vega, theta, rho, vanna, volga
  const labels = ['Δ', 'Γ', 'V', 'Θ', 'ρ', 'vanna', 'volga'];
  return labels.map((l, i) => `${l}=${arr[i].toFixed(6)}`).join(', ');
}

// =OPENFERRIC.HESTON_PRICE(spot, strike, rate, div_yield, v0, kappa, theta, sigma_v, rho, maturity, is_call)
async function hestonPrice(spot, strike, rate, divYield, v0, kappa, theta, sigmaV, rho, maturity, isCall) {
  const w = await ensureWasm();
  return w.heston_price(spot, strike, rate, divYield, v0, kappa, theta, sigmaV, rho, maturity, isCall);
}

// =OPENFERRIC.BARRIER_PRICE(spot, strike, barrier, rate, div_yield, vol, maturity, barrier_type, is_call)
async function barrierPrice(spot, strike, barrier, rate, divYield, vol, maturity, barrierType, isCall) {
  const w = await ensureWasm();
  return w.barrier_price(spot, strike, barrier, rate, divYield, vol, maturity, barrierType, isCall);
}

// =OPENFERRIC.BOND_PRICE(face, coupon_rate, maturity, yield, frequency)
async function bondPrice(face, couponRate, maturity, yieldRate, frequency) {
  const w = await ensureWasm();
  return w.bond_price(face, couponRate, maturity, yieldRate, Math.round(frequency));
}

// =OPENFERRIC.CDS_SPREAD(notional, maturity, recovery, hazard_rate, discount_rate)
async function cdsSpread(notional, maturity, recovery, hazardRate, discountRate) {
  const w = await ensureWasm();
  return w.cds_fair_spread(notional, maturity, recovery, hazardRate, discountRate);
}

// =OPENFERRIC.VAR(returns_range, confidence)
async function varHistorical(returns, confidence) {
  const w = await ensureWasm();
  // Flatten Excel range (2D array) to 1D
  const flat = Array.isArray(returns[0])
    ? returns.flat().filter(v => typeof v === 'number')
    : returns.filter(v => typeof v === 'number');
  return w.var_historical(new Float64Array(flat), confidence);
}

// =OPENFERRIC.SABR_VOL(forward, strike, t, alpha, beta, rho, nu)
async function sabrVol(forward, strike, t, alpha, beta, rho, nu) {
  const w = await ensureWasm();
  return w.sabr_vol(forward, strike, t, alpha, beta, rho, nu);
}

// Register with Excel CustomFunctions API
if (typeof CustomFunctions !== 'undefined') {
  CustomFunctions.associate("BS_PRICE", bsPrice);
  CustomFunctions.associate("BS_IMPLIED_VOL", bsImpliedVol);
  CustomFunctions.associate("BS_GREEKS", bsGreeks);
  CustomFunctions.associate("HESTON_PRICE", hestonPrice);
  CustomFunctions.associate("BARRIER_PRICE", barrierPrice);
  CustomFunctions.associate("BOND_PRICE", bondPrice);
  CustomFunctions.associate("CDS_SPREAD", cdsSpread);
  CustomFunctions.associate("VAR", varHistorical);
  CustomFunctions.associate("SABR_VOL", sabrVol);
}

// Export for testing outside Excel
if (typeof module !== 'undefined') {
  module.exports = {
    bsPrice, bsImpliedVol, bsGreeks, hestonPrice,
    barrierPrice, bondPrice, cdsSpread, varHistorical, sabrVol,
    ensureWasm,
  };
}
