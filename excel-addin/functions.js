"use strict";
let wasm;
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}
function isUnknownArray(value) {
    return Array.isArray(value);
}
function isCustomFunctionsRuntime(value) {
    return isRecord(value) && typeof value["associate"] === "function";
}
function isWasmModule(value) {
    if (!isRecord(value)) {
        return false;
    }
    return (typeof value["default"] === "function" &&
        typeof value["bs_price"] === "function" &&
        typeof value["bs_implied_vol"] === "function" &&
        typeof value["bsm_greeks_wasm"] === "function" &&
        typeof value["heston_price"] === "function" &&
        typeof value["barrier_price"] === "function" &&
        typeof value["bond_price"] === "function" &&
        typeof value["cds_fair_spread"] === "function" &&
        typeof value["var_historical"] === "function" &&
        typeof value["sabr_vol"] === "function");
}
async function importWasm(specifier) {
    const candidate = await import(specifier);
    if (!isWasmModule(candidate)) {
        throw new TypeError(`WASM module at ${specifier} does not match expected OpenFerric interface.`);
    }
    await candidate.default();
    return candidate;
}
async function ensureWasm() {
    if (wasm) {
        return wasm;
    }
    try {
        wasm = await importWasm("./pkg/openferric.js");
    }
    catch {
        wasm = await importWasm("https://rosssaunders.github.io/openferric/pkg/openferric.js");
    }
    return wasm;
}
function toNumericArray(matrixOrArray) {
    if (!Array.isArray(matrixOrArray)) {
        return [];
    }
    const rows = matrixOrArray;
    const first = rows[0];
    if (isUnknownArray(first)) {
        const flattened = [];
        for (const row of rows) {
            if (isUnknownArray(row)) {
                for (const value of row) {
                    flattened.push(value);
                }
            }
        }
        return flattened.filter(isFiniteNumber);
    }
    return rows.filter(isFiniteNumber);
}
function toFixedLabelList(values) {
    const labels = ["Δ", "Γ", "V", "Θ", "ρ", "vanna", "volga"];
    return labels.map((label, index) => `${label}=${values[index]?.toFixed(6) ?? "NaN"}`).join(", ");
}
async function bsPrice(spot, strike, rate, divYield, vol, maturity, isCall) {
    const module = await ensureWasm();
    return module.bs_price(spot, strike, rate, divYield, vol, maturity, isCall);
}
async function bsImpliedVol(price, spot, strike, rate, divYield, maturity, isCall) {
    const module = await ensureWasm();
    return module.bs_implied_vol(price, spot, strike, rate, divYield, maturity, isCall);
}
async function bsGreeks(spot, strike, rate, divYield, vol, maturity, isCall) {
    const module = await ensureWasm();
    const greeks = module.bsm_greeks_wasm(spot, strike, rate, divYield, vol, maturity, isCall);
    return toFixedLabelList(greeks);
}
async function hestonPrice(spot, strike, rate, divYield, v0, kappa, theta, sigmaV, rho, maturity, isCall) {
    const module = await ensureWasm();
    return module.heston_price(spot, strike, rate, divYield, v0, kappa, theta, sigmaV, rho, maturity, isCall);
}
async function barrierPrice(spot, strike, barrier, rate, divYield, vol, maturity, barrierType, isCall) {
    const module = await ensureWasm();
    return module.barrier_price(spot, strike, barrier, rate, divYield, vol, maturity, barrierType, isCall);
}
async function bondPrice(face, couponRate, maturity, yieldRate, frequency) {
    const module = await ensureWasm();
    return module.bond_price(face, couponRate, maturity, yieldRate, Math.round(frequency));
}
async function cdsSpread(notional, maturity, recovery, hazardRate, discountRate) {
    const module = await ensureWasm();
    return module.cds_fair_spread(notional, maturity, recovery, hazardRate, discountRate);
}
async function varHistorical(returns, confidence) {
    const module = await ensureWasm();
    const flat = toNumericArray(returns);
    return module.var_historical(new Float64Array(flat), confidence);
}
async function sabrVol(forward, strike, t, alpha, beta, rho, nu) {
    const module = await ensureWasm();
    return module.sabr_vol(forward, strike, t, alpha, beta, rho, nu);
}
function tryRegisterCustomFunctions() {
    const customFunctionsCandidate = Reflect.get(globalThis, "CustomFunctions");
    if (!isCustomFunctionsRuntime(customFunctionsCandidate)) {
        return;
    }
    customFunctionsCandidate.associate("BS_PRICE", bsPrice);
    customFunctionsCandidate.associate("BS_IMPLIED_VOL", bsImpliedVol);
    customFunctionsCandidate.associate("BS_GREEKS", bsGreeks);
    customFunctionsCandidate.associate("HESTON_PRICE", hestonPrice);
    customFunctionsCandidate.associate("BARRIER_PRICE", barrierPrice);
    customFunctionsCandidate.associate("BOND_PRICE", bondPrice);
    customFunctionsCandidate.associate("CDS_SPREAD", cdsSpread);
    customFunctionsCandidate.associate("VAR", varHistorical);
    customFunctionsCandidate.associate("SABR_VOL", sabrVol);
}
tryRegisterCustomFunctions();
//# sourceMappingURL=functions.js.map