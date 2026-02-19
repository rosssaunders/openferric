interface OpenFerricWasmModule {
  default: () => Promise<unknown>;
  bs_price: (spot: number, strike: number, rate: number, divYield: number, vol: number, maturity: number, isCall: boolean) => number;
  bs_implied_vol: (price: number, spot: number, strike: number, rate: number, divYield: number, maturity: number, isCall: boolean) => number;
  bsm_greeks_wasm: (spot: number, strike: number, rate: number, divYield: number, vol: number, maturity: number, isCall: boolean) => ArrayLike<number>;
  heston_price: (
    spot: number,
    strike: number,
    rate: number,
    divYield: number,
    v0: number,
    kappa: number,
    theta: number,
    sigmaV: number,
    rho: number,
    maturity: number,
    isCall: boolean
  ) => number;
  barrier_price: (
    spot: number,
    strike: number,
    barrier: number,
    rate: number,
    divYield: number,
    vol: number,
    maturity: number,
    barrierType: string,
    isCall: boolean
  ) => number;
  bond_price: (face: number, couponRate: number, maturity: number, yieldRate: number, frequency: number) => number;
  cds_fair_spread: (notional: number, maturity: number, recovery: number, hazardRate: number, discountRate: number) => number;
  var_historical: (returns: Float64Array, confidence: number) => number;
  sabr_vol: (forward: number, strike: number, t: number, alpha: number, beta: number, rho: number, nu: number) => number;
}

interface CustomFunctionsRuntime {
  associate: <TArgs extends Array<unknown>, TResult>(
    id: string,
    implementation: (...args: TArgs) => TResult
  ) => void;
}

let wasm: OpenFerricWasmModule | undefined;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isWasmModule(value: unknown): value is OpenFerricWasmModule {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value["default"] === "function" &&
    typeof value["bs_price"] === "function" &&
    typeof value["bs_implied_vol"] === "function" &&
    typeof value["bsm_greeks_wasm"] === "function" &&
    typeof value["heston_price"] === "function" &&
    typeof value["barrier_price"] === "function" &&
    typeof value["bond_price"] === "function" &&
    typeof value["cds_fair_spread"] === "function" &&
    typeof value["var_historical"] === "function" &&
    typeof value["sabr_vol"] === "function"
  );
}

async function importWasm(specifier: string): Promise<OpenFerricWasmModule> {
  const candidate: unknown = await import(specifier);
  if (!isWasmModule(candidate)) {
    throw new TypeError(`WASM module at ${specifier} does not match expected OpenFerric interface.`);
  }

  await candidate.default();
  return candidate;
}

async function ensureWasm(): Promise<OpenFerricWasmModule> {
  if (wasm) {
    return wasm;
  }

  try {
    wasm = await importWasm("./pkg/openferric.js");
  } catch {
    wasm = await importWasm("https://rosssaunders.github.io/openferric/pkg/openferric.js");
  }

  return wasm;
}

function toNumericArray(matrixOrArray: unknown): Array<number> {
  if (!Array.isArray(matrixOrArray)) {
    return [];
  }

  const first = matrixOrArray[0];
  if (Array.isArray(first)) {
    return matrixOrArray
      .flatMap((row) => (Array.isArray(row) ? row : []))
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  }

  return matrixOrArray.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function toFixedLabelList(values: ArrayLike<number>): string {
  const labels = ["Δ", "Γ", "V", "Θ", "ρ", "vanna", "volga"];
  return labels.map((label, index) => `${label}=${values[index]?.toFixed(6) ?? "NaN"}`).join(", ");
}

async function bsPrice(
  spot: number,
  strike: number,
  rate: number,
  divYield: number,
  vol: number,
  maturity: number,
  isCall: boolean
): Promise<number> {
  const module = await ensureWasm();
  return module.bs_price(spot, strike, rate, divYield, vol, maturity, isCall);
}

async function bsImpliedVol(
  price: number,
  spot: number,
  strike: number,
  rate: number,
  divYield: number,
  maturity: number,
  isCall: boolean
): Promise<number> {
  const module = await ensureWasm();
  return module.bs_implied_vol(price, spot, strike, rate, divYield, maturity, isCall);
}

async function bsGreeks(
  spot: number,
  strike: number,
  rate: number,
  divYield: number,
  vol: number,
  maturity: number,
  isCall: boolean
): Promise<string> {
  const module = await ensureWasm();
  const greeks = module.bsm_greeks_wasm(spot, strike, rate, divYield, vol, maturity, isCall);
  return toFixedLabelList(greeks);
}

async function hestonPrice(
  spot: number,
  strike: number,
  rate: number,
  divYield: number,
  v0: number,
  kappa: number,
  theta: number,
  sigmaV: number,
  rho: number,
  maturity: number,
  isCall: boolean
): Promise<number> {
  const module = await ensureWasm();
  return module.heston_price(spot, strike, rate, divYield, v0, kappa, theta, sigmaV, rho, maturity, isCall);
}

async function barrierPrice(
  spot: number,
  strike: number,
  barrier: number,
  rate: number,
  divYield: number,
  vol: number,
  maturity: number,
  barrierType: string,
  isCall: boolean
): Promise<number> {
  const module = await ensureWasm();
  return module.barrier_price(spot, strike, barrier, rate, divYield, vol, maturity, barrierType, isCall);
}

async function bondPrice(
  face: number,
  couponRate: number,
  maturity: number,
  yieldRate: number,
  frequency: number
): Promise<number> {
  const module = await ensureWasm();
  return module.bond_price(face, couponRate, maturity, yieldRate, Math.round(frequency));
}

async function cdsSpread(
  notional: number,
  maturity: number,
  recovery: number,
  hazardRate: number,
  discountRate: number
): Promise<number> {
  const module = await ensureWasm();
  return module.cds_fair_spread(notional, maturity, recovery, hazardRate, discountRate);
}

async function varHistorical(returns: unknown, confidence: number): Promise<number> {
  const module = await ensureWasm();
  const flat = toNumericArray(returns);
  return module.var_historical(new Float64Array(flat), confidence);
}

async function sabrVol(
  forward: number,
  strike: number,
  t: number,
  alpha: number,
  beta: number,
  rho: number,
  nu: number
): Promise<number> {
  const module = await ensureWasm();
  return module.sabr_vol(forward, strike, t, alpha, beta, rho, nu);
}

function tryRegisterCustomFunctions(): void {
  const customFunctions = (globalThis as { CustomFunctions?: CustomFunctionsRuntime }).CustomFunctions;
  if (!customFunctions) {
    return;
  }

  customFunctions.associate("BS_PRICE", bsPrice);
  customFunctions.associate("BS_IMPLIED_VOL", bsImpliedVol);
  customFunctions.associate("BS_GREEKS", bsGreeks);
  customFunctions.associate("HESTON_PRICE", hestonPrice);
  customFunctions.associate("BARRIER_PRICE", barrierPrice);
  customFunctions.associate("BOND_PRICE", bondPrice);
  customFunctions.associate("CDS_SPREAD", cdsSpread);
  customFunctions.associate("VAR", varHistorical);
  customFunctions.associate("SABR_VOL", sabrVol);
}

tryRegisterCustomFunctions();
