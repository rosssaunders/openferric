// Web Worker — owns WASM instance, runs all calibration + compute off main thread.
// Communicates with main thread via postMessage.

import init, * as wasm from './pkg/openferric_wasm.js';

// ---------------------------------------------------------------------------
//  WASM initialization
// ---------------------------------------------------------------------------
let wasmReady = false;

(async () => {
  try {
    await init();
    wasmReady = true;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    console.error('[worker] WASM init failed:', e);
  }
})();

// ---------------------------------------------------------------------------
//  Config state (updated via config-update messages)
// ---------------------------------------------------------------------------
let activeModel = 'svi';
let activeGreek = 'delta';
let edgeThreshold = 1.0;
let edgeSideFilter = 'all';
let isLinear = false;

const SURFACE_PARAM_KEYS = {
  svi: ['a', 'b', 'rho', 'm', 'sigma'],
  sabr: ['alpha', 'beta', 'rho', 'nu'],
  vv: ['atmVol', 'rr25', 'bf25'],
};

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function defaultSurfaceConfig() {
  return {
    asset: 'BTC',
    model: 'svi',
    mode: 'calibrated',
    params: {
      smilePreset: 'neutral',
      common: {
        interpolationDomain: 'log-moneyness',
        wingSlopeLeft: 1.0,
        wingSlopeRight: 1.0,
        volFloor: 5.0,
        volCeiling: 250.0,
        arbitragePenalty: 1.0,
        calendarSmoothing: 0.0,
        optimizerBoundScale: 1.0,
        optimizerTolerance: 0.0005,
      },
      models: {
        svi: {
          variant: 'raw',
          manual: { a: 0.025, b: 0.22, rho: -0.2, m: 0.0, sigma: 0.28 },
          pins: { a: false, b: false, rho: false, m: false, sigma: false },
        },
        sabr: {
          betaMode: 'free',
          shift: 0.0,
          bounds: { rhoMin: -0.999, rhoMax: 0.999, nuMin: 0.01, nuMax: 3.0 },
          manual: { alpha: 0.55, beta: 0.5, rho: -0.2, nu: 1.0 },
          pins: { alpha: false, beta: false, rho: false, nu: false },
        },
        vv: {
          anchorPutDelta: 0.25,
          anchorCallDelta: 0.25,
          dampening: 1.0,
          barrierCorrection: false,
          manual: { atmVol: 0.55, rr25: 0.0, bf25: 0.01 },
          pins: { atmVol: false, rr25: false, bf25: false },
        },
      },
    },
    calibration: {
      fitWeighting: 'mid',
      autoRecalibrate: true,
      freezeMarket: false,
    },
    conventions: {
      stickyRule: 'delta',
    },
    timestamp: Date.now(),
    source: 'default',
  };
}

function sanitizeSurfaceConfig(raw) {
  const defaults = defaultSurfaceConfig();
  if (!raw || typeof raw !== 'object') return defaults;
  const cfg = deepClone(defaults);
  if (typeof raw.asset === 'string') cfg.asset = raw.asset;
  if (raw.model === 'svi' || raw.model === 'sabr' || raw.model === 'vv') cfg.model = raw.model;
  if (raw.mode === 'calibrated' || raw.mode === 'manual' || raw.mode === 'hybrid') cfg.mode = raw.mode;
  if (raw.params && typeof raw.params === 'object') {
    if (raw.params.smilePreset === 'neutral' || raw.params.smilePreset === 'conservative' || raw.params.smilePreset === 'aggressive') {
      cfg.params.smilePreset = raw.params.smilePreset;
    }
    if (raw.params.common && typeof raw.params.common === 'object') {
      Object.assign(cfg.params.common, raw.params.common);
    }
    if (raw.params.models && typeof raw.params.models === 'object') {
      for (const modelName of ['svi', 'sabr', 'vv']) {
        if (raw.params.models[modelName] && typeof raw.params.models[modelName] === 'object') {
          Object.assign(cfg.params.models[modelName], raw.params.models[modelName]);
          if (raw.params.models[modelName].manual && typeof raw.params.models[modelName].manual === 'object') {
            Object.assign(cfg.params.models[modelName].manual, raw.params.models[modelName].manual);
          }
          if (raw.params.models[modelName].pins && typeof raw.params.models[modelName].pins === 'object') {
            Object.assign(cfg.params.models[modelName].pins, raw.params.models[modelName].pins);
          }
          if (raw.params.models[modelName].bounds && typeof raw.params.models[modelName].bounds === 'object') {
            cfg.params.models[modelName].bounds = {
              ...cfg.params.models[modelName].bounds,
              ...raw.params.models[modelName].bounds,
            };
          }
        }
      }
    }
  }
  if (raw.calibration && typeof raw.calibration === 'object') {
    Object.assign(cfg.calibration, raw.calibration);
    if (cfg.calibration.fitWeighting !== 'mid' && cfg.calibration.fitWeighting !== 'vega-weighted' && cfg.calibration.fitWeighting !== 'bid-ask-aware') {
      cfg.calibration.fitWeighting = 'mid';
    }
  }
  if (raw.conventions && typeof raw.conventions === 'object') {
    Object.assign(cfg.conventions, raw.conventions);
    if (cfg.conventions.stickyRule !== 'delta' && cfg.conventions.stickyRule !== 'strike' && cfg.conventions.stickyRule !== 'moneyness') {
      cfg.conventions.stickyRule = 'delta';
    }
  }
  cfg.timestamp = Number.isFinite(raw.timestamp) ? raw.timestamp : Date.now();
  cfg.source = typeof raw.source === 'string' ? raw.source : cfg.source;
  return cfg;
}

let surfaceConfig = defaultSurfaceConfig();

// ---------------------------------------------------------------------------
//  Pure helpers (copied from index.html — no DOM dependency)
// ---------------------------------------------------------------------------
const MONTHS = { JAN:1,FEB:2,MAR:3,APR:4,MAY:5,JUN:6,JUL:7,AUG:8,SEP:9,OCT:10,NOV:11,DEC:12 };

function parseDeribitExpiry(code) {
  if (code.length !== 7) return null;
  const day = parseInt(code.slice(0, 2), 10);
  const mon = MONTHS[code.slice(2, 5)];
  const yr = 2000 + parseInt(code.slice(5, 7), 10);
  if (!mon || isNaN(day) || isNaN(yr)) return null;
  return new Date(Date.UTC(yr, mon - 1, day, 8, 0, 0));
}

function normalizeIv(raw) {
  if (!isFinite(raw) || raw <= 0) return NaN;
  return raw > 3.0 ? raw / 100.0 : raw;
}

const MODEL_SVI = 0, MODEL_SABR = 1, MODEL_VV = 2;

function modelTypeCode(mt) {
  if (mt === 'svi') return MODEL_SVI;
  if (mt === 'sabr') return MODEL_SABR;
  if (mt === 'vv') return MODEL_VV;
  return 255;
}

function modelParamArray(sl) {
  const mt = sl.modelType, p = sl.params;
  if (mt === 'svi') return [p.a, p.b, p.rho, p.m, p.sigma];
  if (mt === 'sabr') return [p.alpha, p.beta, p.rho, p.nu];
  if (mt === 'vv') return [p.atmVol, p.rr25, p.bf25];
  return [];
}

function packSliceArrays(slices) {
  const headerArr = [];
  const paramArr = [];
  for (const sl of slices) {
    const code = modelTypeCode(sl.modelType);
    const offset = paramArr.length;
    headerArr.push(code, sl.T, sl.forward, offset);
    paramArr.push(...modelParamArray(sl));
  }
  return {
    headers: new Float64Array(headerArr),
    params: new Float64Array(paramArr),
  };
}

const pFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

// ---------------------------------------------------------------------------
//  Unpack calibrate_slice_wasm result into JS slice object
// ---------------------------------------------------------------------------
const MODEL_PARAM_COUNT = { 0: 5, 1: 4, 2: 3 }; // SVI:5, SABR:4, VV:3
const MODEL_NAMES = { 0: 'svi', 1: 'sabr', 2: 'vv' };

function unpackSliceResult(packed, expiryCode) {
  if (!packed || packed.length < 1 || packed[0] === 0) return null;

  const nPts = packed[0];
  const mt = packed[1];
  const T = packed[2];
  const forward = packed[3];
  const nParams = MODEL_PARAM_COUNT[mt] || 0;
  const modelType = MODEL_NAMES[mt] || 'svi';

  // Extract params
  let params;
  const p = packed.slice(4, 4 + nParams);
  if (mt === MODEL_SVI) {
    params = { a: p[0], b: p[1], rho: p[2], m: p[3], sigma: p[4] };
  } else if (mt === MODEL_SABR) {
    params = { alpha: p[0], beta: p[1], rho: p[2], nu: p[3] };
  } else {
    params = { atmVol: p[0], rr25: p[1], bf25: p[2] };
  }

  // Diagnostics
  const diagOff = 4 + nParams;
  const rmse = packed[diagOff];
  const atmVol = packed[diagOff + 1];
  const skew = packed[diagOff + 2];
  const kurtProxy = packed[diagOff + 3];

  // Per-point data: 6 values each (strike, market_iv_pct, fitted_iv_pct, bid_iv_pct, ask_iv_pct, k)
  const dataOff = diagOff + 4;
  const points = [];
  for (let i = 0; i < nPts; i++) {
    const base = dataOff + i * 6;
    points.push({
      strike: packed[base],
      market_iv: packed[base + 1],
      fitted_iv: packed[base + 2],
      bid_iv: packed[base + 3],
      ask_iv: packed[base + 4],
      k: packed[base + 5],
    });
  }

  return {
    expiryCode,
    expiryLabel: expiryCode,
    T,
    params,
    modelType,
    rmse,
    n: nPts,
    atmVol,
    points,
    forward,
    skew,
    kurtProxy,
  };
}

function getVolCaps(cfg) {
  const common = cfg?.params?.common || {};
  const floor = clamp(finiteOr(Number(common.volFloor), 5), 0, 500);
  const ceil = clamp(finiteOr(Number(common.volCeiling), 250), floor + 0.01, 500);
  return { floor, ceil };
}

function capIv(ivPct, caps) {
  return clamp(finiteOr(ivPct, caps.floor), caps.floor, caps.ceil);
}

function paramsToArray(modelType, params) {
  if (modelType === 'svi') return [params.a, params.b, params.rho, params.m, params.sigma];
  if (modelType === 'sabr') return [params.alpha, params.beta, params.rho, params.nu];
  return [params.atmVol, params.rr25, params.bf25];
}

function clampModelParams(modelType, params, modelCfg) {
  const out = { ...params };
  if (modelType === 'svi') {
    out.a = Math.max(1e-8, finiteOr(out.a, 0.01));
    out.b = Math.max(1e-8, finiteOr(out.b, 0.1));
    out.rho = clamp(finiteOr(out.rho, -0.1), -0.999, 0.999);
    out.m = clamp(finiteOr(out.m, 0.0), -3.0, 3.0);
    out.sigma = Math.max(1e-5, finiteOr(out.sigma, 0.2));
    return out;
  }
  if (modelType === 'sabr') {
    const bounds = modelCfg?.bounds || {};
    out.alpha = Math.max(1e-8, finiteOr(out.alpha, 0.2));
    out.beta = clamp(finiteOr(out.beta, 0.5), 0.0, 1.0);
    out.rho = clamp(
      finiteOr(out.rho, -0.1),
      finiteOr(Number(bounds.rhoMin), -0.999),
      finiteOr(Number(bounds.rhoMax), 0.999)
    );
    out.nu = clamp(
      finiteOr(out.nu, 0.8),
      Math.max(1e-6, finiteOr(Number(bounds.nuMin), 0.01)),
      Math.max(1e-5, finiteOr(Number(bounds.nuMax), 3.0))
    );
    return out;
  }
  out.atmVol = Math.max(0.01, finiteOr(out.atmVol, 0.5));
  out.rr25 = clamp(finiteOr(out.rr25, 0.0), -2.5, 2.5);
  out.bf25 = clamp(finiteOr(out.bf25, 0.0), -2.5, 2.5);
  return out;
}

function getSmilePresetScale(preset) {
  if (preset === 'conservative') return 0.8;
  if (preset === 'aggressive') return 1.2;
  return 1.0;
}

function mergeModeParams(slice, cfg) {
  const mode = cfg?.mode || 'calibrated';
  const modelCfg = cfg?.params?.models?.[slice.modelType] || {};
  const manual = modelCfg.manual || {};
  const pins = modelCfg.pins || {};
  const keys = SURFACE_PARAM_KEYS[slice.modelType] || [];
  const next = { ...slice.params };
  for (const key of keys) {
    const manualVal = Number(manual[key]);
    const useManual = mode === 'manual' || (mode === 'hybrid' && !!pins[key]);
    if (useManual && Number.isFinite(manualVal)) next[key] = manualVal;
  }
  return next;
}

function applyModelTweaks(slice, params, cfg) {
  const common = cfg?.params?.common || {};
  const modelCfg = cfg?.params?.models?.[slice.modelType] || {};
  const out = { ...params };
  const wingLeft = clamp(finiteOr(Number(common.wingSlopeLeft), 1.0), 0.2, 3.0);
  const wingRight = clamp(finiteOr(Number(common.wingSlopeRight), 1.0), 0.2, 3.0);
  const wingAvg = 0.5 * (wingLeft + wingRight);
  const wingBias = wingRight - wingLeft;
  const presetScale = getSmilePresetScale(cfg?.params?.smilePreset || 'neutral');

  if (slice.modelType === 'svi') {
    out.b *= wingAvg * presetScale;
    out.sigma *= wingAvg * presetScale;
    out.m += wingBias * 0.03;
    out.rho = clamp(out.rho - wingBias * 0.08, -0.999, 0.999);
    if (modelCfg.variant === 'jump-wings') {
      out.b *= 1.08;
      out.sigma *= 1.1;
      out.rho = clamp(out.rho * 1.05, -0.999, 0.999);
    }
    return clampModelParams(slice.modelType, out, modelCfg);
  }

  if (slice.modelType === 'sabr') {
    out.nu *= wingAvg * presetScale;
    out.rho = clamp(out.rho - wingBias * 0.08, -0.999, 0.999);
    if (modelCfg.betaMode === 'fixed' && Number.isFinite(Number(modelCfg.manual?.beta))) {
      out.beta = Number(modelCfg.manual.beta);
    }
    const shift = finiteOr(Number(modelCfg.shift), 0.0);
    if (shift !== 0.0) out.alpha *= 1.0 + clamp(shift / 200000.0, -0.2, 0.2);
    return clampModelParams(slice.modelType, out, modelCfg);
  }

  const dampening = clamp(finiteOr(Number(modelCfg.dampening), 1.0), 0.1, 2.5);
  out.rr25 *= wingAvg * presetScale * dampening;
  out.bf25 *= presetScale * dampening;
  const anchorPut = finiteOr(Number(modelCfg.anchorPutDelta), 0.25);
  const anchorCall = finiteOr(Number(modelCfg.anchorCallDelta), 0.25);
  out.rr25 += (anchorCall - anchorPut) * 0.12;
  if (modelCfg.barrierCorrection) out.bf25 += 0.02;
  return clampModelParams(slice.modelType, out, modelCfg);
}

function scaleSliceVolLevel(slice, scale) {
  if (!Number.isFinite(scale) || scale <= 0) return;
  if (slice.modelType === 'svi') {
    const factor = scale * scale;
    slice.params.a *= factor;
    slice.params.b *= factor;
    return;
  }
  if (slice.modelType === 'sabr') {
    slice.params.alpha *= scale;
    return;
  }
  slice.params.atmVol *= scale;
  slice.params.rr25 *= scale;
  slice.params.bf25 *= scale;
}

function recomputeSliceStats(slice, cfg) {
  const caps = getVolCaps(cfg);
  const modelCode = modelTypeCode(slice.modelType);
  const paramsArr = new Float64Array(paramsToArray(slice.modelType, slice.params));
  const ks = new Float64Array(slice.points.map(pt => pt.k));
  const marketIvs = new Float64Array(slice.points.map(pt => pt.market_iv));
  const strikes = new Float64Array(slice.points.map(pt => pt.strike));
  const diag = wasm.slice_fit_diagnostics(
    modelCode,
    paramsArr,
    slice.T,
    slice.forward,
    ks,
    marketIvs,
    strikes
  );
  if (diag && diag.length >= 3) {
    slice.rmse = diag[0];
    slice.skew = diag[1];
    slice.kurtProxy = diag[2];
    for (let i = 0; i < slice.points.length; i++) {
      const fitted = diag[3 + i];
      if (Number.isFinite(fitted)) slice.points[i].fitted_iv = capIv(fitted, caps);
      slice.points[i].market_iv = capIv(slice.points[i].market_iv, caps);
      if (slice.points[i].bid_iv > 0) slice.points[i].bid_iv = capIv(slice.points[i].bid_iv, caps);
      if (slice.points[i].ask_iv > 0) slice.points[i].ask_iv = capIv(slice.points[i].ask_iv, caps);
    }
  }
  const headers = new Float64Array([modelCode, slice.T, slice.forward, 0]);
  const atmArr = wasm.iv_grid(headers, paramsArr, new Float64Array([0]));
  if (atmArr && atmArr.length > 0 && Number.isFinite(atmArr[0])) {
    slice.atmVol = capIv(atmArr[0], caps);
  }
}

function applySurfaceConfigToSlices(slices, cfg) {
  const output = slices.map(slice => {
    const next = {
      ...slice,
      params: { ...slice.params },
      points: slice.points.map(pt => ({ ...pt })),
    };
    next.params = mergeModeParams(next, cfg);
    next.params = applyModelTweaks(next, next.params, cfg);
    next.params = clampModelParams(next.modelType, next.params, cfg?.params?.models?.[next.modelType]);
    recomputeSliceStats(next, cfg);
    return next;
  });

  const smooth = clamp(finiteOr(Number(cfg?.params?.common?.calendarSmoothing), 0.0), 0.0, 1.0);
  if (smooth > 0 && output.length >= 3) {
    const targets = output.map((slice, idx) => {
      if (idx === 0 || idx === output.length - 1) return slice.atmVol;
      return 0.5 * (output[idx - 1].atmVol + output[idx + 1].atmVol);
    });
    for (let i = 0; i < output.length; i++) {
      const cur = finiteOr(output[i].atmVol, targets[i]);
      if (cur <= 0) continue;
      const blended = cur + (targets[i] - cur) * smooth;
      scaleSliceVolLevel(output[i], blended / cur);
      recomputeSliceStats(output[i], cfg);
    }
  }

  return output;
}

function preprocessCalibrationQuotes(quotes, forward, weighting) {
  const strikes = [];
  const markIvs = [];
  const bidIvs = [];
  const askIvs = [];
  const ois = [];
  const useBidAsk = weighting === 'bid-ask-aware';
  const useVega = weighting === 'vega-weighted';

  for (const q of quotes) {
    const strike = Number(q.strike);
    if (!Number.isFinite(strike) || strike <= 0) continue;
    const mark = Number(q.mark_iv);
    const bid = Number(q.bid_iv) > 0 ? Number(q.bid_iv) : 0;
    const ask = Number(q.ask_iv) > 0 ? Number(q.ask_iv) : 0;
    let chosenMark = mark;
    let oi = Number(q.open_interest) > 0 ? Number(q.open_interest) : 0;

    if (useBidAsk && bid > 0 && ask > 0) {
      chosenMark = 0.5 * (bid + ask);
      const spread = Math.max(0, ask - bid);
      const quality = 1 / (1 + spread * 80);
      oi = Math.max(oi, 1) * quality;
    } else if (useVega && forward > 0) {
      const k = Math.log(strike / forward);
      const atmWeight = Math.max(0.25, 1.75 - Math.abs(k) * 3.5);
      oi = Math.max(oi, 1) * atmWeight;
    }

    strikes.push(strike);
    markIvs.push(chosenMark);
    bidIvs.push(bid);
    askIvs.push(ask);
    ois.push(oi);
  }

  return {
    strikes: new Float64Array(strikes),
    markIvs: new Float64Array(markIvs),
    bidIvs: new Float64Array(bidIvs),
    askIvs: new Float64Array(askIvs),
    ois: new Float64Array(ois),
  };
}

// ---------------------------------------------------------------------------
//  Calibration — all math in WASM via calibrate_slice_wasm
// ---------------------------------------------------------------------------
function runCalibration(chainEntries, spotPrice, model, cfg) {
  const now = Date.now();
  const nowSec = now / 1000;

  // Reconstruct chain as Map
  const chain = new Map(chainEntries);

  // Group by expiry code
  const grouped = new Map();
  for (const [, q] of chain) {
    if (q.mark_iv <= 0 || q.strike <= 0) continue;
    const key = q.expiryCode;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(q);
  }

  const fitWeighting = cfg?.calibration?.fitWeighting || 'mid';
  const mtCode = modelTypeCode(model);
  const slices = [];
  for (const [expiryCode, quotes] of grouped) {
    if (quotes.length < 5) continue;

    const expiryDate = parseDeribitExpiry(expiryCode);
    if (!expiryDate) continue;
    // Year fraction (epoch arithmetic — not quant math)
    const T = (expiryDate.getTime() / 1000 - nowSec) / (365.25 * 24 * 3600);
    if (T <= 0) continue;

    // Forward: simple average of underlying_price (aggregation, not formula)
    const forward = quotes.reduce((s, q) => s + q.underlying_price, 0) / quotes.length;
    if (!isFinite(forward) || forward <= 0) continue;

    const prep = preprocessCalibrationQuotes(quotes, forward, fitWeighting);
    if (prep.strikes.length < 5) continue;

    // Single WASM call — all filtering, log-moneyness, calibration, diagnostics in Rust
    const packed = wasm.calibrate_slice_wasm(
      prep.strikes,
      prep.markIvs,
      prep.bidIvs,
      prep.askIvs,
      prep.ois,
      forward,
      T,
      mtCode
    );
    const slice = unpackSliceResult(packed, expiryCode);
    if (slice) slices.push(slice);
  }

  slices.sort((a, b) => a.T - b.T);
  return { slices: applySurfaceConfigToSlices(slices, cfg), chain };
}

// ---------------------------------------------------------------------------
//  Compute pipeline — frame cache, surface, greeks, scanner, term
// ---------------------------------------------------------------------------
function computeFrameCache(slices) {
  if (slices.length === 0) return null;
  const packed = packSliceArrays(slices);
  // Single WASM call: 25-delta strikes + ATM IV + RR25/BF25
  const termFlat = wasm.term_structure_batch_wasm(packed.headers, packed.params);
  // Unpack: 7 values per slice [kCall, kPut, ivCall%, ivPut%, atmIv%, rr25, bf25]
  const d25 = new Float64Array(slices.length * 4);
  const atmIv = new Float64Array(slices.length);
  const termRr25 = [], termBf25 = [];
  for (let i = 0; i < slices.length; i++) {
    const off = i * 7;
    d25[i * 4]     = termFlat[off];
    d25[i * 4 + 1] = termFlat[off + 1];
    d25[i * 4 + 2] = termFlat[off + 2];
    d25[i * 4 + 3] = termFlat[off + 3];
    atmIv[i]       = termFlat[off + 4];
    termRr25.push(termFlat[off + 5]);
    termBf25.push(termFlat[off + 6]);
  }
  return { packed, d25, atmIv, termRr25, termBf25 };
}

function computeSurfaceData(slices, packed, cfg) {
  if (slices.length === 0) return null;

  const marketK = [], marketStrike = [], marketMoneyness = [];
  const marketY = [], marketZ = [], marketText = [];
  const marketForwards = [], marketExpiries = [];
  for (const sl of slices) {
    for (const pt of sl.points) {
      if (pt.strike <= 0 || sl.forward <= 0) continue;
      marketK.push(pt.k); // pre-computed in WASM (calibrate_slice_wasm)
      marketStrike.push(pt.strike);
      marketMoneyness.push(Math.exp(pt.k));
      marketForwards.push(sl.forward);
      marketExpiries.push(sl.T);
      marketY.push(sl.T);
      marketZ.push(pt.market_iv);
      marketText.push(
        'Strike: $' + pt.strike.toLocaleString() +
        '<br>Maturity: ' + (sl.expiryCode || sl.T.toFixed(4) + 'y') +
        '<br>IV: ' + pt.market_iv.toFixed(2) + '%'
      );
    }
  }

  let kMin = -0.45, kMax = 0.45;
  if (marketK.length > 0) {
    kMin = Math.min(...marketK);
    kMax = Math.max(...marketK);
    if (kMax - kMin < 0.01) { kMin -= 0.25; kMax += 0.25; }
  }

  const gridN = 15;
  const kGrid = [];
  for (let i = 0; i < gridN; i++) kGrid.push(kMin + (kMax - kMin) * i / (gridN - 1));
  const tGrid = slices.map(s => s.T);
  const kGridF64 = new Float64Array(kGrid);
  const caps = getVolCaps(cfg);
  const flatZ = Array.from(wasm.iv_grid(packed.headers, packed.params, kGridF64)).map(v => capIv(v, caps));

  const stickyRule = cfg?.conventions?.stickyRule || 'delta';
  let xGrid = kGrid;
  let marketX = marketK;
  let xAxisTitle = 'ln(K/F)';
  const avgForward = slices.reduce((acc, sl) => acc + sl.forward, 0) / slices.length;

  if (stickyRule === 'strike') {
    xGrid = kGrid.map(k => avgForward * Math.exp(k));
    marketX = marketStrike;
    xAxisTitle = 'Strike';
  } else if (stickyRule === 'moneyness') {
    xGrid = kGrid.map(k => Math.exp(k));
    marketX = marketMoneyness;
    xAxisTitle = 'K/F';
  } else {
    const gForwards = [];
    const gStrikes = [];
    const gRates = [];
    const gVols = [];
    const gTimes = [];
    const gCalls = [];
    for (let si = 0; si < slices.length; si++) {
      const sl = slices[si];
      for (let ki = 0; ki < gridN; ki++) {
        const iv = flatZ[si * gridN + ki];
        gForwards.push(sl.forward);
        gStrikes.push(sl.forward * Math.exp(kGrid[ki]));
        gRates.push(0.05);
        gVols.push(Math.max(0.0001, iv / 100));
        gTimes.push(sl.T);
        gCalls.push(1);
      }
    }
    const gridGreeks = wasm.black76_greeks_batch_wasm(
      new Float64Array(gForwards),
      new Float64Array(gStrikes),
      new Float64Array(gRates),
      new Float64Array(gVols),
      new Float64Array(gTimes),
      new Uint8Array(gCalls)
    );
    const deltaGrid = [];
    for (let ki = 0; ki < gridN; ki++) {
      let sum = 0;
      let n = 0;
      for (let si = 0; si < slices.length; si++) {
        const val = gridGreeks[(si * gridN + ki) * 7];
        if (!Number.isFinite(val)) continue;
        sum += val;
        n++;
      }
      deltaGrid.push(n > 0 ? sum / n : 0.5);
    }
    xGrid = deltaGrid;

    if (marketStrike.length > 0) {
      const mRates = new Float64Array(marketStrike.length).fill(0.05);
      const mVols = new Float64Array(marketZ.map(v => Math.max(0.0001, v / 100)));
      const mCalls = new Uint8Array(marketStrike.length).fill(1);
      const mGreeks = wasm.black76_greeks_batch_wasm(
        new Float64Array(marketForwards),
        new Float64Array(marketStrike),
        mRates,
        mVols,
        new Float64Array(marketExpiries),
        mCalls
      );
      marketX = [];
      for (let i = 0; i < marketStrike.length; i++) marketX.push(mGreeks[i * 7]);
    }
    xAxisTitle = 'Call Delta';
  }

  return { xGrid, xAxisTitle, tGrid, flatZ, gridN, marketX, marketY, marketZ, marketText, stickyRule };
}

function computeGreeksData(slices, packed, spotPrice) {
  if (slices.length === 0 || spotPrice <= 0) return null;

  const expLabels = slices.map(s => s.expiryLabel);
  const allStrikes = new Set();
  for (const sl of slices) {
    for (const pt of sl.points) allStrikes.add(pt.strike);
  }
  const strikes = [...allStrikes].sort((a, b) => a - b);
  const maxS = 20;
  const step = strikes.length > maxS ? Math.ceil(strikes.length / maxS) : 1;
  const sampledStrikes = strikes.filter((_, i) => i % step === 0);

  const greekIdx = { delta: 0, gamma: 1, vega: 2, theta: 3 }[activeGreek] ?? 0;

  // Batch IV lookup — compute log-moneyness in WASM
  const gStrikesRaw = [], gForwardsRaw = [], gSliceIdx = [];
  for (let si = 0; si < slices.length; si++) {
    const sl = slices[si];
    for (const strike of sampledStrikes) {
      gStrikesRaw.push(strike);
      gForwardsRaw.push(sl.forward);
      gSliceIdx.push(si);
    }
  }
  const gKVals = Array.from(wasm.log_moneyness_batch_wasm(
    new Float64Array(gStrikesRaw), new Float64Array(gForwardsRaw)));
  const gIvs = wasm.batch_slice_iv(packed.headers, packed.params,
    new Float64Array(gKVals), new Uint32Array(gSliceIdx));

  // Batch Greeks via black76_greeks_batch_wasm (forward delta convention)
  const bForwards = [], bStrikes = [], bRates = [], bVols = [], bExpiries = [], bCalls = [];
  const validMap = []; // maps batch index -> { sliceIdx, strikeIdx }
  let gIdx = 0;
  for (let si = 0; si < slices.length; si++) {
    const sl = slices[si];
    for (let sti = 0; sti < sampledStrikes.length; sti++) {
      const strike = sampledStrikes[sti];
      const iv = gIvs[gIdx++];
      if (!isFinite(iv) || iv <= 0 || sl.T <= 0) {
        validMap.push(null);
        continue;
      }
      validMap.push({ batchIdx: bForwards.length, sliceIdx: si, strikeIdx: sti });
      bForwards.push(sl.forward);
      bStrikes.push(strike);
      bRates.push(0.05);
      bVols.push(iv / 100);
      bExpiries.push(sl.T);
      bCalls.push(1); // call for greeks display
    }
  }

  let greeksFlat = null;
  if (bForwards.length > 0) {
    greeksFlat = wasm.black76_greeks_batch_wasm(
      new Float64Array(bForwards), new Float64Array(bStrikes),
      new Float64Array(bRates),
      new Float64Array(bVols), new Float64Array(bExpiries),
      new Uint8Array(bCalls)
    );
  }

  // Build z grid
  const z = [];
  let mapIdx = 0;
  for (let si = 0; si < slices.length; si++) {
    const row = [];
    for (let sti = 0; sti < sampledStrikes.length; sti++) {
      const entry = validMap[mapIdx++];
      if (!entry || !greeksFlat) {
        row.push(0);
      } else {
        row.push(greeksFlat[entry.batchIdx * 7 + greekIdx]);
      }
    }
    z.push(row);
  }

  return { z, sampledStrikes, expLabels };
}

function computeScannerData(slices, packed, chain, spotPrice) {
  if (slices.length === 0 || spotPrice <= 0) return null;

  // Build slice index map
  const sliceIdxMap = new Map();
  slices.forEach((sl, i) => sliceIdxMap.set(sl.expiryCode, i));

  // Collect strikes/forwards and slice indices for all options — compute k in WASM
  const scStrikesRaw = [], scForwardsRaw = [], preSliceIdx = [], preItems = [];
  for (const sl of slices) {
    for (const [name, q] of chain) {
      if (q.expiryCode !== sl.expiryCode) continue;
      if (!q.mark_iv || q.mark_iv <= 0 || q.strike <= 0) continue;
      scStrikesRaw.push(q.strike);
      scForwardsRaw.push(sl.forward);
      preSliceIdx.push(sliceIdxMap.get(sl.expiryCode));
      preItems.push({ name, q, sl });
    }
  }

  // Batch log-moneyness in WASM
  const preKVals = scStrikesRaw.length > 0
    ? Array.from(wasm.log_moneyness_batch_wasm(
        new Float64Array(scStrikesRaw), new Float64Array(scForwardsRaw)))
    : [];

  const scIvs = preKVals.length > 0
    ? wasm.batch_slice_iv(packed.headers, packed.params,
        new Float64Array(preKVals), new Uint32Array(preSliceIdx))
    : [];

  // Collect pricing requests
  const pricingRequests = [];
  const bForwards = [], bStrikes = [], bRates = [], bVols = [], bMats = [], bCalls = [];

  for (let pi = 0; pi < preItems.length; pi++) {
    const { name, q, sl } = preItems[pi];
    const modelIvPct = scIvs[pi];
    if (!isFinite(modelIvPct) || modelIvPct <= 0) continue;

    const marketIvPct = q.mark_iv * 100;
    const ivEdge = modelIvPct - marketIvPct;
    const modelSigma = modelIvPct / 100;
    const marketMid = isLinear ? q.mark_price : q.mark_price * spotPrice;
    const callFlag = q.isCall ? 1 : 0;

    const theoIdx = bForwards.length;
    bForwards.push(sl.forward); bStrikes.push(q.strike); bRates.push(0.05);
    bVols.push(modelSigma); bMats.push(sl.T); bCalls.push(callFlag);

    const bidIdx = q.bid_iv > 0 ? bForwards.length : -1;
    if (bidIdx >= 0) {
      bForwards.push(sl.forward); bStrikes.push(q.strike); bRates.push(0.05);
      bVols.push(q.bid_iv); bMats.push(sl.T); bCalls.push(callFlag);
    }

    const askIdx = q.ask_iv > 0 ? bForwards.length : -1;
    if (askIdx >= 0) {
      bForwards.push(sl.forward); bStrikes.push(q.strike); bRates.push(0.05);
      bVols.push(q.ask_iv); bMats.push(sl.T); bCalls.push(callFlag);
    }

    pricingRequests.push({ name, expiryCode: sl.expiryCode, T: sl.T,
      strike: q.strike, isCall: q.isCall, marketIvPct, modelIvPct, ivEdge,
      marketMid, theoIdx, bidIdx, askIdx });
  }

  // Single batch Black-76 pricing call
  const prices = bForwards.length > 0
    ? wasm.black76_price_batch_wasm(
        new Float64Array(bForwards), new Float64Array(bStrikes),
        new Float64Array(bRates),
        new Float64Array(bVols), new Float64Array(bMats),
        new Uint8Array(bCalls))
    : [];

  // Unpack edges
  const edges = [];
  for (const req of pricingRequests) {
    const theoPrice = prices[req.theoIdx];
    const bidPrice = req.bidIdx >= 0 ? prices[req.bidIdx] : NaN;
    const askPrice = req.askIdx >= 0 ? prices[req.askIdx] : NaN;
    const dollarEdge = theoPrice - req.marketMid;

    edges.push({
      name: req.name,
      expiryCode: req.expiryCode,
      T: req.T,
      strike: req.strike,
      isCall: req.isCall,
      marketIvPct: req.marketIvPct,
      modelIvPct: req.modelIvPct,
      ivEdge: req.ivEdge,
      bidPrice,
      askPrice,
      marketMid: req.marketMid,
      theoPrice,
      dollarEdge,
      absIvEdge: Math.abs(req.ivEdge),
      buySignal: isFinite(askPrice) && theoPrice > askPrice,
      sellSignal: isFinite(bidPrice) && theoPrice < bidPrice,
    });
  }

  // Build heatmap data
  const expLabels = slices.map(s => s.expiryLabel);
  const strikeSet = new Set();
  for (const e of edges) strikeSet.add(e.strike);
  const allStrikes = [...strikeSet].sort((a, b) => a - b);
  const maxStrikes = 35;
  const step = allStrikes.length > maxStrikes ? Math.ceil(allStrikes.length / maxStrikes) : 1;
  const sampledStrikes = allStrikes.filter((_, i) => i % step === 0);

  const edgeLookup = new Map();
  for (const e of edges) {
    const key = e.expiryCode + '|' + e.strike;
    const existing = edgeLookup.get(key);
    if (!existing || e.absIvEdge > Math.abs(existing.ivEdge)) {
      edgeLookup.set(key, e);
    }
  }

  const heatZ = [];
  const heatText = [];
  for (const sl of slices) {
    const row = [];
    const textRow = [];
    for (const strike of sampledStrikes) {
      const e = edgeLookup.get(sl.expiryCode + '|' + strike);
      if (e) {
        row.push(e.ivEdge);
        textRow.push(
          `${e.isCall ? 'C' : 'P'} ${sl.expiryCode} $${pFmt.format(strike)}`
          + `<br>Mkt IV: ${e.marketIvPct.toFixed(1)}%`
          + `<br>Mdl IV: ${e.modelIvPct.toFixed(1)}%`
          + `<br>Edge: ${e.ivEdge > 0 ? '+' : ''}${e.ivEdge.toFixed(2)} vol`
          + `<br>Theo: $${pFmt.format(e.theoPrice)}`
          + `<br>Mid: $${pFmt.format(e.marketMid)}`
          + (e.buySignal ? '<br><b>BUY (theo > ask)</b>' : '')
          + (e.sellSignal ? '<br><b>SELL (theo < bid)</b>' : '')
        );
      } else {
        row.push(0);
        textRow.push('');
      }
    }
    heatZ.push(row);
    heatText.push(textRow);
  }

  const maxEdge = Math.max(3, ...edges.map(e => e.absIvEdge));
  const sliceTValues = slices.map(s => s.T);

  return { heatZ, heatText, sampledStrikes, expLabels, maxEdge, edges, sliceTValues };
}

function computeRealizedVol(spotHistory) {
  if (!spotHistory || spotHistory.length < 3) return NaN;
  // Compute log returns in WASM, then realized vol in WASM
  const logReturns = wasm.log_returns_batch_wasm(new Float64Array(spotHistory));
  if (logReturns.length < 2) return NaN;
  const obsPerYear = 1 * 86400 * 365.25;
  return wasm.realized_vol(logReturns, obsPerYear);
}

function computeForwardVolData(slices, packed) {
  if (slices.length < 2) return null;
  const kPoints = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2];
  const fvFlat = wasm.forward_vol_grid_wasm(packed.headers, packed.params, new Float64Array(kPoints));
  const nK = kPoints.length;
  const stride = 2 + nK;
  const labels = [], spotVols = [], fwdVols = [], skewTraces = [];
  for (let i = 0; i < slices.length - 1; i++) {
    const off = i * stride;
    const spotV = fvFlat[off], fwdV = fvFlat[off + 1];
    if (!isFinite(spotV)) continue;
    labels.push(slices[i].expiryLabel + '\u2192' + slices[i + 1].expiryLabel);
    spotVols.push(spotV);
    fwdVols.push(fwdV);
    const fwdSkew = [];
    for (let ki = 0; ki < nK; ki++) fwdSkew.push(fvFlat[off + 2 + ki]);
    skewTraces.push({ label: labels[labels.length - 1], kPoints, fwdSkew });
  }
  return { labels, spotVols, fwdVols, skewTraces, kPoints };
}

// ---------------------------------------------------------------------------
//  Message handler
// ---------------------------------------------------------------------------
self.onmessage = function(e) {
  const { type, payload } = e.data;

  if (type === 'config-update') {
    if (payload.activeModel !== undefined) activeModel = payload.activeModel;
    if (payload.activeGreek !== undefined) activeGreek = payload.activeGreek;
    if (payload.edgeThreshold !== undefined) edgeThreshold = payload.edgeThreshold;
    if (payload.edgeSideFilter !== undefined) edgeSideFilter = payload.edgeSideFilter;
    if (payload.isLinear !== undefined) isLinear = payload.isLinear;
    if (payload.surfaceConfig !== undefined) surfaceConfig = sanitizeSurfaceConfig(payload.surfaceConfig);
    return;
  }

  if (type === 'strategy-compute') {
    if (!wasmReady) return;
    const { legs, spotPrice, spotShock, volShock, timeShock } = payload;
    if (!legs || legs.length === 0 || spotPrice <= 0) return;
    const nSpots = 100;
    const spotLow = spotPrice * 0.7, spotHigh = spotPrice * 1.3;
    const spotAxis = [];
    for (let i = 0; i < nSpots; i++) spotAxis.push(spotLow + (spotHigh - spotLow) * i / (nSpots - 1));
    const eFw = [], eK = [], eR = [], eV = [], eT = [], eC = [];
    for (const leg of legs) {
      eFw.push(spotPrice); eK.push(leg.strike); eR.push(0.05);
      eV.push(leg.iv); eT.push(leg.T); eC.push(leg.isCall ? 1 : 0);
    }
    const entryPrices = wasm.black76_price_batch_wasm(
      new Float64Array(eFw), new Float64Array(eK), new Float64Array(eR),
      new Float64Array(eV), new Float64Array(eT), new Uint8Array(eC));
    let totalCost = 0;
    for (let i = 0; i < legs.length; i++) totalCost += legs[i].quantity * entryPrices[i];
    const pnlAtExpiry = Array.from(wasm.strategy_intrinsic_pnl_wasm(
      new Float64Array(spotAxis),
      new Float64Array(legs.map(l => l.strike)),
      new Float64Array(legs.map(l => l.quantity)),
      new Uint8Array(legs.map(l => l.isCall ? 1 : 0)),
      totalCost));
    let pnlBeforeExpiry = null;
    if (spotShock !== 0 || volShock !== 0 || timeShock !== 0) {
      const bFw = [], bK = [], bR = [], bV = [], bT = [], bC = [];
      for (const s of spotAxis) {
        for (const leg of legs) {
          bFw.push(s); bK.push(leg.strike); bR.push(0.05);
          bV.push(Math.max(0.01, leg.iv + volShock)); bT.push(Math.max(1/365, leg.T + timeShock));
          bC.push(leg.isCall ? 1 : 0);
        }
      }
      const bp = wasm.black76_price_batch_wasm(
        new Float64Array(bFw), new Float64Array(bK), new Float64Array(bR),
        new Float64Array(bV), new Float64Array(bT), new Uint8Array(bC));
      pnlBeforeExpiry = spotAxis.map((_, si) => {
        let pnl = 0;
        for (let li = 0; li < legs.length; li++) pnl += legs[li].quantity * bp[si * legs.length + li];
        return pnl - totalCost;
      });
    }
    const gFw = [], gK = [], gR = [], gV = [], gT = [], gC = [];
    for (const leg of legs) {
      gFw.push(spotPrice); gK.push(leg.strike); gR.push(0.05);
      gV.push(leg.iv); gT.push(leg.T); gC.push(leg.isCall ? 1 : 0);
    }
    const greeks = wasm.black76_greeks_batch_wasm(
      new Float64Array(gFw), new Float64Array(gK), new Float64Array(gR),
      new Float64Array(gV), new Float64Array(gT), new Uint8Array(gC));
    const netGreeks = [0, 0, 0, 0];
    for (let i = 0; i < legs.length; i++) {
      netGreeks[0] += legs[i].quantity * greeks[i * 7 + 0];
      netGreeks[1] += legs[i].quantity * greeks[i * 7 + 1];
      netGreeks[2] += legs[i].quantity * greeks[i * 7 + 2];
      netGreeks[3] += legs[i].quantity * greeks[i * 7 + 3];
    }
    self.postMessage({ type: 'strategy-result', payload: { spotAxis, pnlAtExpiry, pnlBeforeExpiry, netGreeks, totalCost } });
    return;
  }

  if (type === 'market-update') {
    if (!wasmReady) return;

    const { chainEntries, spotPrice, spotHistory, renderCycle } = payload;
    const t0 = performance.now();

    // 1. Calibrate
    const modelForRun = surfaceConfig?.model || activeModel;
    const { slices, chain } = runCalibration(chainEntries, spotPrice, modelForRun, surfaceConfig);
    const calibTimeUs = Math.round((performance.now() - t0) * 1000);

    if (slices.length === 0) {
      self.postMessage({ type: 'compute-result', payload: {
        calibratedSlices: slices, calibTimeUs,
        realizedVol: NaN,
        d25: null, atmIv: null,
        termRr25: [], termBf25: [],
        surface: null, greeks: null, scanner: null,
      }});
      return;
    }

    // 2. Frame cache (always needed)
    const fc = computeFrameCache(slices);
    const d25 = fc ? Array.from(fc.d25) : null;
    const atmIv = fc ? Array.from(fc.atmIv) : null;

    // 3. Realized vol (log returns computed in WASM from raw prices)
    const realizedVol = computeRealizedVol(spotHistory);

    // 4. Term data (computed in frame cache WASM call)
    const termRr25 = fc ? fc.termRr25 : [];
    const termBf25 = fc ? fc.termBf25 : [];

    // 5. Tiered computation (compute everything on first 5 cycles for fast initial load)
    let surface = null, greeks = null, scanner = null;
    const warmup = renderCycle <= 5;

    if ((warmup || renderCycle % 5 === 0) && fc) {
      surface = computeSurfaceData(slices, fc.packed, surfaceConfig);
    }

    if ((warmup || renderCycle % 2 === 0) && fc) {
      greeks = computeGreeksData(slices, fc.packed, spotPrice);
    }

    if ((warmup || renderCycle % 3 === 0) && fc) {
      scanner = computeScannerData(slices, fc.packed, chain, spotPrice);
    }

    let forwardVol = null;
    if ((warmup || renderCycle % 3 === 0) && fc) {
      forwardVol = computeForwardVolData(slices, fc.packed);
    }

    // 6. Post result
    self.postMessage({ type: 'compute-result', payload: {
      calibratedSlices: slices,
      calibTimeUs,
      realizedVol,
      d25,
      atmIv,
      termRr25,
      termBf25,
      surface,
      greeks,
      scanner,
      forwardVol,
    }});
  }
};
