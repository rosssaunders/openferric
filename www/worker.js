// Web Worker — owns WASM instance, runs all calibration + compute off main thread.
// Communicates with main thread via postMessage.

import init, * as wasm from './pkg/openferric.js';

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
//  Calibration — runs the full per-expiry model calibration
// ---------------------------------------------------------------------------
function runCalibration(chainEntries, spotPrice, model) {
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

  const slices = [];
  for (const [expiryCode, quotes] of grouped) {
    if (quotes.length < 5) continue;

    const expiryDate = parseDeribitExpiry(expiryCode);
    if (!expiryDate) continue;
    const T = (expiryDate.getTime() / 1000 - nowSec) / (365.25 * 24 * 3600);
    if (T <= 0) continue;

    const forward = quotes.reduce((s, q) => s + q.underlying_price, 0) / quotes.length;
    if (!isFinite(forward) || forward <= 0) continue;

    const qByKRaw = [];
    for (const q of quotes) {
      const k = Math.log(q.strike / forward);
      qByKRaw.push({ k, q });
    }

    const qByK = qByKRaw.filter(({ k, q }) => {
      if (!q.mark_iv || q.mark_iv <= 0) return false;
      if (q.bid_iv !== undefined && q.bid_iv <= 0 && q.open_interest <= 0) return false;
      if (q.bid_iv > 0 && q.ask_iv > 0) {
        const spread = q.ask_iv - q.bid_iv;
        const mid = (q.ask_iv + q.bid_iv) / 2;
        if (mid > 0 && spread / mid > 0.5) return false;
      }
      if (T < 7/365 && Math.abs(k) > 0.15) return false;
      if (T < 30/365 && Math.abs(k) > 0.5) return false;
      return true;
    });

    if (qByK.length < 3) {
      qByK.length = 0;
      for (const item of qByKRaw) {
        if (item.q.mark_iv > 0) qByK.push(item);
      }
    }

    // ATM vol
    let atmVol = 0;
    let minAbsK = Infinity;
    for (const { k, q } of qByK) {
      if (Math.abs(k) < minAbsK) {
        minAbsK = Math.abs(k);
        atmVol = q.mark_iv;
      }
    }

    // Model-specific calibration
    let params, modelType;

    if (model === 'sabr') {
      modelType = 'sabr';
      const strikes = qByK.map(({ q }) => q.strike);
      const vols = qByK.map(({ q }) => q.mark_iv);
      const strikesArr = new Float64Array(strikes);
      const volsArr = new Float64Array(vols);
      const wp = wasm.fit_sabr_wasm(forward, strikesArr, volsArr, T, 0.5);
      params = { alpha: wp.alpha, beta: wp.beta, rho: wp.rho, nu: wp.nu };
    } else if (model === 'vv') {
      modelType = 'vv';
      const sorted = [...qByK].sort((a, b) => a.k - b.k);
      const n = sorted.length;
      const putIdx = Math.max(0, Math.round(n * 0.15));
      const callIdx = Math.min(n - 1, Math.round(n * 0.85));
      const putVol = sorted[putIdx].q.mark_iv;
      const callVol = sorted[callIdx].q.mark_iv;
      const rr25 = callVol - putVol;
      const bf25 = 0.5 * (callVol + putVol) - atmVol;
      params = { atmVol, rr25, bf25 };
    } else {
      modelType = 'svi';
      const points = [];
      const maxOi = Math.max(1, ...qByK.map(({ q }) => q.open_interest || 0));
      for (const { k, q } of qByK) {
        const iv2 = q.mark_iv * q.mark_iv;
        const oi = q.open_interest || 0;
        const reps = 1 + Math.floor((oi / maxOi) * 4);
        for (let r = 0; r < reps; r++) {
          points.push(k, iv2);
        }
      }
      const atmIv2 = Math.max(atmVol * atmVol, 1e-4);
      const flat = new Float64Array(points);
      const wp = wasm.calibrate_svi_wasm(flat, atmIv2*0.5, atmIv2*1.5, -0.1, 0.0, 0.15, 3000, 0.002);
      params = { a: wp.a * T, b: wp.b * T, rho: wp.rho, m: wp.m, sigma: wp.sigma };
    }

    // Fit diagnostics
    const diagKs = new Float64Array(qByK.map(({ k }) => k));
    const diagIvs = new Float64Array(qByK.map(({ q }) => q.mark_iv * 100));
    const diagStrikes = new Float64Array(qByK.map(({ q }) => q.strike));
    const diagParams = new Float64Array(modelParamArray({ modelType, params }));
    const diag = wasm.slice_fit_diagnostics(
      modelTypeCode(modelType), diagParams, T, forward, diagKs, diagIvs, diagStrikes
    );
    const rmse = diag[0];
    const skew = diag[1];
    const kurtProxy = diag[2];
    const slicePoints = [];
    for (let j = 0; j < qByK.length; j++) {
      const { q } = qByK[j];
      slicePoints.push({
        strike: q.strike,
        market_iv: q.mark_iv * 100,
        fitted_iv: diag[3 + j],
        bid_iv: (q.bid_iv || 0) * 100,
        ask_iv: (q.ask_iv || 0) * 100,
      });
    }
    slicePoints.sort((a, b) => a.strike - b.strike);

    slices.push({
      expiryCode,
      expiryLabel: expiryCode,
      T,
      params,
      modelType,
      rmse,
      n: qByK.length,
      atmVol: atmVol * 100,
      points: slicePoints,
      forward,
      skew,
      kurtProxy,
    });
  }

  slices.sort((a, b) => a.T - b.T);
  return { slices, chain };
}

// ---------------------------------------------------------------------------
//  Compute pipeline — frame cache, surface, greeks, scanner, term
// ---------------------------------------------------------------------------
function computeFrameCache(slices) {
  if (slices.length === 0) return null;
  const packed = packSliceArrays(slices);
  const d25 = wasm.find_25d_strikes_batch(packed.headers, packed.params);
  const atmIv = wasm.iv_grid(packed.headers, packed.params, new Float64Array([0]));
  return { packed, d25, atmIv };
}

function computeSurfaceData(slices, packed) {
  if (slices.length === 0) return null;

  const marketX = [], marketY = [], marketZ = [], marketText = [];
  for (const sl of slices) {
    for (const pt of sl.points) {
      if (pt.strike <= 0 || sl.forward <= 0) continue;
      marketX.push(Math.log(pt.strike / sl.forward));
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
  if (marketX.length > 0) {
    kMin = Math.min(...marketX);
    kMax = Math.max(...marketX);
    if (kMax - kMin < 0.01) { kMin -= 0.25; kMax += 0.25; }
  }

  const gridN = 15;
  const kGrid = [];
  for (let i = 0; i < gridN; i++) kGrid.push(kMin + (kMax - kMin) * i / (gridN - 1));
  const tGrid = slices.map(s => s.T);
  const kGridF64 = new Float64Array(kGrid);
  const flatZ = Array.from(wasm.iv_grid(packed.headers, packed.params, kGridF64));

  return { kGrid, tGrid, flatZ, gridN, marketX, marketY, marketZ, marketText };
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

  // Batch IV lookup
  const gKVals = [], gSliceIdx = [];
  for (let si = 0; si < slices.length; si++) {
    const sl = slices[si];
    for (const strike of sampledStrikes) {
      gKVals.push(Math.log(strike / sl.forward));
      gSliceIdx.push(si);
    }
  }
  const gIvs = wasm.batch_slice_iv(packed.headers, packed.params,
    new Float64Array(gKVals), new Uint32Array(gSliceIdx));

  // Batch greeks via bsm_greeks_batch_wasm
  const bSpots = [], bStrikes = [], bRates = [], bDivs = [], bVols = [], bExpiries = [], bCalls = [];
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
      validMap.push({ batchIdx: bSpots.length, sliceIdx: si, strikeIdx: sti });
      bSpots.push(spotPrice);
      bStrikes.push(strike);
      bRates.push(0.05);
      bDivs.push(0.0);
      bVols.push(iv / 100);
      bExpiries.push(sl.T);
      bCalls.push(1); // call for greeks display
    }
  }

  let greeksFlat = null;
  if (bSpots.length > 0) {
    greeksFlat = wasm.bsm_greeks_batch_wasm(
      new Float64Array(bSpots), new Float64Array(bStrikes),
      new Float64Array(bRates), new Float64Array(bDivs),
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

  // Collect k-values and slice indices for all options
  const preKVals = [], preSliceIdx = [], preItems = [];
  for (const sl of slices) {
    for (const [name, q] of chain) {
      if (q.expiryCode !== sl.expiryCode) continue;
      if (!q.mark_iv || q.mark_iv <= 0 || q.strike <= 0) continue;
      const k = Math.log(q.strike / sl.forward);
      preKVals.push(k);
      preSliceIdx.push(sliceIdxMap.get(sl.expiryCode));
      preItems.push({ name, q, sl, k });
    }
  }

  const scIvs = preKVals.length > 0
    ? wasm.batch_slice_iv(packed.headers, packed.params,
        new Float64Array(preKVals), new Uint32Array(preSliceIdx))
    : [];

  // Collect pricing requests
  const pricingRequests = [];
  const bSpots = [], bStrikes = [], bRates = [], bDivs = [], bVols = [], bMats = [], bCalls = [];

  for (let pi = 0; pi < preItems.length; pi++) {
    const { name, q, sl } = preItems[pi];
    const modelIvPct = scIvs[pi];
    if (!isFinite(modelIvPct) || modelIvPct <= 0) continue;

    const marketIvPct = q.mark_iv * 100;
    const ivEdge = modelIvPct - marketIvPct;
    const modelSigma = modelIvPct / 100;
    const marketMid = q.mark_price * spotPrice;
    const callFlag = q.isCall ? 1 : 0;

    const theoIdx = bSpots.length;
    bSpots.push(spotPrice); bStrikes.push(q.strike); bRates.push(0.05);
    bDivs.push(0.0); bVols.push(modelSigma); bMats.push(sl.T); bCalls.push(callFlag);

    const bidIdx = q.bid_iv > 0 ? bSpots.length : -1;
    if (bidIdx >= 0) {
      bSpots.push(spotPrice); bStrikes.push(q.strike); bRates.push(0.05);
      bDivs.push(0.0); bVols.push(q.bid_iv); bMats.push(sl.T); bCalls.push(callFlag);
    }

    const askIdx = q.ask_iv > 0 ? bSpots.length : -1;
    if (askIdx >= 0) {
      bSpots.push(spotPrice); bStrikes.push(q.strike); bRates.push(0.05);
      bDivs.push(0.0); bVols.push(q.ask_iv); bMats.push(sl.T); bCalls.push(callFlag);
    }

    pricingRequests.push({ name, expiryCode: sl.expiryCode, T: sl.T,
      strike: q.strike, isCall: q.isCall, marketIvPct, modelIvPct, ivEdge,
      marketMid, theoIdx, bidIdx, askIdx });
  }

  // Single batch BS pricing call
  const prices = bSpots.length > 0
    ? wasm.bs_price_batch_wasm(
        new Float64Array(bSpots), new Float64Array(bStrikes),
        new Float64Array(bRates), new Float64Array(bDivs),
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

  return { heatZ, heatText, sampledStrikes, expLabels, maxEdge, edges };
}

function computeRealizedVol(spotLogReturns) {
  if (spotLogReturns.length < 2) return NaN;
  const obsPerYear = 1 * 86400 * 365.25;
  return wasm.realized_vol(new Float64Array(spotLogReturns), obsPerYear);
}

function computeTermData(d25, atmIvFlat, slices) {
  const rr25 = [], bf25 = [];
  for (let i = 0; i < slices.length; i++) {
    const ivCall = d25[i * 4 + 2];
    const ivPut = d25[i * 4 + 3];
    const atmIv = atmIvFlat[i];
    if (isFinite(ivCall) && isFinite(ivPut)) {
      rr25.push(ivCall - ivPut);
      bf25.push(0.5 * (ivCall + ivPut) - atmIv);
    } else {
      rr25.push(NaN);
      bf25.push(NaN);
    }
  }
  return { termRr25: rr25, termBf25: bf25 };
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
    return;
  }

  if (type === 'market-update') {
    if (!wasmReady) return;

    const { chainEntries, spotPrice, spotLogReturns, renderCycle } = payload;
    const t0 = performance.now();

    // 1. Calibrate
    const { slices, chain } = runCalibration(chainEntries, spotPrice, activeModel);
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

    // 3. Realized vol
    const realizedVol = computeRealizedVol(spotLogReturns);

    // 4. Term data (cheap, always computed)
    const { termRr25, termBf25 } = fc
      ? computeTermData(fc.d25, fc.atmIv, slices)
      : { termRr25: [], termBf25: [] };

    // 5. Tiered computation
    let surface = null, greeks = null, scanner = null;

    if (renderCycle % 5 === 0 && fc) {
      surface = computeSurfaceData(slices, fc.packed);
    }

    if (renderCycle % 2 === 0 && fc) {
      greeks = computeGreeksData(slices, fc.packed, spotPrice);
    }

    if (renderCycle % 3 === 0 && fc) {
      scanner = computeScannerData(slices, fc.packed, chain, spotPrice);
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
    }});
  }
};
