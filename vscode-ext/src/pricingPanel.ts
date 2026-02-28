import * as vscode from "vscode";

export interface MarketSnapshot {
  rate: number;
  assets: AssetSnapshot[];
  correlation: number[][];
}

export interface AssetSnapshot {
  name: string;
  spot: number;
  vol: number;
  dividendYield: number;
}

export interface PricingResult {
  productName: string;
  notional: number;
  maturity: number;
  underlyings: string[];
  price: number;
  stderr: number | null;
  greeks: GreeksEntry[];
  crossGreeks: CrossGreeksEntry[];
  payoffProfile: PayoffPoint[];
  error: string | null;
  market: MarketSnapshot | null;
}

interface GreeksEntry {
  asset: string;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
  vanna: number;
  volga: number;
}

interface CrossGreeksEntry {
  assetI: string;
  assetJ: string;
  crossGamma: number;
  corrSens: number;
}

interface PayoffPoint {
  spot_pct: number;
  pv: number;
}

export class PricingPanelProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "openferric.pricingPanel";

  private view?: vscode.WebviewView;
  private latestResult?: PricingResult;

  private readonly _onMarketUpdate = new vscode.EventEmitter<MarketSnapshot>();
  public readonly onMarketUpdate = this._onMarketUpdate.event;

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.getHtml();

    webviewView.webview.onDidReceiveMessage((msg) => {
      if (msg.type === "marketUpdate") {
        this._onMarketUpdate.fire(msg.data as MarketSnapshot);
      }
    });

    if (this.latestResult) {
      webviewView.webview.postMessage({
        type: "update",
        data: this.latestResult,
      });
    }
  }

  update(result: PricingResult): void {
    this.latestResult = result;
    if (this.view) {
      this.view.webview.postMessage({ type: "update", data: result });
    }
  }

  private getHtml(): string {
    return /*html*/ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-editor-foreground);
    background: var(--vscode-editor-background);
    padding: 12px;
  }
  .card {
    background: var(--vscode-editorWidget-background, var(--vscode-editor-background));
    border: 1px solid var(--vscode-editorWidget-border, var(--vscode-panel-border));
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 10px;
  }
  .card h2 {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
    color: var(--vscode-editor-foreground);
  }
  .card h3 {
    font-size: 11px;
    font-weight: 600;
    margin: 8px 0 4px;
    color: var(--vscode-descriptionForeground);
  }
  .meta {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin-bottom: 2px;
  }
  .price-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--vscode-charts-green, #4ec9b0);
  }
  .price-stderr {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin-top: 2px;
  }
  .error-text {
    color: var(--vscode-errorForeground, #f44747);
    font-size: 12px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }
  th, td {
    text-align: right;
    padding: 3px 6px;
    border-bottom: 1px solid var(--vscode-editorWidget-border, var(--vscode-panel-border));
  }
  th:first-child, td:first-child { text-align: left; }
  th {
    font-weight: 600;
    color: var(--vscode-descriptionForeground);
  }
  svg {
    width: 100%;
    height: 160px;
    display: block;
  }
  .chart-line {
    fill: none;
    stroke: var(--vscode-charts-blue, #569cd6);
    stroke-width: 2;
  }
  .chart-area {
    fill: var(--vscode-charts-blue, #569cd6);
    opacity: 0.1;
  }
  .chart-grid {
    stroke: var(--vscode-editorWidget-border, var(--vscode-panel-border));
    stroke-width: 0.5;
  }
  .chart-axis-label {
    font-size: 9px;
    fill: var(--vscode-descriptionForeground);
  }
  .chart-marker {
    fill: var(--vscode-charts-orange, #ce9178);
  }
  .chart-zero {
    stroke: var(--vscode-descriptionForeground);
    stroke-width: 0.5;
    stroke-dasharray: 3,3;
  }
  #placeholder {
    text-align: center;
    color: var(--vscode-descriptionForeground);
    padding: 40px 12px;
    font-size: 12px;
  }
  .field-row {
    display: flex;
    align-items: center;
    margin-bottom: 4px;
    gap: 6px;
  }
  .field-row label {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    min-width: 60px;
    flex-shrink: 0;
  }
  .field-row input {
    flex: 1;
    min-width: 0;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 11px;
    font-family: var(--vscode-editor-font-family, monospace);
  }
  .field-row input:focus {
    outline: 1px solid var(--vscode-focusBorder);
    border-color: var(--vscode-focusBorder);
  }
  .corr-grid {
    margin-top: 4px;
  }
  .corr-grid table {
    font-size: 11px;
  }
  .corr-grid td, .corr-grid th {
    text-align: center;
    padding: 2px 4px;
  }
  .corr-grid input {
    width: 50px;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 10px;
    font-family: var(--vscode-editor-font-family, monospace);
    text-align: center;
  }
  .corr-grid input:focus {
    outline: 1px solid var(--vscode-focusBorder);
  }
  .corr-grid .diag {
    color: var(--vscode-descriptionForeground);
    font-size: 10px;
  }
  .greeks-asset {
    font-size: 11px;
    font-weight: 600;
    color: var(--vscode-descriptionForeground);
    margin: 8px 0 4px;
  }
  .greeks-asset:first-child { margin-top: 0; }
  .greeks-pair {
    font-size: 10px;
    color: var(--vscode-descriptionForeground);
    margin: 4px 0 2px;
    padding-left: 4px;
  }
  .greeks-grid {
    display: grid;
    grid-template-columns: auto 1fr auto 1fr;
    gap: 2px 8px;
    font-size: 11px;
  }
  .greeks-grid .gl {
    color: var(--vscode-descriptionForeground);
  }
  .greeks-grid .gv {
    text-align: right;
    font-family: var(--vscode-editor-font-family, monospace);
  }
</style>
</head>
<body>
  <div id="placeholder">Open an <code>.of</code> file to see pricing results.</div>
  <div id="content" style="display:none;">
    <div class="card" id="product-card">
      <h2 id="product-name"></h2>
      <div class="meta" id="product-meta"></div>
      <div class="meta" id="product-underlyings"></div>
    </div>
    <div class="card" id="market-card" style="display:none;">
      <h2>Market Data</h2>
      <div id="market-inputs"></div>
    </div>
    <div class="card" id="price-card">
      <h2>Price</h2>
      <div class="price-value" id="price-value"></div>
      <div class="price-stderr" id="price-stderr"></div>
      <div class="error-text" id="price-error" style="display:none;"></div>
    </div>
    <div class="card" id="greeks-card" style="display:none;">
      <h2>Greeks</h2>
      <div id="greeks-body"></div>
    </div>
    <div class="card" id="payoff-card" style="display:none;">
      <h2>Payoff Profile</h2>
      <svg id="payoff-chart" viewBox="0 0 400 160" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
  </div>
  <script>
    const vscode = acquireVsCodeApi();

    let currentMarket = null;
    let debounceTimer = null;

    function fmt(n, d) {
      if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      return n.toFixed(d || 4);
    }

    function fmtGreek(n) {
      const abs = Math.abs(n);
      if (abs < 0.00005) return '~0';
      if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M';
      if (abs >= 1e3) return (n / 1e3).toFixed(1) + 'K';
      return n.toFixed(4);
    }

    function emitMarketUpdate() {
      if (!currentMarket) return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        // Convert camelCase back to snake_case for the LSP
        const payload = {
          rate: currentMarket.rate,
          assets: currentMarket.assets.map(a => ({
            spot: a.spot,
            vol: a.vol,
            dividend_yield: a.dividendYield,
          })),
          correlation: currentMarket.correlation,
        };
        vscode.postMessage({ type: 'marketUpdate', data: payload });
      }, 300);
    }

    function renderMarket(market) {
      if (!market) {
        document.getElementById('market-card').style.display = 'none';
        return;
      }
      currentMarket = JSON.parse(JSON.stringify(market));
      document.getElementById('market-card').style.display = 'block';
      const container = document.getElementById('market-inputs');
      container.innerHTML = '';

      // Rate input
      const rateRow = document.createElement('div');
      rateRow.className = 'field-row';
      rateRow.innerHTML = '<label>Rate</label>';
      const rateInput = document.createElement('input');
      rateInput.type = 'number';
      rateInput.step = '0.01';
      rateInput.value = market.rate;
      rateInput.addEventListener('input', () => {
        currentMarket.rate = parseFloat(rateInput.value) || 0;
        emitMarketUpdate();
      });
      rateRow.appendChild(rateInput);
      container.appendChild(rateRow);

      // Per-asset sections
      for (let i = 0; i < market.assets.length; i++) {
        const asset = market.assets[i];
        const heading = document.createElement('h3');
        heading.textContent = asset.name;
        container.appendChild(heading);

        const fields = [
          { label: 'Spot', key: 'spot', step: '1', value: asset.spot },
          { label: 'Vol', key: 'vol', step: '0.01', value: asset.vol },
          { label: 'Div Yield', key: 'dividendYield', step: '0.01', value: asset.dividendYield },
        ];
        for (const f of fields) {
          const row = document.createElement('div');
          row.className = 'field-row';
          row.innerHTML = '<label>' + f.label + '</label>';
          const inp = document.createElement('input');
          inp.type = 'number';
          inp.step = f.step;
          inp.value = f.value;
          inp.dataset.assetIdx = String(i);
          inp.dataset.field = f.key;
          inp.addEventListener('input', () => {
            currentMarket.assets[i][f.key] = parseFloat(inp.value) || 0;
            emitMarketUpdate();
          });
          row.appendChild(inp);
          container.appendChild(row);
        }
      }

      // Correlation matrix (only for 2+ assets)
      if (market.assets.length >= 2) {
        const corrHeading = document.createElement('h3');
        corrHeading.textContent = 'Correlation';
        container.appendChild(corrHeading);

        const corrDiv = document.createElement('div');
        corrDiv.className = 'corr-grid';
        const n = market.assets.length;
        let html = '<table><tr><th></th>';
        for (let j = 0; j < n; j++) html += '<th>' + market.assets[j].name + '</th>';
        html += '</tr>';
        for (let i = 0; i < n; i++) {
          html += '<tr><th>' + market.assets[i].name + '</th>';
          for (let j = 0; j < n; j++) {
            if (i === j) {
              html += '<td class="diag">1.00</td>';
            } else if (j > i) {
              html += '<td><input type="number" step="0.1" min="-1" max="1" value="' + market.correlation[i][j] + '" data-row="' + i + '" data-col="' + j + '"></td>';
            } else {
              html += '<td class="diag">' + market.correlation[i][j].toFixed(2) + '</td>';
            }
          }
          html += '</tr>';
        }
        html += '</table>';
        corrDiv.innerHTML = html;
        corrDiv.querySelectorAll('input').forEach(inp => {
          inp.addEventListener('input', () => {
            const r = parseInt(inp.dataset.row);
            const c = parseInt(inp.dataset.col);
            const v = parseFloat(inp.value) || 0;
            currentMarket.correlation[r][c] = v;
            currentMarket.correlation[c][r] = v;
            emitMarketUpdate();
          });
        });
        container.appendChild(corrDiv);
      }
    }

    function renderChart(points) {
      const svg = document.getElementById('payoff-chart');
      if (!points || points.length === 0) return;
      svg.innerHTML = '';

      const pad = { top: 10, right: 15, bottom: 22, left: 50 };
      const w = 400, h = 160;
      const iw = w - pad.left - pad.right;
      const ih = h - pad.top - pad.bottom;

      const xs = points.map(p => p.spot_pct);
      const ys = points.map(p => p.pv);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const yRange = yMax - yMin || 1;

      function sx(v) { return pad.left + ((v - xMin) / (xMax - xMin)) * iw; }
      function sy(v) { return pad.top + (1 - (v - (yMin - yRange * 0.05)) / (yRange * 1.1)) * ih; }

      // Gridlines
      for (let i = 0; i <= 4; i++) {
        const yv = yMin + (yRange * i) / 4;
        const y = sy(yv);
        svg.innerHTML += '<line class="chart-grid" x1="' + pad.left + '" y1="' + y + '" x2="' + (w - pad.right) + '" y2="' + y + '"/>';
        svg.innerHTML += '<text class="chart-axis-label" x="' + (pad.left - 4) + '" y="' + (y + 3) + '" text-anchor="end">' + fmt(yv, 0) + '</text>';
      }

      // X-axis labels
      const xTicks = [50, 75, 100, 125, 150];
      for (const xt of xTicks) {
        if (xt >= xMin && xt <= xMax) {
          svg.innerHTML += '<text class="chart-axis-label" x="' + sx(xt) + '" y="' + (h - 2) + '" text-anchor="middle">' + xt + '%</text>';
        }
      }

      // Zero line if in range
      if (yMin < 0 && yMax > 0) {
        svg.innerHTML += '<line class="chart-zero" x1="' + pad.left + '" y1="' + sy(0) + '" x2="' + (w - pad.right) + '" y2="' + sy(0) + '"/>';
      }

      // Area fill
      let areaD = 'M' + sx(xs[0]) + ',' + sy(ys[0]);
      for (let i = 1; i < points.length; i++) areaD += 'L' + sx(xs[i]) + ',' + sy(ys[i]);
      const baseY = sy(yMin - yRange * 0.05);
      areaD += 'L' + sx(xs[xs.length - 1]) + ',' + baseY + 'L' + sx(xs[0]) + ',' + baseY + 'Z';
      svg.innerHTML += '<path class="chart-area" d="' + areaD + '"/>';

      // Line
      let lineD = 'M' + sx(xs[0]) + ',' + sy(ys[0]);
      for (let i = 1; i < points.length; i++) lineD += 'L' + sx(xs[i]) + ',' + sy(ys[i]);
      svg.innerHTML += '<path class="chart-line" d="' + lineD + '"/>';

      // Current price marker (100%)
      const idx100 = points.findIndex(p => p.spot_pct === 100);
      if (idx100 >= 0) {
        const cx = sx(100), cy = sy(ys[idx100]);
        svg.innerHTML += '<circle class="chart-marker" cx="' + cx + '" cy="' + cy + '" r="4"/>';
        svg.innerHTML += '<line class="chart-zero" x1="' + cx + '" y1="' + pad.top + '" x2="' + cx + '" y2="' + (h - pad.bottom) + '"/>';
      }
    }

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type !== 'update') return;
      const d = msg.data;

      document.getElementById('placeholder').style.display = 'none';
      document.getElementById('content').style.display = 'block';

      document.getElementById('product-name').textContent = d.productName;
      document.getElementById('product-meta').textContent =
        'Notional: ' + fmt(d.notional, 0) + ' | Maturity: ' + d.maturity.toFixed(2) + 'y';
      document.getElementById('product-underlyings').textContent =
        'Underlyings: ' + (d.underlyings.length > 0 ? d.underlyings.join(', ') : 'none');

      if (d.error) {
        document.getElementById('price-value').textContent = '--';
        document.getElementById('price-stderr').textContent = '';
        document.getElementById('price-error').style.display = 'block';
        document.getElementById('price-error').textContent = d.error;
      } else {
        document.getElementById('price-value').textContent = fmt(d.price);
        document.getElementById('price-stderr').textContent = d.stderr != null ? '\u00b1 ' + fmt(d.stderr) : '';
        document.getElementById('price-error').style.display = 'none';
      }

      // Market data inputs — only rebuild if asset count changed
      if (d.market) {
        const assetCount = d.market.assets ? d.market.assets.length : 0;
        const currentCount = currentMarket ? currentMarket.assets.length : -1;
        if (assetCount !== currentCount) {
          renderMarket(d.market);
        }
      }

      // Greeks per-asset grid
      const greeksCard = document.getElementById('greeks-card');
      const greeksBody = document.getElementById('greeks-body');
      if (d.greeks && d.greeks.length > 0) {
        greeksCard.style.display = 'block';
        greeksBody.innerHTML = '';
        for (const g of d.greeks) {
          const heading = document.createElement('div');
          heading.className = 'greeks-asset';
          heading.textContent = g.asset;
          greeksBody.appendChild(heading);
          const grid = document.createElement('div');
          grid.className = 'greeks-grid';
          const pairs = [
            ['\u0394', g.delta],   ['\u0393', g.gamma],
            ['\u03BD', g.vega],    ['\u03B8', g.theta],
            ['\u03C1', g.rho],     ['Vanna', g.vanna],
            ['Volga', g.volga],
          ];
          for (const [label, val] of pairs) {
            grid.innerHTML += '<span class="gl">' + label + '</span><span class="gv">' + fmtGreek(val) + '</span>';
          }
          greeksBody.appendChild(grid);
        }

        // Cross-greeks (only for 2+ assets)
        if (d.crossGreeks && d.crossGreeks.length > 0) {
          const crossHeading = document.createElement('div');
          crossHeading.className = 'greeks-asset';
          crossHeading.textContent = 'Cross Sensitivities';
          greeksBody.appendChild(crossHeading);
          for (const cg of d.crossGreeks) {
            const pairLabel = document.createElement('div');
            pairLabel.className = 'greeks-pair';
            pairLabel.textContent = cg.assetI + ' / ' + cg.assetJ;
            greeksBody.appendChild(pairLabel);
            const grid = document.createElement('div');
            grid.className = 'greeks-grid';
            const pairs = [
              ['X-\u0393', cg.crossGamma], ['\u03C1-Sens', cg.corrSens],
            ];
            for (const [label, val] of pairs) {
              grid.innerHTML += '<span class="gl">' + label + '</span><span class="gv">' + fmtGreek(val) + '</span>';
            }
            greeksBody.appendChild(grid);
          }
        }
      } else {
        greeksCard.style.display = 'none';
      }

      // Payoff chart
      const payoffCard = document.getElementById('payoff-card');
      if (d.payoffProfile && d.payoffProfile.length > 0) {
        payoffCard.style.display = 'block';
        renderChart(d.payoffProfile);
      } else {
        payoffCard.style.display = 'none';
      }
    });
  </script>
</body>
</html>`;
  }
}
