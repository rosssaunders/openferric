import * as vscode from "vscode";

export interface PricingResult {
  productName: string;
  notional: number;
  maturity: number;
  underlyings: string[];
  price: number;
  stderr: number | null;
  greeks: GreeksEntry[];
  payoffProfile: PayoffPoint[];
  error: string | null;
}

interface GreeksEntry {
  asset: string;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

interface PayoffPoint {
  spot_pct: number;
  pv: number;
}

export class PricingPanelProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "openferric.pricingPanel";

  private view?: vscode.WebviewView;
  private latestResult?: PricingResult;

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.getHtml();

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
    <div class="card" id="price-card">
      <h2>Price</h2>
      <div class="price-value" id="price-value"></div>
      <div class="price-stderr" id="price-stderr"></div>
      <div class="error-text" id="price-error" style="display:none;"></div>
    </div>
    <div class="card" id="greeks-card" style="display:none;">
      <h2>Greeks</h2>
      <table>
        <thead>
          <tr><th>Asset</th><th>&Delta;</th><th>&Gamma;</th><th>&nu;</th><th>&theta;</th><th>&rho;</th></tr>
        </thead>
        <tbody id="greeks-body"></tbody>
      </table>
    </div>
    <div class="card" id="payoff-card" style="display:none;">
      <h2>Payoff Profile</h2>
      <svg id="payoff-chart" viewBox="0 0 400 160" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
  </div>
  <script>
    const vscode = acquireVsCodeApi();

    function fmt(n, d) {
      if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      return n.toFixed(d || 4);
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

      // Greeks table
      const greeksCard = document.getElementById('greeks-card');
      const greeksBody = document.getElementById('greeks-body');
      if (d.greeks && d.greeks.length > 0) {
        greeksCard.style.display = 'block';
        greeksBody.innerHTML = '';
        for (const g of d.greeks) {
          const tr = document.createElement('tr');
          tr.innerHTML =
            '<td>' + g.asset + '</td>' +
            '<td>' + fmt(g.delta) + '</td>' +
            '<td>' + fmt(g.gamma) + '</td>' +
            '<td>' + fmt(g.vega) + '</td>' +
            '<td>' + fmt(g.theta) + '</td>' +
            '<td>' + fmt(g.rho) + '</td>';
          greeksBody.appendChild(tr);
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
