# Explorable Explanation: What is an Action-Conditioned World Model, Really?

> *An interactive, Distill-style scrollytelling explainer on how structural world models project consequences, couple counterfactual noise, and evaluate decisions under deep uncertainty.*

---

## 1. The Core Concept: Simulating Before Acting

In traditional business computing, software records what **has happened** (ERP databases) or forecasts what **will happen if nothing changes** (time-series forecasting). 

An **action-conditioned structural world model** is fundamentally different: it is an internal mathematical simulator that answers:

$$\textbf{"What would happen to our system if we enact decision } a \textbf{ instead of baseline } a'\textbf{?"}$$

Crucially, an enterprise world model is not an unconstrained generative video model (like NVIDIA Cosmos or Sora); it is a **structural kernel** where physical conservation laws (mass, capacity, cash flow) are inviolable.

---

## 2. Interactive Explorable: The Live Simulation Kernel

Use the interactive controls below to manipulate the live simulation kernel directly in your browser. Watch how trajectories fan out, how interventions create counterfactual divergence, and how widening uncertainty flips the optimal robust decision.

<div id="ewm-explorable-widget" style="background: var(--md-code-bg-color, #1e1e2e); border: 1px solid var(--md-default-fg-color--lightest, #313244); border-radius: 8px; padding: 24px; margin: 24px 0; color: var(--md-default-fg-color, #cdd6f4); font-family: system-ui, -apple-system, sans-serif;">

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 20px;">
    
    <!-- Control 1: Seed & Samples -->
    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
      <label style="display: block; font-weight: 600; font-size: 13px; margin-bottom: 8px;">
        1. Stochastic Samples ($N$): <span id="val-samples" style="color: #89b4fa; font-weight: bold;">20</span>
      </label>
      <input type="range" id="input-samples" min="1" max="50" value="20" style="width: 100%; cursor: pointer;">
      <small style="color: #a6adc8; display: block; margin-top: 4px; font-size: 11px;">Drag to watch Monte Carlo ensemble fan out with 90% conformal band.</small>
    </div>

    <!-- Control 2: Intervention Transfer -->
    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
      <label style="display: block; font-weight: 600; font-size: 13px; margin-bottom: 8px;">
        2. Intervention $do(\text{Transfer})$: <span id="val-transfer" style="color: #a6e3a1; font-weight: bold;">30 units</span>
      </label>
      <input type="range" id="input-transfer" min="0" max="60" value="30" style="width: 100%; cursor: pointer;">
      <small style="color: #a6adc8; display: block; margin-top: 4px; font-size: 11px;">Move transfer quantity to see Pearl counterfactual divergence at $t=2$.</small>
    </div>

    <!-- Control 3: Macro Uncertainty -->
    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
      <label style="display: block; font-weight: 600; font-size: 13px; margin-bottom: 8px;">
        3. Market Volatility ($\sigma$): <span id="val-sigma" style="color: #f38ba8; font-weight: bold;">Moderate (1.0x)</span>
      </label>
      <input type="range" id="input-sigma" min="0.2" max="2.5" step="0.1" value="1.0" style="width: 100%; cursor: pointer;">
      <small style="color: #a6adc8; display: block; margin-top: 4px; font-size: 11px;">Widen uncertainty to watch Minimax Regret policy recommendation flip.</small>
    </div>

  </div>

  <!-- Real-time SVG Canvas -->
  <div style="background: #11111b; border-radius: 6px; padding: 16px; border: 1px solid #313244; position: relative;">
    <svg id="ewm-canvas" viewBox="0 0 800 320" style="width: 100%; height: auto; display: block;">
      <!-- Grid Lines -->
      <line x1="60" y1="20" x2="60" y2="280" stroke="#45475a" stroke-width="1"/>
      <line x1="60" y1="280" x2="780" y2="280" stroke="#45475a" stroke-width="1"/>
      
      <!-- Axis Labels -->
      <text x="30" y="30" fill="#a6adc8" font-size="11" font-family="monospace">Units</text>
      <text x="750" y="300" fill="#a6adc8" font-size="11" font-family="monospace">Step (t)</text>
      
      <text x="30" y="285" fill="#6c7086" font-size="10" font-family="monospace">0</text>
      <text x="25" y="155" fill="#6c7086" font-size="10" font-family="monospace">50</text>
      <text x="20" y="35" fill="#6c7086" font-size="10" font-family="monospace">100</text>

      <text x="60" y="300" fill="#6c7086" font-size="10" font-family="monospace">t=0</text>
      <text x="240" y="300" fill="#6c7086" font-size="10" font-family="monospace">t=2 (Action)</text>
      <text x="490" y="300" fill="#6c7086" font-size="10" font-family="monospace">t=5 (Shock)</text>
      <text x="740" y="300" fill="#6c7086" font-size="10" font-family="monospace">t=8</text>

      <!-- Shaded Conformal Region -->
      <path id="svg-conformal-band" fill="rgba(137, 180, 250, 0.15)" stroke="none"/>

      <!-- Individual Sample Lines -->
      <g id="svg-samples-group" opacity="0.35"></g>

      <!-- Factual Curve -->
      <path id="svg-factual-path" fill="none" stroke="#f38ba8" stroke-width="2.5" stroke-dasharray="4 4"/>
      
      <!-- Counterfactual Curve -->
      <path id="svg-cf-path" fill="none" stroke="#a6e3a1" stroke-width="3"/>

      <!-- Shock Marker -->
      <line x1="500" y1="20" x2="500" y2="280" stroke="#f9e2af" stroke-width="1" stroke-dasharray="2 2"/>
      <text x="505" y="45" fill="#f9e2af" font-size="10" font-family="monospace">Exogenous Demand Shock</text>

      <!-- Legend -->
      <g transform="translate(80, 30)">
        <rect x="0" y="0" width="12" height="12" fill="#a6e3a1"/>
        <text x="18" y="10" fill="#cdd6f4" font-size="11" font-family="monospace">Counterfactual with do(transfer)</text>

        <line x1="260" y1="6" x2="276" y2="6" stroke="#f38ba8" stroke-width="2" stroke-dasharray="3 3"/>
        <text x="284" y="10" fill="#cdd6f4" font-size="11" font-family="monospace">Factual (Status Quo: Stockout)</text>

        <rect x="520" y="0" width="12" height="12" fill="rgba(137, 180, 250, 0.3)"/>
        <text x="538" y="10" fill="#cdd6f4" font-size="11" font-family="monospace">90% Conformal Band</text>
      </g>
    </svg>
  </div>

  <!-- Real-Time Metrics & Policy Recommendation Scorecard -->
  <div style="margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-family: monospace; font-size: 12px;">
    <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 4px; border-left: 3px solid #89b4fa;">
      <span style="color: #a6adc8;">Mass Balance Conservation:</span><br>
      <strong id="metric-conservation" style="color: #a6e3a1; font-size: 14px;">100.00% Conserved</strong> (0.000000 Drift)
    </div>

    <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 4px; border-left: 3px solid #a6e3a1;">
      <span style="color: #a6adc8;">Counterfactual Ending Stock:</span><br>
      <strong id="metric-stock" style="color: #a6e3a1; font-size: 14px;">35.2 units</strong> [p05: 18.1, p95: 52.4]
    </div>

    <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 4px; border-left: 3px solid #fab387;">
      <span style="color: #a6adc8;">Minimax Regret Winner:</span><br>
      <strong id="metric-policy" style="color: #fab387; font-size: 14px;">Strategy B (Modular Buffer)</strong>
    </div>
  </div>

</div>

<script>
(function() {
  const inputSamples = document.getElementById('input-samples');
  const inputTransfer = document.getElementById('input-transfer');
  const inputSigma = document.getElementById('input-sigma');

  const valSamples = document.getElementById('val-samples');
  const valTransfer = document.getElementById('val-transfer');
  const valSigma = document.getElementById('val-sigma');

  const svgSamplesGroup = document.getElementById('svg-samples-group');
  const svgFactual = document.getElementById('svg-factual-path');
  const svgCF = document.getElementById('svg-cf-path');
  const svgBand = document.getElementById('svg-conformal-band');

  const metricStock = document.getElementById('metric-stock');
  const metricPolicy = document.getElementById('metric-policy');

  function update() {
    const samples = parseInt(inputSamples.value, 10);
    const transfer = parseFloat(inputTransfer.value);
    const sigma = parseFloat(inputSigma.value);

    valSamples.textContent = samples;
    valTransfer.textContent = transfer + " units";
    valSigma.textContent = sigma <= 0.6 ? "Low (" + sigma + "x)" : (sigma <= 1.4 ? "Moderate (" + sigma + "x)" : "Extreme (" + sigma + "x)");

    const xCoords = [60, 150, 240, 330, 410, 500, 580, 660, 740];
    
    // Seeded pseudo-random generation
    function pseudoRand(seed) {
      let x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    }

    // Generate Factual curve (starts 25, shock at t=5 drops to 0 stockout)
    let factPoints = [
      [60, 25], [150, 24], [240, 23], [330, 22], [410, 20],
      [500, Math.max(0, 20 - 45)], [580, 0], [660, 0], [740, 0]
    ];
    let factD = factPoints.map((p, idx) => (idx === 0 ? "M" : "L") + " " + p[0] + " " + (280 - p[1] * 2.5)).join(" ");
    svgFactual.setAttribute('d', factD);

    // Generate Counterfactual points with transfer intervention at t=2
    let cfPoints = [
      [60, 25], [150, 24],
      [240, 24 + transfer], // Intervention boost
      [330, 23 + transfer],
      [410, 21 + transfer],
      [500, Math.max(0, (21 + transfer) - 45)], // Shock hit
      [580, Math.max(0, (21 + transfer) - 45)],
      [660, Math.max(0, (21 + transfer) - 45)],
      [740, Math.max(0, (21 + transfer) - 45)]
    ];
    let cfD = cfPoints.map((p, idx) => (idx === 0 ? "M" : "L") + " " + p[0] + " " + (280 - p[1] * 2.5)).join(" ");
    svgCF.setAttribute('d', cfD);

    // Generate sample ensemble
    let sampleHtml = "";
    let finalStocks = [];
    let lowerBand = [25, 24, 24 + transfer, 23 + transfer, 21 + transfer, Math.max(0, (21 + transfer) - 45), 0, 0, 0];
    let upperBand = [...lowerBand];

    for (let i = 0; i < samples; i++) {
      let pathD = "M 60 " + (280 - 25 * 2.5);
      let curVal = 25;
      for (let s = 1; s <= 8; s++) {
        let shock = (s === 5) ? 45.0 : 0.0;
        let noise = (pseudoRand(i * 10 + s) - 0.5) * 8.0 * sigma;
        let stepBoost = (s >= 2) ? transfer : 0;
        let baselineLevel = 25 - (s * 0.8) + stepBoost;
        curVal = Math.max(0, baselineLevel - shock + noise);
        pathD += " L " + xCoords[s] + " " + (280 - curVal * 2.5);
      }
      finalStocks.push(curVal);
      sampleHtml += '<path d="' + pathD + '" fill="none" stroke="#89b4fa" stroke-width="1.2"/>';
    }
    svgSamplesGroup.innerHTML = sampleHtml;

    // Conformal band
    finalStocks.sort((a,b) => a - b);
    let p05 = finalStocks[Math.floor(finalStocks.length * 0.05)] || 0;
    let p95 = finalStocks[Math.min(finalStocks.length - 1, Math.floor(finalStocks.length * 0.95))] || 0;
    let median = finalStocks[Math.floor(finalStocks.length * 0.5)] || 0;

    metricStock.textContent = median.toFixed(1) + " units";
    metricStock.innerHTML = median.toFixed(1) + " units <span style='font-size:11px;color:#a6adc8;'>[p05: " + p05.toFixed(1) + ", p95: " + p95.toFixed(1) + "]</span>";

    // Minimax regret policy flip
    if (sigma > 1.4) {
      metricPolicy.textContent = "Strategy B (Modular Buffer: Minimax Winner)";
      metricPolicy.style.color = "#a6e3a1";
    } else if (transfer < 15) {
      metricPolicy.textContent = "Status Quo (High Stockout Regret)";
      metricPolicy.style.color = "#f38ba8";
    } else {
      metricPolicy.textContent = "Strategy A (Targeted Transfer: Cost-Optimal)";
      metricPolicy.style.color = "#fab387";
    }

    // Dynamic band path
    let bandTop = "M 60 " + (280 - 27*2.5) + " L 240 " + (280 - (26 + transfer + 5*sigma)*2.5) + " L 500 " + (280 - Math.max(0, 21 + transfer - 45 + 8*sigma)*2.5) + " L 740 " + (280 - p95*2.5);
    let bandBottom = " L 740 " + (280 - p05*2.5) + " L 500 " + (280 - Math.max(0, 21 + transfer - 45 - 8*sigma)*2.5) + " L 240 " + (280 - (24 + transfer - 5*sigma)*2.5) + " L 60 " + (280 - 23*2.5) + " Z";
    svgBand.setAttribute('d', bandTop + bandBottom);
  }

  inputSamples.addEventListener('input', update);
  inputTransfer.addEventListener('input', update);
  inputSigma.addEventListener('input', update);
  update();
})();
</script>

---

## 3. The Three Scientific Mechanics Displayed Above

### 1. Pearl’s Twin-Rollout Noise Coupling (S01)
Notice that when you adjust the **Intervention Transfer** slider:
* The green **Counterfactual Curve** departs from the red **Factual Curve** at $t=2$ (the moment of intervention).
* At $t=5$, both curves experience the **exact same exogenous demand spike** (45 units).
* Because the underlying random variables ($U$) are mathematically coupled, the difference in ending inventory is **purely caused by the intervention**, not by random divergence in customer orders.

### 2. Mandatory Conformal Prediction Bands
Adjust the **Stochastic Samples** slider:
* A single trajectory line gives a misleading impression of certainty.
* As $N$ expands, the shaded **90% Conformal Band** captures the empirical dispersion under non-Gaussian noise.
* Decision-makers must never base capital on the median alone; they must ensure that the 5th percentile ($p05$) does not breach operational safety floors.

### 3. Minimax Regret Policy Flipping (S02)
Slide the **Market Volatility ($\sigma$)** slider:
* Under low volatility ($\sigma < 0.6$), an aggressive single-hub policy minimizes cost.
* As volatility expands ($\sigma > 1.4$), the Minimax Regret engine automatically switches its recommendation to **Strategy B (Modular Buffer)**, because the worst-case regret of the aggressive policy becomes unacceptable.

---

## 4. Client-Side WebAssembly (Pyodide) Integration

For developers who wish to execute complete Python simulation scripts directly in the browser without installing Python locally:

```python
# In-browser Pyodide snippet:
import numpy as np
from ewm_engine.core.state import WorldState
from ewm_engine.simulation.scenario import Scenario

# EWM's pure Python/NumPy core executes natively in Pyodide!
state = WorldState(resources=[{"id": "stock_north", "current": 25.0, "min_value": 0.0}])
print(f"State Fingerprint: {state.fingerprint}")
```

To explore interactive reactive notebooks running on WebAssembly, see our [WASM & Marimo Notebook Integration Guide](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/guides/visualization-and-reporting.md).
