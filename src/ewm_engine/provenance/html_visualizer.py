"""Self-contained interactive HTML visualizer for SystemicTrace dependency graphs."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ewm_engine.provenance.trace import SystemicTrace


EVIDENCE_COLORS = {
    "structural": {"bg": "#1e3a8a", "border": "#3b82f6", "badge": "#60a5fa"},
    "interventional": {"bg": "#064e3b", "border": "#10b981", "badge": "#34d399"},
    "quasi_causal": {"bg": "#78350f", "border": "#f59e0b", "badge": "#fbbf24"},
    "predictive": {"bg": "#4c1d95", "border": "#8b5cf6", "badge": "#a78bfa"},
    "assumed": {"bg": "#374151", "border": "#6b7280", "badge": "#9ca3af"},
}


def render_trace_html(trace: SystemicTrace, title: str = "Systemic Trace Dependency Graph") -> str:
    """Generate a fully self-contained, offline-ready interactive HTML visualizer.

    Args:
        trace: The SystemicTrace instance containing nodes and edges.
        title: Page title displayed in the visualizer header.

    Returns:
        Complete HTML5 string with embedded CSS and vanilla JavaScript.
    """
    # 1. Structure node and edge data safely
    nodes_data: list[dict[str, Any]] = []
    steps_set: set[int] = set()
    evidence_counts: dict[str, int] = {
        "structural": 0,
        "interventional": 0,
        "quasi_causal": 0,
        "predictive": 0,
        "assumed": 0,
    }

    for node in trace.nodes.values():
        ev_val = node.evidence_level.value
        evidence_counts[ev_val] = evidence_counts.get(ev_val, 0) + 1
        steps_set.add(node.step)
        nodes_data.append(
            {
                "id": node.id,
                "step": node.step,
                "category": node.category,
                "label": node.label,
                "evidence_level": ev_val,
                "details": node.details,
            }
        )

    edges_data: list[dict[str, Any]] = [
        {
            "source": e.source,
            "target": e.target,
            "step": e.step,
            "relation": e.relation,
            "evidence_level": e.evidence_level.value,
        }
        for e in trace.edges
    ]

    sorted_steps = sorted(steps_set)
    graph_payload = json.dumps({"nodes": nodes_data, "edges": edges_data}, default=str)
    safe_title = html.escape(title)

    # 2. Build template
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title} - EWM Engine</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --surface: #151d30;
      --surface-border: #22304e;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--surface-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      z-index: 10;
    }}
    .header-left h1 {{
      font-size: 1.15rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge {{
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 9999px;
      background: #1e293b;
      color: #94a3b8;
      border: 1px solid #334155;
    }}
    .stats-bar {{
      display: flex;
      gap: 12px;
      align-items: center;
      font-size: 0.85rem;
    }}
    .stat-pill {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      background: #1e293b;
      border: 1px solid #334155;
    }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
    .dot-structural {{ background: #3b82f6; }}
    .dot-interventional {{ background: #10b981; }}
    .dot-quasi_causal {{ background: #f59e0b; }}
    .dot-predictive {{ background: #8b5cf6; }}
    .dot-assumed {{ background: #6b7280; }}

    .toolbar {{
      background: #0f172a;
      border-bottom: 1px solid var(--surface-border);
      padding: 8px 24px;
      display: flex;
      gap: 12px;
      align-items: center;
      font-size: 0.85rem;
    }}
    .step-btn {{
      background: #1e293b;
      color: var(--text-main);
      border: 1px solid #334155;
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.8rem;
      transition: all 0.15s ease;
    }}
    .step-btn:hover, .step-btn.active {{
      background: var(--primary);
      border-color: var(--primary-hover);
    }}

    .workspace {{
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }}
    #graph-container {{
      flex: 1;
      position: relative;
      overflow: auto;
      background: radial-gradient(circle, #1e293b1a 10%, transparent 11%);
      background-size: 20px 20px;
      cursor: grab;
    }}
    #graph-container:active {{ cursor: grabbing; }}
    svg {{
      min-width: 100%;
      min-height: 100%;
      display: block;
    }}

    .side-panel {{
      width: 340px;
      background: var(--surface);
      border-left: 1px solid var(--surface-border);
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .side-panel h2 {{
      font-size: 1rem;
      border-bottom: 1px solid var(--surface-border);
      padding-bottom: 8px;
    }}
    .meta-row {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 0.85rem;
    }}
    .meta-label {{ color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; }}
    .meta-value {{ font-family: monospace; word-break: break-all; }}
    pre.json-view {{
      background: #0b0f19;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid #22304e;
      font-size: 0.75rem;
      overflow-x: auto;
      max-height: 250px;
    }}

    /* Graph SVG styling */
    .edge-line {{
      stroke: #475569;
      stroke-width: 1.5;
      fill: none;
      transition: stroke 0.2s, stroke-width 0.2s;
    }}
    .edge-line.highlight {{
      stroke: #38bdf8;
      stroke-width: 2.5;
    }}
    .edge-label {{
      font-size: 10px;
      fill: #94a3b8;
      text-anchor: middle;
      font-family: monospace;
    }}
    .node-group {{
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.2s;
    }}
    .node-group:hover {{
      transform: translateY(-2px);
    }}
    .node-rect {{
      rx: 8;
      ry: 8;
      stroke-width: 1.5;
      transition: stroke-width 0.15s, stroke 0.15s;
    }}
    .node-group.active .node-rect {{
      stroke-width: 3;
      stroke: #38bdf8 !important;
    }}
    .node-title {{
      font-size: 12px;
      font-weight: 600;
      fill: #f8fafc;
    }}
    .node-sub {{
      font-size: 10px;
      fill: #94a3b8;
    }}
    .dimmed {{ opacity: 0.2; }}
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <h1>{safe_title} <span class="badge">EWM Trace</span></h1>
    </div>
    <div class="stats-bar">
      <div class="stat-pill"><strong>Nodes:</strong> {len(nodes_data)}</div>
      <div class="stat-pill"><strong>Edges:</strong> {len(edges_data)}</div>
      <div class="stat-pill"><span class="dot dot-structural"></span> Structural: {evidence_counts["structural"]}</div>
      <div class="stat-pill"><span class="dot dot-interventional"></span> Interventional: {evidence_counts["interventional"]}</div>
      <div class="stat-pill"><span class="dot dot-quasi_causal"></span> Quasi-Causal: {evidence_counts["quasi_causal"]}</div>
      <div class="stat-pill"><span class="dot dot-predictive"></span> Predictive: {evidence_counts["predictive"]}</div>
      <div class="stat-pill"><span class="dot dot-assumed"></span> Assumed: {evidence_counts["assumed"]}</div>
    </div>
  </header>

  <div class="toolbar">
    <span>Filter by Step:</span>
    <button class="step-btn active" onclick="filterStep('all')">All</button>
    {"".join(f'<button class="step-btn" onclick="filterStep({s})">Step {s}</button>' for s in sorted_steps)}
    <button class="step-btn" onclick="resetHighlight()" style="margin-left: auto;">Reset Selection</button>
  </div>

  <div class="workspace">
    <div id="graph-container">
      <svg id="trace-svg">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
          </marker>
          <marker id="arrow-highlight" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#38bdf8" />
          </marker>
        </defs>
        <g id="edges-layer"></g>
        <g id="nodes-layer"></g>
      </svg>
    </div>

    <aside class="side-panel" id="inspector">
      <h2>Trace Node Inspector</h2>
      <p style="font-size: 0.85rem; color: var(--text-muted);">Click any node in the dependency graph to inspect quantitative attributes and provenance standing.</p>
      <div id="inspector-content" style="display: none;">
        <div class="meta-row">
          <span class="meta-label">Node Identifier</span>
          <span class="meta-value" id="insp-id">-</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Simulation Step</span>
          <span class="meta-value" id="insp-step">-</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Category</span>
          <span class="meta-value" id="insp-category">-</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Evidence Level</span>
          <span class="meta-value" id="insp-evidence">-</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Label</span>
          <span class="meta-value" id="insp-label">-</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Quantitative Details</span>
          <pre class="json-view" id="insp-details"></pre>
        </div>
      </div>
    </aside>
  </div>

  <script id="trace-data" type="application/json">
    {graph_payload}
  </script>

  <script>
    const data = JSON.parse(document.getElementById('trace-data').textContent);
    const svg = document.getElementById('trace-svg');
    const nodesLayer = document.getElementById('nodes-layer');
    const edgesLayer = document.getElementById('edges-layer');

    const colors = {{
      structural: {{ bg: '#1e3a8a', border: '#3b82f6', badge: '#93c5fd' }},
      interventional: {{ bg: '#064e3b', border: '#10b981', badge: '#6ee7b7' }},
      quasi_causal: {{ bg: '#78350f', border: '#f59e0b', badge: '#fde68a' }},
      predictive: {{ bg: '#4c1d95', border: '#8b5cf6', badge: '#c4b5fd' }},
      assumed: {{ bg: '#374151', border: '#6b7280', badge: '#d1d5db' }}
    }};

    // Layout calculation: Column per step
    const stepGroups = {{}};
    data.nodes.forEach(n => {{
      if (!stepGroups[n.step]) stepGroups[n.step] = [];
      stepGroups[n.step].push(n);
    }});

    const colWidth = 260;
    const rowHeight = 90;
    const startX = 60;
    const startY = 60;
    const nodeCoords = {{}};

    const sortedStepKeys = Object.keys(stepGroups).map(Number).sort((a,b)=>a-b);
    let maxRows = 1;
    sortedStepKeys.forEach((step, colIdx) => {{
      const nodes = stepGroups[step];
      if (nodes.length > maxRows) maxRows = nodes.length;
      nodes.forEach((node, rowIdx) => {{
        const x = startX + colIdx * colWidth;
        const y = startY + rowIdx * rowHeight;
        nodeCoords[node.id] = {{ x, y, width: 200, height: 60, node }};
      }});
    }});

    const totalWidth = Math.max(1000, startX + sortedStepKeys.length * colWidth + 100);
    const totalHeight = Math.max(600, startY + maxRows * rowHeight + 100);
    svg.setAttribute('width', totalWidth);
    svg.setAttribute('height', totalHeight);

    // Render Edges
    data.edges.forEach((edge, i) => {{
      const src = nodeCoords[edge.source];
      const tgt = nodeCoords[edge.target];
      if (!src || !tgt) return;

      const x1 = src.x + src.width;
      const y1 = src.y + src.height / 2;
      const x2 = tgt.x;
      const y2 = tgt.y + tgt.height / 2;
      const dx = (x2 - x1) * 0.5;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${{x1}} ${{y1}} C ${{x1 + dx}} ${{y1}}, ${{x2 - dx}} ${{y2}}, ${{x2}} ${{y2}}`);
      path.setAttribute('class', 'edge-line');
      path.setAttribute('id', `edge-${{i}}`);
      path.setAttribute('marker-end', 'url(#arrow)');
      path.dataset.source = edge.source;
      path.dataset.target = edge.target;
      path.dataset.step = edge.step;
      edgesLayer.appendChild(path);

      if (edge.relation) {{
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2 - 6;
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', midX);
        text.setAttribute('y', midY);
        text.setAttribute('class', 'edge-label');
        text.textContent = edge.relation;
        edgesLayer.appendChild(text);
      }}
    }});

    // Render Nodes
    Object.values(nodeCoords).forEach(item => {{
      const n = item.node;
      const c = colors[n.evidence_level] || colors.assumed;

      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'node-group');
      g.setAttribute('id', `node-${{n.id}}`);
      g.dataset.id = n.id;
      g.dataset.step = n.step;

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', item.x);
      rect.setAttribute('y', item.y);
      rect.setAttribute('width', item.width);
      rect.setAttribute('height', item.height);
      rect.setAttribute('class', 'node-rect');
      rect.setAttribute('fill', c.bg);
      rect.setAttribute('stroke', c.border);
      g.appendChild(rect);

      // Title
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      title.setAttribute('x', item.x + 12);
      title.setAttribute('y', item.y + 24);
      title.setAttribute('class', 'node-title');
      const maxLen = 22;
      title.textContent = n.label.length > maxLen ? n.label.substring(0, maxLen) + '...' : n.label;
      g.appendChild(title);

      // Subtitle (category + evidence)
      const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      sub.setAttribute('x', item.x + 12);
      sub.setAttribute('y', item.y + 44);
      sub.setAttribute('class', 'node-sub');
      sub.setAttribute('fill', c.badge);
      sub.textContent = `${{n.category.toUpperCase()}} (${{n.evidence_level}})`;
      g.appendChild(sub);

      g.onclick = () => selectNode(n.id);
      nodesLayer.appendChild(g);
    }});

    function selectNode(nodeId) {{
      const item = nodeCoords[nodeId];
      if (!item) return;
      const n = item.node;

      // Update inspector
      document.getElementById('inspector-content').style.display = 'flex';
      document.getElementById('insp-id').textContent = n.id;
      document.getElementById('insp-step').textContent = n.step;
      document.getElementById('insp-category').textContent = n.category;
      document.getElementById('insp-evidence').textContent = n.evidence_level;
      document.getElementById('insp-label').textContent = n.label;
      document.getElementById('insp-details').textContent = JSON.stringify(n.details, null, 2);

      // Highlight connections
      const connected = new Set([nodeId]);
      document.querySelectorAll('.edge-line').forEach(line => {{
        if (line.dataset.source === nodeId || line.dataset.target === nodeId) {{
          line.classList.add('highlight');
          line.setAttribute('marker-end', 'url(#arrow-highlight)');
          connected.add(line.dataset.source);
          connected.add(line.dataset.target);
        }} else {{
          line.classList.remove('highlight');
          line.setAttribute('marker-end', 'url(#arrow)');
        }}
      }});

      document.querySelectorAll('.node-group').forEach(grp => {{
        if (connected.has(grp.dataset.id)) {{
          grp.classList.remove('dimmed');
          if (grp.dataset.id === nodeId) grp.classList.add('active');
          else grp.classList.remove('active');
        }} else {{
          grp.classList.add('dimmed');
          grp.classList.remove('active');
        }}
      }});
    }}

    function resetHighlight() {{
      document.querySelectorAll('.node-group').forEach(grp => {{
        grp.classList.remove('dimmed');
        grp.classList.remove('active');
      }});
      document.querySelectorAll('.edge-line').forEach(line => {{
        line.classList.remove('highlight');
        line.setAttribute('marker-end', 'url(#arrow)');
      }});
    }}

    function filterStep(step) {{
      document.querySelectorAll('.step-btn').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');

      document.querySelectorAll('.node-group').forEach(grp => {{
        if (step === 'all' || parseInt(grp.dataset.step) <= step) {{
          grp.style.display = 'block';
        }} else {{
          grp.style.display = 'none';
        }}
      }});

      document.querySelectorAll('.edge-line').forEach(line => {{
        if (step === 'all' || parseInt(line.dataset.step) <= step) {{
          line.style.display = 'block';
        }} else {{
          line.style.display = 'none';
        }}
      }});
    }}
  </script>
</body>
</html>
"""
