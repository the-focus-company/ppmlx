"""Local read-only web view for the ppmlx temporal memory graph."""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ppmlx.memory_store import MemoryStore


def serve_graph_view(
    store: MemoryStore,
    *,
    host: str = "127.0.0.1",
    port: int = 6777,
    status: str | None = "active",
    query: str | None = None,
    app_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    limit: int = 120,
    open_browser: bool = True,
    on_start: Callable[[str], None] | None = None,
) -> str:
    """Serve the local graph viewer until interrupted and return the URL."""
    defaults = {
        "status": status or "active",
        "query": query or "",
        "app_id": app_id or "",
        "project_id": project_id or "",
        "session_id": session_id or "",
        "limit": str(limit),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_text(render_graph_html(defaults), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/graph":
                params = _query_params(parsed.query, defaults)
                snapshot = store.graph_snapshot(
                    status=params["status"],
                    query=params["query"],
                    app_id=params["app_id"] or None,
                    project_id=params["project_id"] or None,
                    session_id=params["session_id"] or None,
                    limit=int(params["limit"]),
                )
                self._send_json(snapshot)
                return
            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            return

        def _send_json(self, data: dict[str, Any]) -> None:
            raw = json.dumps(data, ensure_ascii=False, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_text(self, text: str, *, content_type: str) -> None:
            raw = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = httpd.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    if on_start:
        on_start(url)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return url


def render_graph_html(defaults: dict[str, str] | None = None) -> str:
    defaults_json = json.dumps(defaults or {}, ensure_ascii=False)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ppmlx graph</title>
<style>
:root { color-scheme: dark; --bg:#0b0f14; --panel:#111923; --muted:#8493a7; --text:#e6edf3; --accent:#7dd3fc; --edge:#334155; --hot:#fbbf24; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
header { display:flex; gap:16px; align-items:center; padding:14px 18px; border-bottom:1px solid #1f2937; background:#0d131b; position:sticky; top:0; z-index:2; }
h1 { font-size:18px; margin:0; letter-spacing:.02em; }
.badge { color:#08111a; background:var(--accent); border-radius:999px; padding:2px 8px; font-weight:700; }
.controls { display:grid; grid-template-columns:repeat(7, minmax(90px, 1fr)); gap:8px; padding:12px 18px; border-bottom:1px solid #1f2937; background:#0d131b; }
label { color:var(--muted); font-size:12px; display:flex; flex-direction:column; gap:4px; }
input, select, button { background:#0f1720; color:var(--text); border:1px solid #263244; border-radius:8px; padding:8px; }
button { cursor:pointer; background:#132033; }
main { display:grid; grid-template-columns:minmax(420px, 1fr) 420px; min-height:calc(100vh - 116px); }
#graph { width:100%; height:calc(100vh - 116px); background:radial-gradient(circle at center, #101923 0, #0b0f14 70%); }
aside { border-left:1px solid #1f2937; background:var(--panel); overflow:auto; height:calc(100vh - 116px); }
section { padding:14px 16px; border-bottom:1px solid #223044; }
h2 { font-size:13px; margin:0 0 8px; color:var(--accent); text-transform:uppercase; letter-spacing:.08em; }
.statgrid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.stat { background:#0d141e; border:1px solid #1f2b3d; border-radius:10px; padding:9px; }
.stat b { display:block; font-size:18px; }
.item { border:1px solid #1f2b3d; border-radius:10px; padding:10px; margin:8px 0; background:#0d141e; cursor:pointer; }
.item:hover { border-color:#3b82f6; }
.muted { color:var(--muted); }
pre { white-space:pre-wrap; word-break:break-word; background:#0b1119; border:1px solid #223044; border-radius:10px; padding:10px; max-height:360px; overflow:auto; }
.node { cursor:pointer; }
.node circle { fill:#1d4ed8; stroke:#93c5fd; stroke-width:1.5; }
.node.hot circle { fill:#92400e; stroke:#fbbf24; }
.node text { fill:#dbeafe; font-size:11px; paint-order:stroke; stroke:#0b0f14; stroke-width:3px; stroke-linejoin:round; }
.edge { stroke:var(--edge); stroke-width:1.2; opacity:.85; }
.edge-label { fill:#93a4b8; font-size:10px; paint-order:stroke; stroke:#0b0f14; stroke-width:3px; }
@media (max-width: 960px) { .controls { grid-template-columns:1fr 1fr; } main { grid-template-columns:1fr; } aside { height:auto; border-left:0; border-top:1px solid #1f2937; } #graph { height:60vh; } }
</style>
</head>
<body>
<header><h1>ppmlx graph</h1><span class="badge">local read-only</span><span class="muted" id="path"></span></header>
<div class="controls">
<label>Search <input id="query" placeholder="fact, entity, source quote" /></label>
<label>Project <input id="project_id" placeholder="project_id" /></label>
<label>Session <input id="session_id" placeholder="session_id" /></label>
<label>App <input id="app_id" placeholder="app_id" /></label>
<label>Status <select id="status"><option>active</option><option>all</option><option>disputed</option><option>rejected</option><option>superseded</option><option>forgotten</option></select></label>
<label>Limit <input id="limit" type="number" min="1" max="500" value="120" /></label>
<button id="refresh">Refresh</button>
</div>
<main>
<svg id="graph" role="img" aria-label="temporal memory graph"></svg>
<aside>
<section><h2>Stats</h2><div class="statgrid" id="stats"></div></section>
<section><h2>Selected</h2><pre id="details">Click a node, edge, or fact.</pre></section>
<section><h2>Facts</h2><div id="facts"></div></section>
<section><h2>Timeline</h2><div id="events"></div></section>
</aside>
</main>
<script>
const defaults = __DEFAULTS_JSON__;
const $ = id => document.getElementById(id);
for (const [k,v] of Object.entries(defaults)) if ($(k) && v) $(k).value = v;
$('refresh').onclick = load;
for (const id of ['query','project_id','session_id','app_id','status','limit']) $(id).addEventListener('keydown', e => { if (e.key === 'Enter') load(); });
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function load() {
  const params = new URLSearchParams();
  for (const id of ['query','project_id','session_id','app_id','status','limit']) params.set(id, $(id).value || '');
  const res = await fetch('/api/graph?' + params.toString());
  const data = await res.json();
  render(data);
}
function render(data) {
  $('path').textContent = data.path || '';
  $('stats').innerHTML = [
    ['nodes', data.nodes.length], ['edges', data.edges.length], ['facts', data.candidates.length], ['events', data.events.length],
    ['db candidates', data.stats.candidates], ['db edges', data.stats.edges]
  ].map(([k,v]) => `<div class="stat"><span class="muted">${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
  renderGraph(data);
  $('facts').innerHTML = data.candidates.map(c => `<div class="item" data-id="${esc(c.candidate_id)}"><b>${esc(c.type)} · ${esc(c.subject)} → ${esc(c.predicate)} → ${esc(c.object)}</b><div>${esc(c.text)}</div><div class="muted">${esc(c.status)} · confidence ${esc(c.confidence)} · ${esc(c.project_id || '')}/${esc(c.session_id || '')}</div></div>`).join('') || '<p class="muted">No facts for current filters.</p>';
  for (const el of document.querySelectorAll('#facts .item')) el.onclick = () => show(data.candidates.find(c => c.candidate_id === el.dataset.id));
  $('events').innerHTML = data.events.map(e => `<div class="item"><b>${esc(e.timestamp)}</b><div>${esc(e.endpoint || '')}</div><div class="muted">${esc(e.event_id)} · ${esc(e.project_id || '')}/${esc(e.session_id || '')}</div></div>`).join('') || '<p class="muted">No events.</p>';
}
function renderGraph(data) {
  const svg = $('graph'); svg.innerHTML = '';
  const w = svg.clientWidth || 800, h = svg.clientHeight || 600, cx = w/2, cy = h/2;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const nodes = data.nodes, edges = data.edges;
  if (!nodes.length) { svg.innerHTML = '<text x="24" y="40" fill="#8493a7">No graph data for current filters.</text>'; return; }
  const degree = Object.fromEntries(nodes.map(n => [n.id, 0]));
  for (const e of edges) { degree[e.from_entity_id] = (degree[e.from_entity_id] || 0) + 1; degree[e.to_entity_id] = (degree[e.to_entity_id] || 0) + 1; }
  const r = Math.max(120, Math.min(w,h) * 0.36);
  const pos = {};
  nodes.forEach((n,i) => { const a = (Math.PI*2*i/nodes.length) - Math.PI/2; pos[n.id] = {x:cx + r*Math.cos(a), y:cy + r*Math.sin(a)}; });
  for (const e of edges) {
    const a = pos[e.from_entity_id], b = pos[e.to_entity_id]; if (!a || !b) continue;
    line(a.x,a.y,b.x,b.y,'edge', () => show(e));
    text((a.x+b.x)/2, (a.y+b.y)/2, e.relation || '', 'edge-label');
  }
  for (const n of nodes) {
    const p = pos[n.id], g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.setAttribute('class', 'node ' + ((degree[n.id] || 0) > 1 ? 'hot' : ''));
    g.onclick = () => show(n); svg.appendChild(g);
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle'); c.setAttribute('cx',p.x); c.setAttribute('cy',p.y); c.setAttribute('r', String(8 + Math.min(10, degree[n.id] || n.candidate_count || 1))); g.appendChild(c);
    const t = document.createElementNS('http://www.w3.org/2000/svg','text'); t.setAttribute('x',p.x+12); t.setAttribute('y',p.y+4); t.textContent = n.name.length > 34 ? n.name.slice(0,34)+'…' : n.name; g.appendChild(t);
  }
  function line(x1,y1,x2,y2,cls,click) { const el = document.createElementNS('http://www.w3.org/2000/svg','line'); Object.entries({x1,y1,x2,y2}).forEach(([k,v]) => el.setAttribute(k,v)); el.setAttribute('class',cls); el.onclick = click; svg.appendChild(el); }
  function text(x,y,s,cls) { if (!s) return; const el = document.createElementNS('http://www.w3.org/2000/svg','text'); el.setAttribute('x',x); el.setAttribute('y',y); el.setAttribute('class',cls); el.textContent = s; svg.appendChild(el); }
}
function show(obj) { $('details').textContent = JSON.stringify(obj || null, null, 2); }
load();
</script>
</body>
</html>"""
    return html.replace("__DEFAULTS_JSON__", defaults_json)


def _query_params(raw_query: str, defaults: dict[str, str]) -> dict[str, str]:
    parsed = parse_qs(raw_query)
    out = dict(defaults)
    for key in ("status", "query", "app_id", "project_id", "session_id", "limit"):
        if key in parsed:
            out[key] = parsed[key][0]
    try:
        limit = max(1, min(int(out.get("limit") or 120), 500))
    except ValueError:
        limit = 120
    out["limit"] = str(limit)
    if not out.get("status"):
        out["status"] = "active"
    return out
