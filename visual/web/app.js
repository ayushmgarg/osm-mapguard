/* MapGuard review console.
   Five ways in (overview map, scenarios, the test set, a changeset id, the live feed), one
   detail view: maps in the middle, decision and evidence on the right.
   Every string from OSM is untrusted user text, so the DOM is built with textContent only -
   never innerHTML. Every number shown comes from the server, which reads round 4's stored
   outputs (replay) or scores on this CPU (live). */
"use strict";

const FEED_REFRESH_MS = 15000;
const REPLAY_ROWS_SHOWN = 200;
const EDIT_ROWS_SHOWN = 200;
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const ESRI_CREDIT = "Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community";
const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_CREDIT = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const SIGNAL_NAMES = { who: "Who · the account", what: "What · the edits", text: "Text · what they wrote" };
const SIGNAL_SCALE = 6;   // log-odds at the end of a signal bar
const OUTCOMES = {
  caught: { name: "Caught", note: "flagged, and it was reverted", color: "--o-caught" },
  false_alarm: { name: "False alarm", note: "flagged, never reverted", color: "--o-false" },
  missed: { name: "Missed", note: "not flagged, but reverted", color: "--o-missed" },
  cleared: { name: "Cleared", note: "not flagged, never reverted", color: "--o-cleared" },
};
const TASK_NAMES = { ovid: "Ovid benchmark, 2014–2019", ovid_modern: "Ovid rule, 2014–2026" };

// loadSeq / modeSeq: only the newest request may paint, so a slow live fetch cannot
// overwrite a changeset clicked after it, and a stale picker cannot start a feed timer.
const state = {
  config: null, mode: "scenarios", current: null, feedTimer: null, replayTask: "ovid",
  loadSeq: 0, modeSeq: 0, focus: null, show: "both", types: { node: true, way: true },
  imagery: "esri", outcomes: new Set(Object.keys(OUTCOMES)), overviewTask: "ovid",
};

/* ---------------------------------------------------------------- helpers */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

async function api(path) {
  const t0 = performance.now();
  const r = await fetch(path);
  const body = await r.json();
  setTiming(path, Math.round(performance.now() - t0), body);
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

const pct = (p) => (100 * p).toFixed(0);
const heat = (p, t) => (p >= t ? "num hot" : "num cool");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const riskColor = (p, t) => (p >= t ? css("--high") : p >= t * 0.6 ? css("--mid") : css("--clean"));

function outcomeOf(label, score, threshold) {
  if (label === null || label === undefined) return null;
  const flagged = score >= threshold;
  return flagged ? (label ? "caught" : "false_alarm") : (label ? "missed" : "cleared");
}

/* ------------------------------------------------------------- status bar */

function setStatus() {
  const c = state.config;
  const modeText = { overview: "overview", scenarios: "replay · scenarios", replay: "replay · test set",
    live: "live scoring", feed: "live feed" }[state.mode];
  document.getElementById("status-mode").textContent = `mode: ${modeText}`;
  const d = state.current;
  document.getElementById("status-source").textContent = d && d.source === "live"
    ? `source: ${d.api === "sandbox" ? "OSM sandbox API" : "OSM API"}, scored on this CPU`
    : `source: ${c ? c.round : "round 4"} stored outputs`;
}

function setTiming(path, ms, body) {
  const box = document.getElementById("status-timing");
  if (!box || path.startsWith("/api/feed")) return;
  const t = body && body.timing;
  box.textContent = !t ? `last request ${ms} ms`
    : t.cached ? `cached result (scored earlier) · round trip ${ms} ms`
    : `fetch ${t.fetch_ms} ms · scoring total ${t.total_ms} ms · round trip ${ms} ms`;
}

function renderChips() {
  const c = state.config;
  const chips = document.getElementById("chips");
  chips.replaceChildren(
    el("span", { class: c.offline ? "chip off" : "chip ok" }, c.offline ? "offline: replay only" : "online"),
    el("span", { class: "chip", title: c.round }, "round 4 · 3 seeds"),
    el("span", { class: "chip", title: "live changesets: the refit hybrid tree model on this CPU" }, "live: trees on CPU"),
    el("span", { class: c.google_maps_key ? "chip ok" : "chip" }, c.google_maps_key ? "Google Maps" : "Google: link"));
}

/* ------------------------------------------------------------------- maps */

const osmMap = L.map("map-osm", { zoomControl: true, zoomSnap: 0.25, maxZoom: 19 }).setView([20, 0], 2);
const satMap = L.map("map-sat", { zoomControl: false, zoomSnap: 0.25, maxZoom: 19 }).setView([20, 0], 2);
const osmLayer = L.layerGroup().addTo(osmMap);
const satLayer = L.layerGroup().addTo(satMap);
const overviewLayer = L.layerGroup();
const canvas = L.canvas({ padding: 0.3 });
let googleMap = null;

function addTiles(offline) {
  if (offline) {
    // no network: geometry on a neutral grid instead of tiles
    document.getElementById("map-osm").classList.add("grid-bg");
    document.getElementById("map-sat").classList.add("grid-bg");
    return;
  }
  L.tileLayer(OSM_TILES, { maxZoom: 19, attribution: OSM_CREDIT }).addTo(osmMap);
  L.tileLayer(ESRI, { maxZoom: 19, attribution: ESRI_CREDIT }).addTo(satMap);
}

// Keep the maps on the same spot; the guard stops each move echoing back forever.
let syncing = false;
function follow(from, to) {
  from.on("move", () => {
    if (syncing) return;
    syncing = true;
    to.setView(from.getCenter(), from.getZoom(), { animate: false });
    syncGoogle();
    syncing = false;
    updateGoogleLink();
  });
}
follow(osmMap, satMap);
follow(satMap, osmMap);

function updateGoogleLink() {
  const c = osmMap.getCenter();
  document.getElementById("gmaps").href =
    `https://www.google.com/maps/@${c.lat.toFixed(6)},${c.lng.toFixed(6)},${osmMap.getZoom()}z/data=!3m1!1e3`;
}

function syncGoogle() {
  if (!googleMap) return;
  const c = osmMap.getCenter();
  googleMap.setCenter({ lat: c.lat, lng: c.lng });
  googleMap.setZoom(osmMap.getZoom());
}

function loadGoogle(key) {
  // Google Maps JavaScript API, loaded only when a key is configured. Display only:
  // nothing is read back from Google or copied into OSM.
  return new Promise((resolve, reject) => {
    if (window.google && window.google.maps) return resolve();
    window.__mapguardGoogleReady = resolve;
    const s = document.createElement("script");
    s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&callback=__mapguardGoogleReady&v=weekly`;
    s.async = true;
    s.onerror = () => reject(new Error("Google Maps did not load (check the key and its allowed referrers)"));
    document.head.append(s);
  });
}

async function setImagery(which) {
  state.imagery = which;
  document.querySelectorAll("#imagery-toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.imagery === which)));
  const sat = document.getElementById("map-sat");
  const goo = document.getElementById("map-google");
  if (which === "google") {
    try {
      await loadGoogle(state.config.google_maps_key);
      sat.hidden = true; goo.hidden = false;
      if (!googleMap) {
        googleMap = new google.maps.Map(goo, { mapTypeId: "hybrid", disableDefaultUI: false, streetViewControl: false, tilt: 0 });
      }
      syncGoogle();
      drawGoogleOverlay();
    } catch (err) {
      document.getElementById("imagery-error").textContent = err.message;
      document.getElementById("imagery-error").hidden = false;
      setImagery("esri");
    }
  } else {
    goo.hidden = true; sat.hidden = false;
    satMap.invalidateSize();
  }
}

let googleShapes = [];
function drawGoogleOverlay() {
  if (!googleMap || !state.current) return;
  googleShapes.forEach((s) => s.setMap(null));
  googleShapes = [];
  const d = state.current;
  if (state.mode === "overview") return;
  ((d.map && d.map.display) || []).forEach((e) => {
    if (!state.types[e.otype === "node" ? "node" : "way"]) return;
    const coords = state.show === "before" ? e.before : e.after;
    if (!coords || !coords.length) return;
    const path = coords.map(([lat, lng]) => ({ lat, lng }));
    const isFocus = state.focus !== null && d.edits[state.focus] && sameEdit(e, d.edits[state.focus]);
    googleShapes.push(path.length === 1
      ? new google.maps.Marker({ position: path[0], map: googleMap })
      : new google.maps.Polyline({ path, map: googleMap, strokeColor: isFocus ? "#e0157a" : "#ffffff", strokeWeight: isFocus ? 5 : 2 }));
  });
}

function actionColor(action) {
  return action === "create" ? css("--created") : action === "delete" ? css("--deleted") : css("--modified");
}

function shape(coords, style) {
  if (!coords || !coords.length) return null;
  return coords.length === 1 ? L.circleMarker(coords[0], { radius: 6, ...style }) : L.polyline(coords, style);
}

// An object can be edited twice in one changeset (49 per task); the version tells them apart.
// Live edits and older payloads may lack it, then otype + oid decides.
const sameEdit = (a, b) => a.otype === b.otype && a.oid === b.oid
  && (a.version === undefined || b.version === undefined || a.version === b.version);

function editIndexFor(detail, e) {
  return detail.edits.findIndex((x) => sameEdit(x, e));
}

function drawMap(detail, focusIndex, refit = true) {
  osmLayer.clearLayers();
  satLayer.clearLayers();
  const map = detail.map;
  if (!map || map.error || !map.display) return;
  const bounds = L.latLngBounds([]);
  const focusKey = focusIndex === null ? null : detail.edits[focusIndex];
  let focusBounds = null;
  map.display.forEach((e) => {
    if (!state.types[e.otype === "node" ? "node" : "way"]) return;
    const isFocus = focusKey && sameEdit(e, focusKey);
    const color = isFocus ? css("--focus") : actionColor(e.action);
    const showBefore = state.show !== "after" && e.before && e.action !== "create";
    const showAfter = state.show !== "before" && e.after && e.action !== "delete";
    const before = showBefore ? shape(e.before, { color: e.action === "delete" ? css("--deleted") : css("--before"),
      weight: isFocus ? 4 : 3, dashArray: "6 6", fill: false, opacity: 0.95 }) : null;
    const after = showAfter ? shape(e.after, { color, weight: isFocus ? 6 : 3, fillOpacity: 0.25 }) : null;
    const outline = shape(e.after || e.before, { color: isFocus ? css("--focus") : "#ffffff", weight: isFocus ? 5 : 2, fill: false });
    const pick = () => { const i = editIndexFor(detail, e); if (i >= 0) selectEdit(i); };
    for (const layer of [before, after]) {
      if (!layer) continue;
      layer.on("click", pick);
      layer.bindTooltip(`${e.action} ${e.otype} ${e.oid}`, { className: "map-label", sticky: true });
      layer.addTo(osmLayer);
      const b = layer.getBounds ? layer.getBounds() : L.latLngBounds([layer.getLatLng()]);
      bounds.extend(b);
      if (isFocus) focusBounds = b;
    }
    if (outline) { outline.on("click", pick); outline.addTo(satLayer); }
  });
  if (refit && focusBounds) osmMap.fitBounds(focusBounds, { maxZoom: 18, padding: [50, 50] });
  else if (refit && focusIndex === null && bounds.isValid()) osmMap.fitBounds(bounds, { maxZoom: 18, padding: [30, 30] });
  state.lastBounds = bounds;
  updateGoogleLink();
  drawGoogleOverlay();
}

/* --------------------------------------------------------------- decision */

function headline(detail) { return detail.models[detail.headline_model]; }

function arc(cx, cy, r, a0, a1) {
  const p = (a) => [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  const [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M ${x0} ${y0} A ${r} ${r} 0 ${a0 - a1 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`;
}

function gauge(score, threshold) {
  // a half dial: 0 at the left, 100 at the right; the tick is the flagging threshold
  const cx = 75, cy = 78, r = 62;
  const a = (v) => Math.PI * (1 - v);
  const tick = a(threshold);
  const color = riskColor(score, threshold);
  return svg("svg", { class: "gauge", viewBox: "0 0 150 92", role: "img", "aria-label": `Damage score ${pct(score)} of 100, flags at ${pct(threshold)}` },
    svg("path", { d: arc(cx, cy, r, Math.PI, 0), stroke: "#ece8de", "stroke-width": 12, fill: "none", "stroke-linecap": "round" }),
    score > 0.005 ? svg("path", { d: arc(cx, cy, r, Math.PI, a(Math.min(score, 0.999))), stroke: color, "stroke-width": 12, fill: "none", "stroke-linecap": "round" }) : svg("g"),
    svg("line", { x1: cx + (r - 11) * Math.cos(tick), y1: cy - (r - 11) * Math.sin(tick), x2: cx + (r + 9) * Math.cos(tick), y2: cy - (r + 9) * Math.sin(tick), stroke: "#0f1724", "stroke-width": 2.5 }),
    svg("text", { x: cx, y: cy - 6, "text-anchor": "middle", class: "big", fill: color }, pct(score)),
    svg("text", { x: cx, y: cy + 10, "text-anchor": "middle", class: "small" }, `flags at ${pct(threshold)}`));
}

function decisionSection(detail) {
  const m = headline(detail);
  const flagged = m.score >= m.threshold;
  const out = outcomeOf(detail.label, m.score, m.threshold);
  const truth = detail.label === null
    ? el("span", { class: "truth none" }, "live edit · no label yet")
    : el("span", { class: detail.label ? "truth bad" : "truth good" },
      detail.label ? "truth: reverted by another mapper" : "truth: never reverted",
      out && el("span", { class: "outcome-tag" }, `· ${OUTCOMES[out].name.toLowerCase()}`));
  const modelLine = detail.source === "replay"
    ? `${m.name} · round 4 · mean score of 3 seeds · threshold chosen on validation`
    : `${m.name} · refit from round 4's training data · threshold chosen on validation`;
  return el("section", { class: "decision", "aria-label": "Decision" },
    gauge(m.score, m.threshold),
    el("div", {},
      el("div", { class: flagged ? "verdict-call flag" : "verdict-call ok" }, flagged ? "Flag for review" : "No action needed"),
      el("div", { class: "verdict-model" }, modelLine),
      truth));
}

function signalsSection(detail) {
  const s = detail.signals;
  const rows = ["who", "what", "text"].map((k) => {
    const v = s[k];
    const width = Math.min(50, (50 * Math.abs(v)) / SIGNAL_SCALE);
    const fill = el("span", { class: v >= 0 ? "fill up" : "fill down" });
    fill.style.left = v >= 0 ? "50%" : `${50 - width}%`;
    fill.style.width = `${width}%`;
    return el("div", { class: "sig" }, el("span", {}, SIGNAL_NAMES[k]),
      el("span", { class: "track", role: "img", "aria-label": `${k} pull ${v.toFixed(2)}` }, fill),
      el("span", { class: "value" }, (v >= 0 ? "+" : "") + v.toFixed(1)));
  });
  const reasons = detail.reasons;
  return el("section", { class: "section", "aria-label": "Why" },
    el("h3", {}, "Why it scored this way"),
    rows,
    el("p", { class: "note" }, "Right of centre pushes towards damaging, left towards clean: the tree model's own feature contributions, in log-odds."),
    reasons.damaging.length > 0 && el("ul", { class: "tokens", "aria-label": "Edit patterns pointing to damage" },
      reasons.damaging.map((t) => el("li", { class: "bad" }, t.token))),
    reasons.benign.length > 0 && el("ul", { class: "tokens", "aria-label": "Edit patterns pointing to care" },
      reasons.benign.map((t) => el("li", { class: "good" }, t.token))));
}

function editsSection(detail) {
  const rankBy = detail.edit_ranking;
  const order = detail.edits.map((e, i) => i).sort((a, b) => (detail.edits[b][rankBy] || 0) - (detail.edits[a][rankBy] || 0));
  const top = Math.max(...detail.edits.map((e) => e[rankBy] || 0), 1e-9);
  const list = el("ul", { class: "edit-list" });
  order.slice(0, EDIT_ROWS_SHOWN).forEach((i, rank) => {
    const e = detail.edits[i];
    const name = (e.text || "").trim();
    const bar = el("span", { class: "bar" });
    bar.style.width = `${(100 * (e[rankBy] || 0)) / top}%`;
    const value = rankBy === "attention" ? `${(100 * e.attention).toFixed(1)}%` : pct(e.token_score);
    const button = el("button", { "aria-current": "false", "data-index": i },
      el("span", { class: "rank" }, `#${rank + 1}`),
      el("span", { class: `act ${e.action}` }, e.action),
      el("span", { class: "what", title: e.tokens }, `${e.otype}${name ? ` "${name.slice(0, 48)}"` : ` ${e.oid}`}`, bar),
      el("span", { class: "value" }, value));
    button.addEventListener("click", () => selectEdit(i));
    list.append(el("li", {}, button));
  });
  const hidden = order.length - EDIT_ROWS_SHOWN;
  const explain = rankBy === "attention"
    ? "Ranked by the deep model's attention: the share of its focus each edit received."
    : "Ranked by the edit-pattern model on each edit alone (the deep model's attention needs the GPU-trained network, which does not run live).";
  return el("section", { class: "section", "aria-label": "Edits" },
    el("h3", {}, `Which edits look suspicious (${detail.edits.length})`),
    el("p", { class: "note" }, explain), list,
    hidden > 0 && el("p", { class: "note" }, `${hidden} less suspicious edits not listed.`));
}

function modelsSection(detail) {
  const entries = Object.entries(detail.models);
  const rows = entries.map(([key, m]) => {
    const fill = el("span", { class: m.score >= m.threshold ? "fill up" : "fill down" });
    fill.style.width = `${100 * m.score}%`;
    const tick = el("span", { class: "tick", title: `flags at ${pct(m.threshold)}` });
    tick.style.left = `${100 * m.threshold}%`;
    return el("div", { class: key === detail.headline_model ? "row headline" : "row" },
      el("span", {}, m.name, el("span", { class: "auc" }, `PR-AUC ${m.test_pr_auc}`)),
      el("span", { class: "track", role: "img", "aria-label": `${m.name} ${pct(m.score)} of 100` }, fill, tick),
      el("span", { class: "value" }, pct(m.score)));
  });
  const note = detail.source === "replay"
    ? "Round 4's stored scores, averaged over three training seeds; the tick is each model's validation threshold. PR-AUC here is of the seed-averaged score on this test set."
    : "Scored now on this laptop's CPU by the tree model refit from round 4's training data.";
  return el("details", { class: "section models", open: entries.length <= 1 ? true : null },
    el("summary", {}, el("h3", {}, entries.length > 1 ? `Every model, same changeset (${entries.length})` : "Model")),
    rows, el("p", { class: "note" }, note));
}

function factsSection(detail) {
  const cs = detail.changeset;
  const site = state.config.apis[detail.api || "osm"];
  const made = (cs.created_at || "").replace("T", " ").replace("Z", " UTC");
  return el("section", { class: "section", "aria-label": "Changeset" },
    el("h3", {}, "The changeset"),
    el("dl", { class: "facts" },
      el("dt", {}, "Id"), el("dd", { class: "mono" }, String(detail.cid), detail.source === "live" ? ` (live, ${detail.api})` : ""),
      el("dt", {}, "Made"), el("dd", { class: "mono" }, made || "(unknown)"),
      el("dt", {}, "Comment"), el("dd", {}, cs.comment || "(none)"),
      el("dt", {}, "Editor"), el("dd", {}, cs.editor || "(unknown)"),
      el("dt", {}, "Edits"), el("dd", {}, `${cs.n_edits}: ${cs.n_create} created, ${cs.n_modify} modified, ${cs.n_delete} deleted`),
      el("dt", {}, "Account"), el("dd", {}, cs.user_history_known ? `${cs.user_n_prior}${cs.user_n_prior >= 100 ? "+" : ""} earlier changesets` : "history unknown"),
      cs.corpus && [el("dt", {}, "Corpus"), el("dd", {}, cs.corpus)]),
    el("div", { class: "links" },
      el("a", { href: `${site}/changeset/${detail.cid}`, target: "_blank", rel: "noopener" }, "OpenStreetMap ↗"),
      detail.api !== "sandbox" && el("a", { href: `https://osmcha.org/changesets/${detail.cid}`, target: "_blank", rel: "noopener" }, "OSMCha ↗")));
}

/* ------------------------------------------------------------ below maps */

function diffCard(detail, i) {
  const e = detail.edits[i];
  const shown = ((detail.map && detail.map.display) || []).find((d) => sameEdit(d, e));
  const head = el("div", { class: "edit-head" },
    el("span", { class: `act ${e.action}` }, e.action),
    el("b", {}, e.otype), el("span", { class: "id" }, `${e.oid}${shown ? ` · v${shown.version}` : ""}`),
    e.attention !== null && e.attention !== undefined && el("span", { class: "id" }, `· attention ${(100 * e.attention).toFixed(1)}%`));
  // colour only what the model itself ranked for this changeset; everything else stays neutral
  const bad = new Set((detail.reasons.damaging || []).map((r) => r.token));
  const good = new Set((detail.reasons.benign || []).map((r) => r.token));
  const patterns = el("ul", { class: "tokens" }, (e.tokens || "").split(" ").filter(Boolean).slice(0, 14)
    .map((t) => el("li", { class: bad.has(t) ? "bad" : good.has(t) ? "good" : "neutral" }, t)));
  let body;
  if (!shown) body = el("p", { class: "note" }, "No map geometry for this edit (relations are not drawn).");
  else if (shown.tags_before === null) body = el("p", { class: "note" }, "OSM hides the previous version, so only the new tags are known.");
  else {
    const before = shown.tags_before, after = shown.tags_after || {};
    const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
    // Hostile text is shown as text only, and long values are cut: a review tool must not
    // amplify an advert or abuse typed into a tag.
    const cut = (v) => (v === undefined ? "" : String(v).length > 120 ? `${String(v).slice(0, 120)}…` : String(v));
    body = keys.length ? el("table", { class: "diff" },
      el("thead", {}, el("tr", {}, el("th", {}, "Tag"), el("th", {}, "Before"), el("th", {}, "After"))),
      el("tbody", {}, keys.map((k) => {
        const kind = !(k in before) ? "added" : !(k in after) ? "removed" : before[k] !== after[k] ? "changed" : "same";
        return el("tr", { class: kind }, el("td", {}, k), el("td", {}, cut(before[k])), el("td", {}, cut(after[k])));
      }))) : el("p", { class: "note" }, "No tags on either side.");
  }
  return el("div", { class: "card" }, el("h3", {}, "What the selected edit changed"), head, body,
    el("h3", { style: "margin-top:14px" }, "Edit patterns the models read"), patterns,
    el("p", { class: "note" }, "Red / green: patterns the tree model ranked as pushing this changeset towards damaging / clean. Grey: not among its top patterns here."));
}

function contextCard(detail) {
  const m = headline(detail);
  const out = outcomeOf(detail.label, m.score, m.threshold);
  const lines = {
    caught: "The model flagged it, and another mapper did revert it.",
    false_alarm: "The model flagged it, but nobody reverted it. In the ovid error analysis, most such cases are newcomers adding real parks and paths - and some look like vandalism that was never reverted.",
    missed: "Reverted, yet scored low. In the ovid error analysis, most misses are good-faith work reverted for other reasons: imports, mapathons, edit wars.",
    cleared: "Not flagged, never reverted.",
  };
  const mapNote = detail.map && detail.map.error ? el("p", { class: "error" }, `Map unavailable: ${detail.map.error}`)
    : detail.map && detail.map.geometry_partial ? el("p", { class: "note" }, "Very large changeset: only part of it is drawn.") : null;
  return el("div", { class: "card" }, el("h3", {}, "Reading this result"),
    el("p", {}, out ? lines[out] : "A live edit has no label: red means worth a reviewer's look, not proof of vandalism."),
    mapNote,
    el("p", { class: "note" }, detail.source === "replay"
      ? "Replayed from round 4: the scores are the stored outputs of the trained models for this held-out changeset; its author never appears in training."
      : "Fetched from the API just now, featurised by the same feature builder as training, and scored by the refit tree model."));
}

/* ----------------------------------------------------------------- detail */

function selectEdit(i) {
  const d = state.current;
  if (!d) return;
  state.focus = i;
  document.querySelectorAll(".edit-list button").forEach((b) => b.setAttribute("aria-current", String(Number(b.dataset.index) === i)));
  const btn = document.querySelector(`.edit-list button[data-index="${i}"]`);
  if (btn) btn.scrollIntoView({ block: "nearest" });
  drawMap(d, i);
  document.getElementById("below").replaceChildren(diffCard(d, i), contextCard(d));
}

function showDetail(detail) {
  state.current = detail;
  state.focus = null;
  document.getElementById("detail").replaceChildren(
    decisionSection(detail), signalsSection(detail), editsSection(detail), modelsSection(detail), factsSection(detail));
  drawMap(detail, null);
  const ranked = document.querySelector(".edit-list button");
  if (ranked) selectEdit(Number(ranked.dataset.index));   // open on the most suspicious edit
  else document.getElementById("below").replaceChildren(contextCard(detail));
  setStatus();
}

// Views are addressable (#replay/ovid/58688771, #live/sandbox/123, #overview/ovid) so a demo
// can be bookmarked and the screenshot script can open a view without clicking.
const HASH_ROUTE = /^#(replay\/\w+|live\/(?:osm|sandbox))\/(\d{1,12})$/;

function openHashRoute() {
  const ov = location.hash.match(/^#overview\/(\w+)$/);
  if (ov) {
    const task = state.config.tasks[ov[1]] ? ov[1] : "ovid";
    if (state.mode !== "overview" || task !== state.overviewTask) { state.overviewTask = task; setMode("overview"); }
    return;
  }
  const route = location.hash.match(HASH_ROUTE);
  const path = route && `/api/${route[1]}/${route[2]}`;
  if (path && (!state.current || path !== currentPath())) {
    const task = route[1].startsWith("replay/") ? route[1].split("/")[1] : null;
    if (task && task !== state.replayTask && state.config.tasks[task]) {
      state.replayTask = task;                       // keep the list on the opened changeset's task
      if (state.mode === "replay") setMode("replay", false);
    }
    if (state.mode === "overview") setMode(route[1].startsWith("replay") ? "replay" : "live", false);
    load(path);
  }
}

function currentPath() {
  const d = state.current;
  return d.source === "live" ? `/api/live/${d.api}/${d.cid}` : `/api/replay/${d.task}/${d.cid}`;
}

async function load(path) {
  history.replaceState(null, "", `#${path.replace("/api/", "")}`);
  const mine = ++state.loadSeq;
  const box = document.getElementById("detail");
  box.replaceChildren(el("p", { class: "loading" }, path.includes("/live/") ? "Fetching the changeset from OSM and scoring it on this laptop…" : "Loading round 4's outputs…"));
  try {
    const detail = await api(path);
    if (mine === state.loadSeq) showDetail(detail);
  } catch (err) {
    if (mine === state.loadSeq) box.replaceChildren(el("p", { class: "error" }, err.message));
  }
}

/* ----------------------------------------------------------------- pickers */

function pickButton(title, sub, score, threshold, onPick, tag) {
  const b = el("button", { "aria-current": "false" },
    el("span", { class: "title" }, title),
    score !== null ? el("span", { class: heat(score, threshold) }, pct(score)) : (tag || el("span")),
    sub && el("span", { class: "sub" }, sub));
  b.addEventListener("click", () => {
    document.querySelectorAll(".pick-list button").forEach((x) => x.setAttribute("aria-current", "false"));
    b.setAttribute("aria-current", "true");
    onPick();
  });
  return el("li", {}, b);
}

function scenariosPicker() {
  return [el("h2", {}, "Scenarios"),
    el("p", { class: "hint" }, "Real held-out changesets, chosen to show each kind of damage - and the model's own mistakes."),
    el("ul", { class: "pick-list" }, state.config.scenarios.map((s) =>
      pickButton(s.title, s.blurb, null, null, () => load(`/api/replay/${s.task}/${s.cid}`),
        el("span", { class: "tag" }, s.task === "ovid" ? "2014–19" : "2014–26"))))];
}

async function replayPicker() {
  const task = state.replayTask;
  const models = state.config.tasks[task];
  const t = models.models[models.headline].threshold;
  const rows = await api(`/api/replay/${task}`);
  const list = el("ul", { class: "pick-list" });
  const filter = el("select", { "aria-label": "Show" },
    el("option", { value: "flagged" }, "Flagged, highest first"),
    el("option", { value: "missed" }, "Missed damage"),
    el("option", { value: "false" }, "False alarms"),
    el("option", { value: "clean" }, "Cleared, lowest first"));
  const search = el("input", { type: "search", placeholder: "Changeset id or comment", "aria-label": "Search" });
  const taskSelect = el("select", { "aria-label": "Labels" },
    Object.entries(TASK_NAMES).map(([k, v]) => el("option", { value: k, selected: task === k }, v)));
  taskSelect.addEventListener("change", () => { state.replayTask = taskSelect.value; setMode("replay"); });
  const rules = {
    flagged: [(r) => r.score >= t, (a, b) => b.score - a.score],
    missed: [(r) => r.label === 1 && r.score < t, (a, b) => a.score - b.score],
    false: [(r) => r.label === 0 && r.score >= t, (a, b) => b.score - a.score],
    clean: [(r) => r.score < t, (a, b) => a.score - b.score],
  };
  const count = el("p", { class: "hint" });
  const render = () => {
    const q = search.value.trim().toLowerCase();
    const [keep, sort] = rules[filter.value];
    const hits = rows.filter((r) => keep(r) && (!q || String(r.cid).includes(q) || (r.comment || "").toLowerCase().includes(q))).sort(sort);
    list.replaceChildren(...hits.slice(0, REPLAY_ROWS_SHOWN).map((r) => pickButton(
      String(r.cid), `${r.label ? "reverted" : "clean"} · ${r.n_edits} edits · ${r.comment || "no comment"}`,
      r.score, t, () => load(`/api/replay/${task}/${r.cid}`))));
    count.textContent = `${hits.length} changesets${hits.length > REPLAY_ROWS_SHOWN ? `, first ${REPLAY_ROWS_SHOWN} shown` : ""}`;
  };
  filter.addEventListener("change", render);
  search.addEventListener("input", render);
  render();
  return [el("h2", {}, "Test set"),
    el("p", { class: "hint" }, "Changesets by authors the models never saw in training."),
    el("div", { class: "controls" }, taskSelect, filter, search), count, list];
}

function livePicker() {
  const input = el("input", { inputmode: "numeric", pattern: "[0-9]*", placeholder: "Changeset id", "aria-label": "Changeset id" });
  const sandbox = el("input", { type: "radio", name: "api", value: "sandbox", checked: true });
  const real = el("input", { type: "radio", name: "api", value: "osm" });
  const go = (ev) => {
    ev.preventDefault();
    const id = input.value.trim();
    if (!/^\d{1,12}$/.test(id)) { input.setCustomValidity("Enter the number from the changeset's URL"); input.reportValidity(); return; }
    input.setCustomValidity("");
    load(`/api/live/${sandbox.checked ? "sandbox" : "osm"}/${id}`);
  };
  const live = state.config.live;
  return [el("h2", {}, "Score a changeset"),
    el("p", { class: "hint" }, "Paste a changeset id. MapGuard fetches it now and scores it on this laptop."),
    el("form", { class: "controls", onsubmit: go },
      el("div", { class: "radio", role: "radiogroup", "aria-label": "Where the changeset lives" },
        el("label", {}, sandbox, " OSM sandbox"), el("label", {}, real, " Real OSM")),
      el("div", { class: "row" }, input, el("button", { class: "primary", type: "submit" }, "Score"))),
    el("div", { class: "metric-grid" },
      el("div", { class: "metric" }, el("div", { class: "v" }, live.test_pr_auc), el("div", { class: "k" }, "refit model's test PR-AUC (ovid_modern)")),
      el("div", { class: "metric" }, el("div", { class: "v" }, pct(live.threshold)), el("div", { class: "k" }, "flags at (validation threshold)"))),
    el("p", { class: "hint" }, "The sandbox (master.apis.dev.openstreetmap.org) is OSM's official test copy: draw a test edit there, never on the real map.")];
}

async function feedPicker() {
  const list = el("ul", { class: "pick-list" });
  const status = el("p", { class: "hint" }, "Waiting for the first changesets to close…");
  let shown = "";
  const refresh = async () => {
    const items = await api("/api/feed");
    if (items.length) status.textContent = `${items.length} recent changesets, newest first · updates every minute`;
    // Re-render only when the list changed, so keyboard focus and the selection survive.
    const ids = items.map((it) => it.cid).join();
    if (ids === shown) return;
    shown = ids;
    list.replaceChildren(...items.map((it) => pickButton(String(it.cid),
      `${it.n_edits} edits · ${it.editor} · ${it.comment || "no comment"}`, it.score, it.threshold,
      () => load(`/api/live/osm/${it.cid}`))));
  };
  const mine = state.modeSeq;
  await refresh();
  if (mine === state.modeSeq) state.feedTimer = setInterval(refresh, FEED_REFRESH_MS);
  return [el("h2", {}, "Live feed"),
    el("p", { class: "hint" }, "Real changesets from around the world, scored as they close. Nobody vandalised anything to make this list."),
    status, list];
}

/* --------------------------------------------------------------- overview */

async function overviewPicker() {
  const task = state.overviewTask;
  const mine = state.modeSeq;
  const [metrics, rows] = await Promise.all([api(`/api/metrics/${task}`), api(`/api/replay/${task}`)]);
  if (mine !== state.modeSeq) return [];   // the user moved on while this loaded
  const t = metrics.threshold;
  const placed = rows.filter((r) => r.point).length;
  const best = metrics.round4[metrics.headline] || {};
  const hyb = metrics.round4.hybrid || {};
  const fmt = (v) => (v ? v[0].toFixed(3) : "–");
  const sd = (v) => (v ? `±${v[1].toFixed(3)}` : "");
  const toggles = Object.entries(OUTCOMES).map(([k, o]) => {
    const dot = el("span", { class: "dot" });
    dot.style.background = css(o.color);
    const b = el("button", { class: "outcome", "aria-pressed": String(state.outcomes.has(k)) },
      dot, el("span", {}, o.name, el("small", {}, o.note)), el("span", { class: "n" }, metrics.counts[k].toLocaleString()));
    b.addEventListener("click", () => {
      state.outcomes.has(k) ? state.outcomes.delete(k) : state.outcomes.add(k);
      b.setAttribute("aria-pressed", String(state.outcomes.has(k)));
      drawOverview(rows, t, task);
    });
    return b;
  });
  const taskSelect = el("select", { "aria-label": "Labels" },
    Object.entries(TASK_NAMES).map(([k, v]) => el("option", { value: k, selected: task === k }, v)));
  taskSelect.addEventListener("change", () => { state.overviewTask = taskSelect.value; history.replaceState(null, "", `#overview/${taskSelect.value}`); setMode("overview"); });
  drawOverview(rows, t, task);
  renderOverviewBelow(metrics);
  renderOverviewInspector(metrics);
  const ovid = metrics.ovid_published;
  return [el("h2", {}, "Overview"),
    el("p", { class: "hint" }, `${placed.toLocaleString()} of ${metrics.n.toLocaleString()} held-out changesets placed where they happened (${(metrics.n - placed).toLocaleString()} have no edit coordinates - tag-only way or relation edits - and are counted below but not drawn), coloured by how the best model (${state.config.tasks[task].models[metrics.headline].name}) did at its threshold.`),
    el("div", { class: "controls" }, taskSelect),
    el("div", { class: "metric-grid" },
      el("div", { class: "metric" }, el("div", { class: "v" }, fmt(best.pr_auc), el("small", {}, ` ${sd(best.pr_auc)}`)), el("div", { class: "k" }, "PR-AUC, mean of 3 seeds")),
      el("div", { class: "metric" }, el("div", { class: "v" }, fmt(best.one_per_author)), el("div", { class: "k" }, "one changeset per author")),
      el("div", { class: "metric" }, el("div", { class: "v" }, fmt(best.f1)), el("div", { class: "k" }, ovid ? `F1 · Ovid published ${ovid.f1.toFixed(3)}` : "F1 at threshold")),
      el("div", { class: "metric" }, el("div", { class: "v" }, fmt(best.accuracy)), el("div", { class: "k" }, ovid ? `accuracy · Ovid ${ovid.accuracy.toFixed(3)}` : "accuracy")),
      el("div", { class: "metric wide" }, el("div", { class: "v" }, fmt(hyb.pr_auc)), el("div", { class: "k" }, "tree baseline (hybrid) PR-AUC, same test set"))),
    el("h3", {}, "Outcomes at the threshold"),
    el("div", { class: "outcomes" }, toggles),
    el("p", { class: "note" }, `Source: ${metrics.source}. Click a dot to open that changeset.`)];
}

function drawOverview(rows, t, task) {
  overviewLayer.clearLayers();
  const bounds = L.latLngBounds([]);
  rows.forEach((r) => {
    if (!r.point) return;
    const out = outcomeOf(r.label, r.score, t);
    if (!state.outcomes.has(out)) return;
    const m = L.circleMarker(r.point, { renderer: canvas, radius: 4, weight: 0.6, color: "#ffffff",
      fillColor: css(OUTCOMES[out].color), fillOpacity: 0.85 });
    m.bindTooltip(`${r.cid} · score ${pct(r.score)} · ${OUTCOMES[out].name.toLowerCase()}`, { className: "map-label" });
    m.on("click", () => { state.replayTask = task; setMode("replay", false); load(`/api/replay/${task}/${r.cid}`); });
    m.addTo(overviewLayer);
    bounds.extend(r.point);
  });
  if (bounds.isValid() && !state.overviewFitted) { osmMap.fitBounds(bounds, { padding: [20, 20] }); state.overviewFitted = true; }
}

function renderOverviewInspector(metrics) {
  const tok = metrics.tokens || { damaging: [], benign: [] };
  const colourRows = Object.values(OUTCOMES).map((o) => {
    const dot = el("span", { class: "dot" });
    dot.style.background = css(o.color);
    return el("div", { class: "outcome", "aria-pressed": "true" }, dot, el("span", {}, o.name, el("small", {}, o.note)), el("span"));
  });
  document.getElementById("detail").replaceChildren(
    el("section", { class: "section" },
      el("h3", {}, "Reading the map"),
      el("p", { class: "note" }, "One dot per held-out changeset that has edit coordinates, at the median position of its edits. Its author never appears in training. The colour is the outcome of the best model at its validation threshold, against what really happened."),
      el("div", { class: "outcomes" }, colourRows)),
    el("section", { class: "section" },
      el("h3", {}, "What the edit-token model learned"),
      el("p", { class: "note" }, metrics.task === "ovid"
        ? "Ranked by its own weights in round 4 (seed 20260904). Damaging: the documented Pokémon Go pattern - fake parks and footpaths - and fake roads."
        : "Ranked by its own weights in round 4 (seed 20260904), on this task's training data."),
      el("ul", { class: "tokens" }, tok.damaging.map((t) => el("li", { class: "bad" }, t))),
      el("p", { class: "note", style: "margin-top:10px" }, "Benign: the habits of careful mappers."),
      el("ul", { class: "tokens" }, tok.benign.map((t) => el("li", { class: "good" }, t)))),
    el("section", { class: "section" },
      el("h3", {}, "Known limits"),
      el("p", { class: "note" }, "Labels are reverts, and a revert is not always vandalism. In the error analysis of the best ovid model's 60 most confident mistakes, most misses are good-faith work reverted for other reasons (imports, mapathons, edit wars); most false alarms are newcomers adding real parks and paths, and a few look like vandalism nobody reverted."),
      el("p", { class: "note" }, "All figures are on balanced data; real vandalism is well under 1% of edits.")));
}

function renderOverviewBelow(metrics) {
  const names = Object.fromEntries(Object.entries(state.config.tasks[metrics.task].models).map(([k, v]) => [k, v.name]));
  names.stack_base = names.stack || "Stack";
  const order = ["hybrid", "hybrid_emb", "deep", "ensemble", "stack_base", "stack_emb"];
  const vs = metrics.vs_hybrid || {};
  const rows = order.filter((k) => metrics.round4[k]).map((k) => {
    const r = metrics.round4[k], d = vs[k];
    const sig = d && d.ci95 && (d.ci95[0] > 0 || d.ci95[1] < 0);
    return el("tr", { class: k === metrics.headline ? "added" : "same" },
      el("td", {}, names[k] || k),
      el("td", {}, `${r.pr_auc[0].toFixed(3)} ±${r.pr_auc[1].toFixed(3)}`),
      el("td", {}, r.one_per_author ? r.one_per_author[0].toFixed(3) : "–"),
      el("td", {}, r.f1 ? r.f1[0].toFixed(3) : "–"),
      el("td", {}, k === "hybrid" ? "baseline" : d ? `${d.mean >= 0 ? "+" : ""}${d.mean.toFixed(4)} [${d.ci95[0].toFixed(3)}, ${d.ci95[1].toFixed(3)}]${sig ? " ✓" : ""}` : "–"));
  });
  document.getElementById("below").replaceChildren(
    el("div", { class: "card wide" },
      el("h3", {}, `Round 4, every model on this test set (${TASK_NAMES[metrics.task]})`),
      el("table", { class: "diff" },
        el("thead", {}, el("tr", {}, ["Model", "PR-AUC (3 seeds)", "One per author", "F1", "vs hybrid, 95% author bootstrap"].map((h) => el("th", {}, h)))),
        el("tbody", {}, rows)),
      el("p", { class: "note" }, "✓ = the interval excludes zero. Every figure is read from round 4's analysis files; nothing here is recomputed by the page.")));
}

/* ----------------------------------------------------------------- modes */

const WELCOME = document.querySelector("#detail .welcome").cloneNode(true);
const PICKERS = { overview: overviewPicker, scenarios: scenariosPicker, replay: replayPicker, live: livePicker, feed: feedPicker };

async function setMode(mode, clearDetail = true) {
  state.mode = mode;
  const mine = ++state.modeSeq;
  clearInterval(state.feedTimer);
  const wasOverview = document.body.classList.contains("overview");
  document.body.classList.toggle("overview", mode === "overview");
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
  if (mode === "overview") {
    state.loadSeq++;                       // a changeset still loading must not paint over the overview
    osmLayer.clearLayers(); satLayer.clearLayers(); overviewLayer.addTo(osmMap);
    state.overviewFitted = false;
    history.replaceState(null, "", `#overview/${state.overviewTask}`);
  } else {
    overviewLayer.remove();
    if (wasOverview) {
      // back from the overview: put the last changeset - or the welcome text - back
      if (state.current && clearDetail) {
        showDetail(state.current);
        history.replaceState(null, "", `#${currentPath().replace("/api/", "")}`);
      } else if (!state.current) {
        document.getElementById("below").replaceChildren();
        document.getElementById("detail").replaceChildren(WELCOME.cloneNode(true));
        history.replaceState(null, "", location.pathname + location.search);
      }
    }
  }
  setTimeout(() => { osmMap.invalidateSize(); satMap.invalidateSize(); }, 50);
  setStatus();
  const picker = document.getElementById("picker");
  picker.replaceChildren(el("p", { class: "loading" }, "Loading…"));
  try {
    const content = await PICKERS[mode]();
    if (mine === state.modeSeq) picker.replaceChildren(...content);
  } catch (err) {
    if (mine === state.modeSeq) picker.replaceChildren(el("p", { class: "error" }, err.message));
  }
}

/* ------------------------------------------------------------- map controls */

document.querySelectorAll("#state-toggle button").forEach((b) => b.addEventListener("click", () => {
  state.show = b.dataset.state;
  document.querySelectorAll("#state-toggle button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  if (state.current) drawMap(state.current, state.focus, false);
}));
document.querySelectorAll("#type-toggle button").forEach((b) => b.addEventListener("click", () => {
  state.types[b.dataset.type] = !state.types[b.dataset.type];
  b.setAttribute("aria-pressed", String(state.types[b.dataset.type]));
  if (state.current) drawMap(state.current, state.focus, false);
}));
document.getElementById("fit").addEventListener("click", () => {
  if (state.lastBounds && state.lastBounds.isValid()) osmMap.fitBounds(state.lastBounds, { maxZoom: 18, padding: [30, 30] });
});
document.querySelectorAll("#imagery-toggle button").forEach((b) => b.addEventListener("click", () => setImagery(b.dataset.imagery)));
document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

/* ------------------------------------------------------------------- start */

api("/api/config").then((config) => {
  state.config = config;
  addTiles(config.offline);
  renderChips();
  if (config.offline) {
    document.querySelectorAll("[data-needs-network]").forEach((b) => { b.disabled = true; b.title = "Offline mode: restart the server without --offline"; });
  }
  if (config.google_maps_key) document.querySelector('#imagery-toggle [data-imagery="google"]').hidden = false;
  updateGoogleLink();
  const ovStart = location.hash.match(/^#overview\/(\w+)$/);
  if (ovStart && config.tasks[ovStart[1]]) state.overviewTask = ovStart[1];
  const start = ovStart ? "overview" : "scenarios";
  setMode(start).then(openHashRoute);
  window.addEventListener("hashchange", openHashRoute);
}).catch((err) => {
  document.getElementById("picker").replaceChildren(el("p", { class: "error" }, `Server not reachable: ${err.message}`));
});
