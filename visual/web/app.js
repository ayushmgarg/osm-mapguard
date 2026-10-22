/* MapGuard review console. One page, four ways to pick a changeset (scenarios, the test
   set, a changeset id, the live feed), one detail view. Every string from OSM is untrusted
   user text, so the DOM is built with textContent only - never innerHTML. */
"use strict";

const FEED_REFRESH_MS = 15000;
const REPLAY_ROWS_SHOWN = 200;
const EDIT_ROWS_SHOWN = 200;
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const ESRI_CREDIT = "Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community";
const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_CREDIT = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const SIGNAL_NAMES = { who: "Who: the account", what: "What: the edits", text: "Text: what they wrote" };
const SIGNAL_SCALE = 6;   // log-odds at the end of a signal bar

// loadSeq / modeSeq: only the newest request may paint, so a slow live fetch cannot
// overwrite a changeset clicked after it, and a stale picker cannot start a feed timer.
const state = { config: null, mode: "scenarios", current: null, feedTimer: null, replayTask: "ovid", loadSeq: 0, modeSeq: 0 };

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

async function api(path) {
  const r = await fetch(path);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

const pct = (p) => (100 * p).toFixed(0);
const heat = (p, t) => (p >= t ? "num hot" : "num cool");

/* ------------------------------------------------------------------- maps */

const osmMap = L.map("map-osm", { zoomControl: true }).setView([20, 0], 2);
L.tileLayer(OSM_TILES, { maxZoom: 19, attribution: OSM_CREDIT }).addTo(osmMap);
const satMap = L.map("map-sat", { zoomControl: false }).setView([20, 0], 2);
L.tileLayer(ESRI, { maxZoom: 19, attribution: ESRI_CREDIT }).addTo(satMap);
const osmLayer = L.layerGroup().addTo(osmMap);
const satLayer = L.layerGroup().addTo(satMap);

// Keep both maps on the same spot; the guard stops each move echoing back forever.
let syncing = false;
function follow(from, to) {
  from.on("move", () => {
    if (syncing) return;
    syncing = true;
    to.setView(from.getCenter(), from.getZoom(), { animate: false });
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

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function shape(coords, style) {
  if (!coords || !coords.length) return null;
  return coords.length === 1 ? L.circleMarker(coords[0], { radius: 6, ...style }) : L.polyline(coords, style);
}

function drawMap(detail, focusIndex) {
  osmLayer.clearLayers();
  satLayer.clearLayers();
  const map = detail.map;
  if (!map || map.error || !map.display) return;
  const bounds = L.latLngBounds([]);
  const focusKey = focusIndex === null ? null : detail.edits[focusIndex];
  map.display.forEach((e) => {
    const isFocus = focusKey && e.otype === focusKey.otype && e.oid === focusKey.oid;
    const before = shape(e.before, { color: css("--before"), weight: 3, dashArray: "6 6", fill: false });
    const after = shape(e.after, { color: isFocus ? css("--focus") : css("--after"), weight: isFocus ? 6 : 3 });
    const outline = shape(e.after, { color: isFocus ? css("--focus") : "#ffffff", weight: isFocus ? 5 : 2, fill: false });
    for (const layer of [before, after]) if (layer) { layer.addTo(osmLayer); bounds.extend(layer.getBounds ? layer.getBounds() : layer.getLatLng()); }
    if (outline) outline.addTo(satLayer);
    if (isFocus) {
      const target = after || before;
      if (target) osmMap.fitBounds(target.getBounds ? target.getBounds() : L.latLngBounds([target.getLatLng()]), { maxZoom: 18, padding: [40, 40] });
    }
  });
  if (focusIndex === null && bounds.isValid()) osmMap.fitBounds(bounds, { maxZoom: 18, padding: [30, 30] });
  updateGoogleLink();
}

/* ----------------------------------------------------------------- detail */

function headline(detail) {
  return detail.models[detail.headline_model];
}

function verdictSection(detail) {
  const m = headline(detail);
  const flagged = m.score >= m.threshold;
  const cs = detail.changeset;
  const site = state.config.apis[detail.api || "osm"];
  const truth = detail.label === null ? null
    : detail.label === 1 ? "Reverted by another mapper (damaging)" : "Not reverted (clean)";
  return el("section", { class: "verdict", "aria-label": "Decision" },
    el("div", { class: flagged ? "sign flag" : "sign" },
      el("span", { class: "score", title: "Damage score out of 100" }, pct(m.score)),
      el("span", { class: "call" }, flagged ? "Flag for review" : "No action needed",
        el("span", { class: "why" }, `${m.name}; flags at ${pct(m.threshold)}`))),
    el("dl", { class: "facts" },
      truth && [el("dt", {}, "Ground truth"), el("dd", { class: "truth" }, truth)],
      el("dt", {}, "Changeset"), el("dd", {}, String(detail.cid), detail.source === "live" ? ` (live, ${detail.api})` : ""),
      el("dt", {}, "Made"), el("dd", {}, (cs.created_at || "").replace("T", " ").replace("Z", " UTC")),
      el("dt", {}, "Comment"), el("dd", {}, cs.comment || "(none)"),
      el("dt", {}, "Editor"), el("dd", {}, cs.editor || "(unknown)"),
      el("dt", {}, "Edits"), el("dd", {}, `${cs.n_edits} (${cs.n_create} created, ${cs.n_modify} modified, ${cs.n_delete} deleted)`),
      el("dt", {}, "Account"), el("dd", {}, cs.user_history_known ? `${cs.user_n_prior} earlier changesets` : "history unknown")),
    el("div", { class: "links" },
      el("a", { href: `${site}/changeset/${detail.cid}`, target: "_blank", rel: "noopener" }, "View on OpenStreetMap"),
      detail.api !== "sandbox" && el("a", { href: `https://osmcha.org/changesets/${detail.cid}`, target: "_blank", rel: "noopener" }, "View on OSMCha")));
}

function signalsSection(detail) {
  const s = detail.signals;
  const rows = ["who", "what", "text"].map((k) => {
    const v = s[k];
    const width = Math.min(50, (50 * Math.abs(v)) / SIGNAL_SCALE);
    const fill = el("span", { class: v >= 0 ? "fill up" : "fill down" });
    fill.style.left = v >= 0 ? "50%" : `${50 - width}%`;
    fill.style.width = `${width}%`;
    return el("div", { class: "row" }, el("span", {}, SIGNAL_NAMES[k]),
      el("span", { class: "track", role: "img", "aria-label": `${k} pull ${v.toFixed(2)}` }, fill),
      el("span", { class: "value" }, (v >= 0 ? "+" : "") + v.toFixed(1)));
  });
  const reasons = detail.reasons;
  return el("section", { class: "signals", "aria-label": "Why" },
    el("h3", {}, "Why it scored this way"),
    rows,
    el("p", { class: "note" }, "Bars right of centre push towards damaging, left towards clean (log-odds, from the tree model's own feature contributions)."),
    reasons.damaging.length > 0 && el("ul", { class: "tokens", "aria-label": "Edit patterns pointing to damage" },
      reasons.damaging.map((t) => el("li", { class: "bad" }, t.token))),
    reasons.benign.length > 0 && el("ul", { class: "tokens", "aria-label": "Edit patterns pointing to care" },
      reasons.benign.map((t) => el("li", { class: "good" }, t.token))));
}

function modelsSection(detail) {
  const entries = Object.entries(detail.models);
  return el("section", { class: "models", "aria-label": "Every model's score" },
    el("h3", {}, entries.length > 1 ? "Every model, same changeset" : "Model"),
    entries.map(([key, m]) => {
      const fill = el("span", { class: m.score >= m.threshold ? "fill up" : "fill down" });
      fill.style.left = "0";
      fill.style.width = `${100 * m.score}%`;
      const tick = el("span", { class: "tick", title: `flags at ${pct(m.threshold)}` });
      tick.style.left = `${100 * m.threshold}%`;
      return el("div", { class: key === detail.headline_model ? "row headline" : "row" },
        el("span", {}, m.name, el("br"), el("span", { class: "auc" }, `test PR-AUC ${m.test_pr_auc}`)),
        el("span", { class: "track", role: "img", "aria-label": `${m.name} ${pct(m.score)} of 100` }, fill, tick),
        el("span", { class: "value" }, pct(m.score)));
    }),
    el("p", { class: "note" }, detail.source === "replay"
      ? "Round 4's stored scores, averaged over three training seeds. The tick marks each model's flagging threshold, chosen on validation data."
      : "Scored now on this laptop's CPU by the tree model refit from round 4's training data."));
}

function editLabel(e) {
  const name = (e.text || "").trim();
  return `${e.action} ${e.otype}${name ? ` "${name.slice(0, 40)}"` : ""}`;
}

function editsSection(detail) {
  const rankBy = detail.edit_ranking;
  const order = detail.edits.map((e, i) => i).sort((a, b) => detail.edits[b][rankBy] - detail.edits[a][rankBy]);
  const top = Math.max(...detail.edits.map((e) => e[rankBy]), 1e-9);
  const diffBox = el("div", { "aria-live": "polite" });
  const list = el("ul", { class: "edit-list" });
  const select = (i, button) => {
    list.querySelectorAll("button").forEach((b) => b.setAttribute("aria-current", "false"));
    button.setAttribute("aria-current", "true");
    drawMap(detail, i);
    diffBox.replaceChildren(tagDiff(detail, i));
  };
  const hidden = order.length - EDIT_ROWS_SHOWN;
  order.slice(0, EDIT_ROWS_SHOWN).forEach((i) => {
    const e = detail.edits[i];
    const fill = el("span", { class: "fill" });
    fill.style.width = `${(100 * e[rankBy]) / top}%`;
    const button = el("button", { "aria-current": "false" },
      el("span", { class: "track" }, fill), el("span", { class: "what" }, editLabel(e)),
      el("span", { class: "value" }, rankBy === "attention" ? `${(100 * e.attention).toFixed(1)}%` : pct(e.token_score)));
    button.addEventListener("click", () => select(i, button));
    list.append(el("li", {}, button));
  });
  const explain = rankBy === "attention"
    ? "Ranked by the deep model's attention: the share of its focus each edit received."
    : "Ranked by the edit-pattern model applied to each edit alone (the deep model's attention needs a GPU-trained network that does not run live).";
  const section = el("section", { class: "edits", "aria-label": "Edits" },
    el("div", {}, el("h3", {}, "Which edits look suspicious"), el("p", { class: "note" }, explain), list,
      hidden > 0 && el("p", { class: "note" }, `${hidden} less suspicious edits not listed.`)),
    el("div", {}, el("h3", {}, "What the selected edit changed"), diffBox));
  // Open on the most suspicious edit: it is the answer to "which edit?".
  section.selectFirst = () => { const first = list.querySelector("button"); if (first) first.click(); };
  return section;
}

function tagDiff(detail, i) {
  const e = detail.edits[i];
  const shown = (detail.map && detail.map.display || []).find((d) => d.otype === e.otype && d.oid === e.oid);
  if (!shown) return el("p", { class: "note" }, `Patterns: ${e.tokens}`);
  const before = shown.tags_before;
  const after = shown.tags_after || {};
  if (before === null) return el("p", { class: "note" }, "The previous version is hidden by OSM, so only the new tags are known.");
  const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
  const rows = keys.map((k) => {
    const kind = !(k in before) ? "added" : !(k in after) ? "removed" : before[k] !== after[k] ? "changed" : "same";
    return el("tr", { class: kind }, el("td", {}, k), el("td", {}, before[k] ?? ""), el("td", {}, after[k] ?? ""));
  });
  return el("div", {},
    el("p", { class: "note" }, `${e.action} ${e.otype} ${e.oid}, version ${shown.version}. Patterns: ${e.tokens}`),
    keys.length ? el("table", { class: "diff" },
      el("thead", {}, el("tr", {}, el("th", {}, "Tag"), el("th", {}, "Before"), el("th", {}, "After"))),
      el("tbody", {}, rows)) : el("p", { class: "note" }, "No tags on either side."),
    !shown.before && !shown.after && el("p", { class: "note" }, "No map geometry for this edit (relations are not drawn)."));
}

function showDetail(detail) {
  state.current = detail;
  const box = document.getElementById("detail");
  const mapNote = detail.map && detail.map.error
    ? el("p", { class: "error" }, `Map unavailable: ${detail.map.error}`)
    : detail.map && detail.map.geometry_partial && el("p", { class: "note" }, "Very large changeset: only part of it is drawn.");
  const edits = editsSection(detail);
  box.replaceChildren(verdictSection(detail), el("div", {}, signalsSection(detail), modelsSection(detail)),
    mapNote || "", edits);
  drawMap(detail, null);
  edits.selectFirst();
  if (window.matchMedia("(max-width: 900px)").matches) box.scrollIntoView({ block: "start" });
}

// A changeset view is addressable (#replay/ovid/58688771, #live/sandbox/123) so a demo can
// be bookmarked and the screenshot script can open a view without clicking.
const HASH_ROUTE = /^#(replay\/\w+|live\/(?:osm|sandbox))\/(\d{1,12})$/;

function openHashRoute() {
  const route = location.hash.match(HASH_ROUTE);
  const path = route && `/api/${route[1]}/${route[2]}`;
  if (path && (!state.current || path !== currentPath())) load(path);
}

function currentPath() {
  const d = state.current;
  return d.source === "live" ? `/api/live/${d.api}/${d.cid}` : `/api/replay/${d.task}/${d.cid}`;
}

async function load(path) {
  history.replaceState(null, "", `#${path.replace("/api/", "")}`);
  const mine = ++state.loadSeq;
  const box = document.getElementById("detail");
  box.replaceChildren(el("p", { class: "loading" }, "Fetching the changeset and scoring it..."));
  try {
    const detail = await api(path);
    if (mine === state.loadSeq) showDetail(detail);
  } catch (err) {
    if (mine === state.loadSeq) box.replaceChildren(el("p", { class: "error" }, err.message));
  }
}

/* ----------------------------------------------------------------- pickers */

function pickButton(title, sub, score, threshold, onPick) {
  const b = el("button", { "aria-current": "false" },
    el("span", { class: "title" }, title),
    score !== null && el("span", { class: heat(score, threshold) }, pct(score)),
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
    el("p", { class: "hint" }, "Real changesets from the held-out test set, chosen to show each kind of damage, and the model's mistakes."),
    el("ul", { class: "pick-list" }, state.config.scenarios.map((s) =>
      pickButton(s.title, s.blurb, null, null, () => load(`/api/replay/${s.task}/${s.cid}`))))];
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
    el("option", { value: "ovid", selected: task === "ovid" }, "Ovid benchmark, 2014 to 2019"),
    el("option", { value: "ovid_modern", selected: task === "ovid_modern" }, "Ovid rule, 2014 to 2026"));
  taskSelect.addEventListener("change", () => { state.replayTask = taskSelect.value; setMode("replay"); });
  const rules = {
    flagged: [(r) => r.score >= t, (a, b) => b.score - a.score],
    missed: [(r) => r.label === 1 && r.score < t, (a, b) => a.score - b.score],
    false: [(r) => r.label === 0 && r.score >= t, (a, b) => b.score - a.score],
    clean: [(r) => r.score < t, (a, b) => a.score - b.score],
  };
  const render = () => {
    const q = search.value.trim().toLowerCase();
    const [keep, sort] = rules[filter.value];
    const hits = rows.filter((r) => keep(r) && (!q || String(r.cid).includes(q) || (r.comment || "").toLowerCase().includes(q))).sort(sort);
    list.replaceChildren(...hits.slice(0, REPLAY_ROWS_SHOWN).map((r) => pickButton(
      String(r.cid), `${r.label ? "reverted" : "clean"}; ${r.n_edits} edits; ${r.comment || "no comment"}`,
      r.score, t, () => load(`/api/replay/${task}/${r.cid}`))));
    count.textContent = `${hits.length} changesets${hits.length > REPLAY_ROWS_SHOWN ? `, first ${REPLAY_ROWS_SHOWN} shown` : ""}`;
  };
  const count = el("p", { class: "hint" });
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
  return [el("h2", {}, "Score a changeset"),
    el("p", { class: "hint" }, "Paste a changeset id. MapGuard fetches it from OpenStreetMap now and scores it on this laptop."),
    el("form", { class: "controls", onsubmit: go },
      el("div", { class: "radio", role: "radiogroup", "aria-label": "Where the changeset lives" },
        el("label", {}, sandbox, " OSM sandbox"), el("label", {}, real, " Real OSM")),
      el("div", { class: "row" }, input, el("button", { class: "primary", type: "submit" }, "Score"))),
    el("p", { class: "hint" }, "The sandbox (master.apis.dev.openstreetmap.org) is OSM's official test copy: draw a test edit there, never on the real map.")];
}

async function feedPicker() {
  const list = el("ul", { class: "pick-list" });
  const status = el("p", { class: "hint" }, "Waiting for the first changesets to close...");
  let shown = "";
  const refresh = async () => {
    const items = await api("/api/feed");
    if (items.length) status.textContent = `${items.length} recent changesets, newest first. Updates every minute.`;
    // Re-render only when the list changed, so keyboard focus and the selection survive.
    const ids = items.map((it) => it.cid).join();
    if (ids === shown) return;
    shown = ids;
    list.replaceChildren(...items.map((it) => pickButton(String(it.cid),
      `${it.n_edits} edits; ${it.editor}; ${it.comment || "no comment"}`, it.score, it.threshold,
      () => load(`/api/live/osm/${it.cid}`))));
  };
  const mine = state.modeSeq;
  await refresh();
  if (mine === state.modeSeq) state.feedTimer = setInterval(refresh, FEED_REFRESH_MS);
  return [el("h2", {}, "Live feed"),
    el("p", { class: "hint" }, "Real changesets from around the world, scored as they close. Nobody vandalised anything to make this list."),
    status, list];
}

const PICKERS = { scenarios: scenariosPicker, replay: replayPicker, live: livePicker, feed: feedPicker };

async function setMode(mode) {
  state.mode = mode;
  const mine = ++state.modeSeq;
  clearInterval(state.feedTimer);
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
  const picker = document.getElementById("picker");
  picker.replaceChildren(el("p", { class: "loading" }, "Loading..."));
  try {
    const content = await PICKERS[mode]();
    if (mine === state.modeSeq) picker.replaceChildren(...content);
  } catch (err) {
    if (mine === state.modeSeq) picker.replaceChildren(el("p", { class: "error" }, err.message));
  }
}

document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

api("/api/config").then((config) => {
  state.config = config;
  setMode("scenarios");
  openHashRoute();
  window.addEventListener("hashchange", openHashRoute);
}).catch((err) => {
  document.getElementById("picker").replaceChildren(el("p", { class: "error" }, `Server not reachable: ${err.message}`));
});
