"use strict";

/* ── helpers ──────────────────────────────────────────────────────────── */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.body !== undefined) {
    opts.method = opts.method || "POST";
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function toast(message, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = message;
  $("#toast").append(t);
  setTimeout(() => t.remove(), kind === "bad" ? 8000 : 3500);
}

const fmt = {
  num: (v, d = 3) => (v === null || v === undefined ? "-" : Number(v).toFixed(d)),
  pct: (v) => (v === null || v === undefined ? "-" : `${Math.round(Number(v) * 100)}%`),
  bool: (v) => (v === null || v === undefined ? "-" : v ? "yes" : "no"),
  int: (v) => (v === null || v === undefined ? "-" : Number(v).toLocaleString()),
  time: (iso) => (iso ? new Date(iso).toLocaleString(undefined,
    { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : ""),
};

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

let STATE = null;
async function refreshState() {
  STATE = await api("/api/state");
  renderSpend();
}

function modelName(key) {
  const m = [...STATE.models, ...STATE.mocks].find((x) => x.key === key);
  return m ? m.name : key;
}

function renderSpend() {
  const s = STATE.spend;
  const node = $("#spend");
  const models = Object.values(s.per_model || {});
  if (!models.length) { node.textContent = "No paid runs yet"; return; }
  if (s.all_priced && s.budget_eur) {
    node.textContent = `Spent about EUR ${s.total_eur.toFixed(2)} of ${s.budget_eur}`;
  } else {
    const tin = models.reduce((a, m) => a + m.prompt_tokens, 0);
    const tout = models.reduce((a, m) => a + m.completion_tokens, 0);
    node.textContent = `Tokens used: ${fmt.int(tin)} in, ${fmt.int(tout)} out`;
  }
}

/* ── routing ──────────────────────────────────────────────────────────── */
const VIEWS = { run: viewRun, results: viewResults, matrix: viewMatrix, prompts: viewPrompts, settings: viewSettings };

function parseHash() {
  const [tab, qs] = location.hash.replace(/^#/, "").split("?");
  return { tab: VIEWS[tab] ? tab : "run", params: Object.fromEntries(new URLSearchParams(qs || "")) };
}

async function route() {
  const { tab, params } = parseHash();
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  const root = $("#view");
  try {
    await refreshState();
    await VIEWS[tab](root, params);
  } catch (err) {
    root.innerHTML = `<div class="notice bad">${esc(err.message)}</div>`;
  }
}

$("#tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (tab) location.hash = tab.dataset.tab;
});
window.addEventListener("hashchange", route);

/* ── Run ──────────────────────────────────────────────────────────────── */
const JOBS = [];
let selectedJob = null;
let pollTimer = null;

function callsFor(slice, mode) {
  const n = Math.min(slice.n, 240);
  const m = mode || slice.group_mode;
  return m === "single" ? 1 : Math.ceil(n / 60) + 1;
}

async function viewRun(root) {
  const slices = STATE.slices.filter((s) => !s.error);
  if (!slices.length) {
    root.innerHTML = `<div class="card"><h2>No data yet</h2><p class="sub">There are no slices in
      <code>data/</code>. Generate them with the generator in <code>nse-genbench-data</code>.</p></div>`;
    return;
  }
  const firstReady = STATE.models.find((m) => m.ready);
  const modelOptions = `
    <optgroup label="Models under test">${STATE.models.map((m) =>
      `<option value="${esc(m.key)}" ${m.ready ? "" : "disabled"}>${esc(m.name)}${m.ready ? "" : ` (${esc(m.not_ready_reason)})`}</option>`).join("")}</optgroup>
    <optgroup label="Mock models (no cost, no network)">${STATE.mocks.map((m) =>
      `<option value="${esc(m.key)}">${esc(m.name)}</option>`).join("")}</optgroup>`;

  root.innerHTML = `
  <div class="cols">
    <section class="card">
      <h2>New run</h2>
      <p class="sub">Pick what to test. Every run is scored automatically and saved.</p>
      <div class="field"><label for="r-slice">Data slice</label>
        <select id="r-slice">${slices.map((s) => `<option value="${esc(s.id)}">${esc(s.id)} (${s.n} answers)</option>`).join("")}</select></div>
      <div id="r-slice-info" class="notice small"></div>
      <div class="field" style="margin-top:12px"><label for="r-model">Model</label>
        <select id="r-model">${modelOptions}</select><div class="hint" id="r-model-hint"></div></div>
      <div class="field"><label for="r-variant">Prompt</label>
        <select id="r-variant">${STATE.variants.map((v) => `<option value="${esc(v.id)}">${esc(v.id)}${v.locked ? " (group prompt, unchanged)" : ` (changes ${esc(v.dimension)})`}</option>`).join("")}</select>
        <div class="hint" id="r-variant-hint"></div></div>
      <div class="row">
        <div class="field"><label for="r-repeats">Repeats</label><input id="r-repeats" type="number" min="1" max="20" value="1"></div>
        <div class="field"><label for="r-mode">Mode</label><select id="r-mode">
          <option value="">As the group pipeline</option><option value="single">Single-pass</option><option value="mapreduce">Map-reduce</option></select></div>
      </div>
      <details><summary>Advanced</summary>
        <div class="row">
          <div class="field"><label for="r-temp">Temperature</label><input id="r-temp" type="number" step="0.1" min="0" max="2" placeholder="0.1 (default)"><div class="hint" id="r-temp-hint"></div></div>
          <div class="field"><label for="r-max">Output limit (tokens)</label><input id="r-max" type="number" min="1" placeholder="8192 (default)"></div>
        </div>
      </details>
      <div class="notice small" id="r-cost"></div>
      <div class="row" style="margin-top:14px"><div class="spacer"></div><button class="btn primary" id="r-start">Start run</button></div>
    </section>
    <section class="card">
      <h2>Progress</h2>
      <p class="sub">Runs go one at a time. You can switch tabs; they keep going.</p>
      <div class="jobs" id="r-jobs"></div>
      <pre class="log" id="r-log">No run started yet.</pre>
      <div class="row" style="margin-top:12px"><div class="spacer"></div><button class="btn hidden" id="r-open">Open these results</button></div>
    </section>
  </div>`;

  const sel = { slice: $("#r-slice"), model: $("#r-model"), variant: $("#r-variant"),
    repeats: $("#r-repeats"), mode: $("#r-mode"), temp: $("#r-temp"), max: $("#r-max") };
  sel.model.value = firstReady ? firstReady.key : STATE.mocks[0].key;

  function update() {
    const slice = slices.find((s) => s.id === sel.slice.value);
    const planted = slice.planted ? Object.entries(slice.planted).map(([k, v]) => `${esc(k)} ${v}`).join(", ") : "";
    $("#r-slice-info").innerHTML = slice.theme
      ? `<dl class="kv"><dt>Theme</dt><dd>${esc(slice.theme)}</dd>
         <dt>Planted</dt><dd>${planted}</dd>
         <dt>Minority</dt><dd>${esc(slice.minority)}</dd>
         <dt>Absent</dt><dd>${esc(slice.absent)} (any mention is invented)</dd>
         <dt>Group mode</dt><dd>${slice.group_mode === "single" ? "single-pass" : "map-reduce"}</dd></dl>`
      : "No answer key for this slice: the truth-based scores will show '-'.";
    const model = STATE.models.find((m) => m.key === sel.model.value);
    $("#r-model-hint").textContent = model ? `${model.role}. Sends ${model.model_id || "?"}.` : "Returns a fixed card. Free.";
    const noTemp = model && !model.temperature;
    sel.temp.disabled = noTemp;
    if (noTemp) sel.temp.value = "";
    $("#r-temp-hint").textContent = noTemp ? `${model.name} does not take a temperature.` : "";
    const variant = STATE.variants.find((v) => v.id === sel.variant.value);
    $("#r-variant-hint").textContent = variant ? variant.description : "";
    const repeats = Math.max(1, Number(sel.repeats.value) || 1);
    const calls = callsFor(slice, sel.mode.value) * repeats;
    $("#r-cost").textContent = model
      ? `This makes ${calls} model call${calls === 1 ? "" : "s"} to ${model.name}. Calls already in the cache cost nothing.`
      : `Mock model: ${calls} call${calls === 1 ? "" : "s"}, no network, no cost.`;
  }
  Object.values(sel).forEach((s) => s.addEventListener("input", update));
  update();

  $("#r-start").addEventListener("click", async () => {
    const btn = $("#r-start");
    btn.disabled = true;
    try {
      const { job_id } = await api("/api/run", { body: {
        slice: sel.slice.value, model: sel.model.value, variant: sel.variant.value,
        repeats: sel.repeats.value, mode: sel.mode.value, temperature: sel.temp.value, max_tokens: sel.max.value } });
      JOBS.unshift({ id: job_id, label: `${modelName(sel.model.value)} / ${sel.variant.value} / ${sel.slice.value} x${sel.repeats.value}`, status: "queued" });
      selectJob(job_id);
    } catch (err) {
      toast(err.message, "bad");
    } finally {
      btn.disabled = false;
    }
  });
  renderJobs();
  if (selectedJob) selectJob(selectedJob);
}

function renderJobs() {
  const box = $("#r-jobs");
  if (!box) return;
  const badge = { queued: "", running: "accent", done: "good", error: "bad" };
  box.innerHTML = JOBS.map((j) => `<div class="job ${j.id === selectedJob ? "selected" : ""}" data-job="${j.id}">
      <span class="badge ${badge[j.status]}">${j.status}</span><span>${esc(j.label)}</span></div>`).join("");
  $$(".job", box).forEach((n) => n.addEventListener("click", () => selectJob(n.dataset.job)));
}

function selectJob(id) {
  selectedJob = id;
  clearTimeout(pollTimer);
  poll();
}

async function poll() {
  if (!selectedJob) return;
  try {
    const job = await api(`/api/jobs/${selectedJob}`);
    const entry = JOBS.find((j) => j.id === job.id);
    if (entry) entry.status = job.status;
    renderJobs();
    const log = $("#r-log");
    if (log) {
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
      log.textContent = job.log.join("\n").trim() + (job.error ? `\n\nerror: ${job.error}` : "") || "Waiting...";
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    const open = $("#r-open");
    if (open) {
      open.classList.toggle("hidden", job.status !== "done");
      open.onclick = () => { location.hash = `results?run_ids=${job.run_ids.join(",")}`; };
    }
    if (job.status === "done") { await refreshState(); toast("Run finished and saved", "good"); }
    if (job.status === "error") toast(job.error, "bad");
    if (job.status === "queued" || job.status === "running") pollTimer = setTimeout(poll, 800);
  } catch (err) {
    toast(err.message, "bad");
  }
}

/* ── Results & scoring ────────────────────────────────────────────────── */
let RESULT_FILTERS = { slice: "", model: "", variant: "", scored: "" };
let selectedRun = null;

function ratingMean(r) {
  const vals = STATE.rating_dimensions.map((d) => r[d]).filter((v) => v !== null && v !== undefined);
  return vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1) : null;
}

async function viewResults(root, params) {
  const onlyIds = params.run_ids || "";
  if (params.variant !== undefined || params.model !== undefined || params.slice !== undefined) {
    RESULT_FILTERS = { slice: params.slice || "", model: params.model || "", variant: params.variant || "", scored: "" };
  }
  const options = (values, current, label) => `<option value="">${label}</option>` +
    values.map((v) => `<option value="${esc(v)}" ${v === current ? "selected" : ""}>${esc(v)}</option>`).join("");

  root.innerHTML = `
  <div class="cols">
    <section class="card">
      <div class="row" style="align-items:center"><h2>Runs</h2><div class="spacer"></div>
        <button class="btn primary" id="blind">Score blind</button></div>
      <p class="sub">Click a run to read its card and score it. "Score blind" picks a random unscored run
        and hides the model, the prompt and the automatic scores until you save.</p>
      <div id="only-ids"></div>
      <div class="row">
        <div class="field"><select id="f-slice">${options(STATE.slices.map((s) => s.id), RESULT_FILTERS.slice, "All slices")}</select></div>
        <div class="field"><select id="f-model">${options([...STATE.models, ...STATE.mocks].map((m) => m.key), RESULT_FILTERS.model, "All models")}</select></div>
        <div class="field"><select id="f-variant">${options(STATE.variants.map((v) => v.id), RESULT_FILTERS.variant, "All prompts")}</select></div>
        <div class="field"><select id="f-scored">
          <option value="">Scored or not</option><option value="no" ${RESULT_FILTERS.scored === "no" ? "selected" : ""}>Not scored yet</option>
          <option value="yes" ${RESULT_FILTERS.scored === "yes" ? "selected" : ""}>Scored</option></select></div>
      </div>
      <div class="table-wrap" style="max-height:62vh"><table class="data" id="res-table"></table></div>
    </section>
    <section class="card" id="detail"><p class="muted">Select a run on the left.</p></section>
  </div>`;

  if (onlyIds) {
    $("#only-ids").innerHTML = `<div class="notice small" style="margin-bottom:12px">Showing the runs from your last job.
      <a href="#results">Show all runs</a></div>`;
  }
  for (const key of ["slice", "model", "variant", "scored"]) {
    $(`#f-${key}`).addEventListener("change", (e) => { RESULT_FILTERS[key] = e.target.value; loadTable(onlyIds); });
  }
  $("#blind").addEventListener("click", () => scoreBlind());
  await loadTable(onlyIds);
  if (onlyIds && onlyIds.split(",").length === 1) showRun(onlyIds, false);
}

async function loadTable(onlyIds) {
  const q = new URLSearchParams({ ...RESULT_FILTERS, ...(onlyIds ? { run_ids: onlyIds } : {}) });
  const { rows } = await api(`/api/results?${q}`);
  const table = $("#res-table");
  if (!table) return;
  if (!rows.length) {
    table.innerHTML = `<tbody><tr><td class="muted">No runs match. Start one on the Run tab.</td></tr></tbody>`;
    return;
  }
  table.innerHTML = `<thead><tr><th>When</th><th>Model</th><th>Prompt</th><th>Slice</th><th class="num">#</th>
      <th class="num">Coverage</th><th>Minority</th><th>Invented</th><th class="num">Off-theme</th><th>Your score</th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="clickable ${r.run_id === selectedRun ? "selected" : ""}" data-run="${r.run_id}">
      <td>${esc(fmt.time(r.created_utc))}</td><td>${esc(modelName(r.model))}</td><td>${esc(r.variant)}</td>
      <td>${esc(r.slice)}</td><td class="num">${r.sample}</td>
      <td class="num">${r.parse_error ? '<span class="badge bad">no JSON</span>' : fmt.num(r.coverage, 2)}</td>
      <td>${badgeBool(r.minority_preserved, true)}</td><td>${badgeBool(r.hallucinated_absent, false)}</td>
      <td class="num">${r.off_theme_cited ?? "-"}</td>
      <td>${r.scored ? `<span class="badge accent">${ratingMean(r) ?? "scored"}</span>` : '<span class="muted">-</span>'}</td>
    </tr>`).join("")}</tbody>`;
  $$("tr.clickable", table).forEach((tr) => tr.addEventListener("click", () => showRun(tr.dataset.run, false)));
}

function badgeBool(v, goodWhenTrue) {
  if (v === null || v === undefined) return '<span class="muted">-</span>';
  const good = goodWhenTrue ? v : !v;
  return `<span class="badge ${good ? "good" : "bad"}">${v ? "yes" : "no"}</span>`;
}

async function scoreBlind() {
  const slice = RESULT_FILTERS.slice;
  const pick = await api(`/api/blind?${new URLSearchParams({ slice })}`);
  if (!pick.run_id) {
    toast("Every real-model run is scored already. Mock runs are left out of blind scoring.");
    return;
  }
  showRun(pick.run_id, true, pick.remaining);
}

const RATING_HELP = {
  usefulness: "Would this help a quality officer decide what to improve?",
  trustworthiness: "Could you check what it says, and does it hold up?",
  completeness: "Does it cover what students raised, including the smaller groups?",
};

async function showRun(runId, blind, remaining) {
  selectedRun = runId;
  $$("#res-table tr.clickable").forEach((tr) => tr.classList.toggle("selected", tr.dataset.run === runId));
  const box = $("#detail");
  box.innerHTML = `<p class="muted">Loading...</p>`;
  const { record, cited, score } = await api(`/api/results/${runId}`);
  const card = record.card || {};
  const s = record.scores || {};
  const ratings = { ...(score?.ratings || {}) };
  const claims = { ...(score?.claims || {}) };

  const metrics = [
    ["Valid JSON", fmt.bool(s.valid_json)], ["Coverage", fmt.num(s.coverage, 2)],
    ["Minority kept", fmt.bool(s.minority_preserved)], ["Invented absent subtheme", fmt.bool(s.hallucinated_absent)],
    ["Off-theme answers cited", s.off_theme_cited ?? "-"], ["Evidence-ID validity", fmt.num(s.evidence_validity, 2)],
    ["Evidence support", fmt.num(s.evidence_support, 2)], ["Max leakage", fmt.num(s.max_leakage, 2)],
    ["Language", s.lang ?? "-"], ["Words", s.words ?? "-"],
  ];
  const tokens = record.tokens || {};
  const header = blind
    ? `<div class="row" style="align-items:center"><h2>Blind scoring</h2><span class="badge warn">model and prompt hidden</span>
        <div class="spacer"></div><span class="muted small">${remaining ?? ""} unscored left</span></div>
       <p class="sub">Score the card on its own. The model, prompt and automatic scores appear after you save.</p>`
    : identityHeader(record, tokens);
  const list = (items) => (items && items.length ? `<ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<p class="muted">none</p>');
  const manifest = (card.subtheme_manifest || []).filter((e) => e && typeof e === "object");

  box.innerHTML = `
    <div id="d-header">${header}</div>
    <div id="d-metrics" class="${blind ? "hidden" : ""}" style="margin-bottom:16px">
      <div class="metrics">${metrics.map(([l, v]) => `<div class="metric"><div class="label">${l}</div><div class="value">${esc(v)}</div></div>`).join("")}</div>
    </div>
    ${record.parse_error ? `<div class="notice bad">${esc(record.parse_error)}</div>` : `
    <div class="insight">
      <section><h3>Summary</h3><p style="margin:0">${esc(card.summary || "")}</p></section>
      <div class="cols even" style="gap:14px">
        <section><h3>Positive points</h3>${list(card.positive_comments)}</section>
        <section><h3>Critical points</h3>${list(card.critical_comments)}</section>
      </div>
      <section><h3>Student suggestions (meant to be verbatim)</h3>${list(card.student_suggestions)}</section>
      <section><h3>Subthemes</h3><div class="chips">${(card.subthemes || []).map((t) => `<span class="badge">${esc(t)}</span>`).join("") || '<span class="muted">none</span>'}</div></section>
      <section><h3>Claim check: is each subtheme supported by the answers it cites?</h3>
        <div class="stack" style="gap:8px">${manifest.map((e, i) => claimBlock(e, i, cited, claims[String(i)])).join("") || '<p class="muted">The card cites no evidence.</p>'}</div>
      </section>
    </div>`}
    <hr style="border:0;border-top:1px solid var(--border);margin:18px 0">
    <h3 style="margin-bottom:10px">Your ratings (1 = poor, 5 = excellent; each judged on its own)</h3>
    ${STATE.rating_dimensions.map((d) => `<div class="rating"><strong style="text-transform:capitalize">${d}</strong>
      <div class="seg" data-dim="${d}">${[1, 2, 3, 4, 5].map((n) => `<button type="button" data-v="${n}" class="${ratings[d] === n ? "on" : ""}">${n}</button>`).join("")}</div>
      <span class="hint">${RATING_HELP[d]}</span></div>`).join("")}
    <div class="row" style="margin-top:8px">
      <div class="field" style="flex:3"><label for="d-note">Note</label><textarea id="d-note" rows="2">${esc(score?.note || "")}</textarea></div>
      <div class="field" style="flex:1"><label for="d-scorer">Your name</label><input id="d-scorer" value="${esc(score?.scorer || localStorage.getItem("scorer") || "")}"></div>
    </div>
    <div class="row" style="align-items:center">
      <span class="muted small" id="d-status">${score ? `Last saved ${esc(fmt.time(score.scored_utc))}${score.scorer ? ` by ${esc(score.scorer)}` : ""}. Saving again keeps the old score as history.` : "Not scored yet."}</span>
      <div class="spacer"></div>
      <button class="btn hidden" id="d-next">Next blind run</button>
      <button class="btn primary" id="d-save">Save score</button>
    </div>
    <details style="margin-top:14px"><summary>Raw card JSON</summary><pre class="diff">${esc(JSON.stringify(card, null, 2))}</pre></details>`;

  $$(".seg[data-dim] button", box).forEach((b) => b.addEventListener("click", () => {
    const dim = b.parentElement.dataset.dim;
    const v = Number(b.dataset.v);
    ratings[dim] = ratings[dim] === v ? null : v;
    $$("button", b.parentElement).forEach((x) => x.classList.toggle("on", Number(x.dataset.v) === ratings[dim]));
  }));
  $$(".seg.yesno", box).forEach((seg) => $$("button", seg).forEach((b) => b.addEventListener("click", () => {
    const idx = seg.dataset.claim;
    const v = b.dataset.v === "yes";
    claims[idx] = claims[idx] === v ? null : v;
    $$("button", seg).forEach((x) => x.classList.toggle("on", claims[idx] !== null && claims[idx] !== undefined && (x.dataset.v === "yes") === claims[idx]));
  })));

  $("#d-save").addEventListener("click", async () => {
    const scorer = $("#d-scorer").value.trim();
    localStorage.setItem("scorer", scorer);
    try {
      const saved = await api("/api/scores", { body: { run_id: runId, ratings, claims, note: $("#d-note").value, scorer } });
      toast("Score saved", "good");
      $("#d-status").textContent = `Last saved ${fmt.time(saved.scored_utc)}${saved.scorer ? ` by ${saved.scorer}` : ""}. Saving again keeps the old score as history.`;
      if (blind) {
        $("#d-header").innerHTML = identityHeader(record, tokens);
        $("#d-metrics").classList.remove("hidden");
        $("#d-next").classList.remove("hidden");
        $("#d-next").onclick = () => scoreBlind();
      }
      await loadTable(parseHash().params.run_ids || "");
    } catch (err) {
      toast(err.message, "bad");
    }
  });
}

function identityHeader(record, tokens) {
  const variant = STATE.variants.find((v) => v.id === record.variant);
  return `<div class="row" style="align-items:center"><h2>${esc(modelName(record.model_key))}</h2>
      <span class="badge accent">${esc(record.variant)}</span><span class="badge">${esc(record.slice)}</span>
      <span class="badge">repeat ${record.sample}</span><span class="badge">${esc(record.mode)}</span></div>
    <p class="sub">${esc(fmt.time(record.created_utc))}${variant && !variant.locked ? `, prompt changes ${esc(record.variant_dimension)}` : ""},
      ${record.request ? `${fmt.int(tokens.prompt_tokens)} tokens in, ${fmt.int(tokens.completion_tokens)} out` : "mock, no tokens"}${tokens.from_cache ? `, ${tokens.from_cache} call(s) from the cache` : ""}</p>`;
}

function claimBlock(entry, i, cited, current) {
  const ids = (entry.evidence_ids || []).map(String);
  const evidence = ids.length
    ? `<ul class="evidence">${ids.map((id) => {
        const c = cited[id] || {};
        const cls = !c.exists ? "missing" : c.off_theme ? "off" : "";
        const tag = !c.exists ? '<span class="badge bad">ID not in the data: invented</span> '
          : c.off_theme ? `<span class="badge warn">off-theme answer (${esc(c.label)})</span> ` : "";
        return `<li class="${cls}"><span class="mono">${esc(id)}</span> ${tag}${esc(c.text || "")}</li>`;
      }).join("")}</ul>`
    : '<p class="muted small" style="margin:0">Cites no answers.</p>';
  const on = (v) => (current === v ? "on" : "");
  return `<div class="claim"><div class="head"><div class="text"><strong>${esc(entry.name || "")}</strong>
      <div class="muted small">${esc(entry.description || "")}</div></div>
      <div class="seg yesno" data-claim="${i}"><button type="button" data-v="yes" class="yes ${on(true)}">Supported</button>
      <button type="button" data-v="no" class="no ${on(false)}">Not supported</button></div></div>${evidence}</div>`;
}

/* ── Matrix ───────────────────────────────────────────────────────────── */
let MATRIX = { slice: "", metric: "coverage" };

async function viewMatrix(root) {
  const metrics = STATE.matrix_metrics;
  root.innerHTML = `
  <section class="card">
    <div class="row"><div><h2>Prompt x model</h2>
      <p class="sub">Each cell is the mean over every run of that prompt on that model. Click a cell to see its runs.</p></div>
      <div class="spacer"></div>
      <div class="field" style="max-width:220px"><label for="m-slice">Slice</label><select id="m-slice">
        <option value="">All slices</option>${STATE.slices.map((s) => `<option ${s.id === MATRIX.slice ? "selected" : ""}>${esc(s.id)}</option>`).join("")}</select></div>
      <div class="field" style="max-width:280px"><label for="m-metric">Show</label><select id="m-metric">
        ${Object.entries(metrics).map(([k, m]) => `<option value="${k}" ${k === MATRIX.metric ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select></div>
    </div>
    <div id="m-table"></div>
    <div class="legend">worse <span class="bar"></span> better</div>
  </section>`;
  $("#m-slice").addEventListener("change", (e) => { MATRIX.slice = e.target.value; drawMatrix(); });
  $("#m-metric").addEventListener("change", (e) => { MATRIX.metric = e.target.value; drawMatrix(); });
  await drawMatrix();
}

async function drawMatrix() {
  const data = await api(`/api/matrix?${new URLSearchParams({ slice: MATRIX.slice })}`);
  const box = $("#m-table");
  if (!box) return;
  if (!data.variants.length) {
    box.innerHTML = `<p class="muted">No runs yet for this selection.</p>`;
    return;
  }
  const meta = data.metrics[MATRIX.metric];
  const isRating = ["usefulness", "trustworthiness", "completeness"].includes(MATRIX.metric);
  const isCount = MATRIX.metric === "off_theme_cited";
  const values = Object.values(data.cells).map((c) => c[MATRIX.metric]).filter((v) => v !== null);
  const maxCount = Math.max(1, ...values);
  const norm = (v) => (isRating ? (v - 1) / 4 : isCount ? v / maxCount : v);
  const show = (v) => (isRating ? Number(v).toFixed(1) : isCount ? Number(v).toFixed(1)
    : ["coverage", "evidence_validity", "evidence_support"].includes(MATRIX.metric) ? Number(v).toFixed(2) : fmt.pct(v));
  const color = (v) => {
    let x = Math.min(1, Math.max(0, norm(v)));
    if (!meta.higher) x = 1 - x;
    const tone = x >= 0.5 ? "var(--good)" : "var(--bad)";
    return `color-mix(in srgb, ${tone} ${Math.round(Math.abs(x - 0.5) * 2 * 38)}%, var(--panel))`;
  };
  box.innerHTML = `<div class="table-wrap"><table class="data matrix"><thead><tr><th>Prompt</th>
      ${data.models.map((m) => `<th style="text-align:center">${esc(modelName(m))}</th>`).join("")}</tr></thead>
    <tbody>${data.variants.map((v) => `<tr><th style="position:static">${esc(v)}</th>${data.models.map((m) => {
      const c = data.cells[`${v}|${m}`];
      if (!c) return `<td class="empty">-</td>`;
      const val = c[MATRIX.metric];
      const sd = MATRIX.metric === "coverage" && c.coverage_sd !== undefined ? ` sd ${c.coverage_sd.toFixed(2)}` : "";
      return `<td class="cell" data-v="${esc(v)}" data-m="${esc(m)}" style="background:${val === null ? "transparent" : color(val)}">
        <div class="v">${val === null ? "-" : show(val)}</div><div class="n">n=${c.n}${sd}${isRating || MATRIX.metric === "claims_supported" ? `, ${c.n_scored} scored` : ""}</div></td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>`;
  $$("td.cell", box).forEach((td) => td.addEventListener("click", () => {
    location.hash = `results?${new URLSearchParams({ variant: td.dataset.v, model: td.dataset.m, slice: MATRIX.slice })}`;
  }));
}

/* ── Prompts ──────────────────────────────────────────────────────────── */
let PROMPT_VIEW = { variant: "P0-neutral", kind: "single", theme: "Examination & Assessment" };

async function viewPrompts(root) {
  if (!STATE.variants.find((v) => v.id === PROMPT_VIEW.variant)) PROMPT_VIEW.variant = STATE.variants[0].id;
  root.innerHTML = `
  <div class="cols">
    <section class="card">
      <div class="row" style="align-items:center"><h2>Prompt variants</h2><div class="spacer"></div>
        <button class="btn primary" id="p-new">New variant</button></div>
      <p class="sub">P0 is the group's prompt, unchanged. Every other variant is P0 with a few declared
        text changes, and should change one thing only, so results stay comparable.</p>
      <div class="stack" id="p-list" style="gap:8px"></div>
    </section>
    <section class="card" id="p-right"></section>
  </div>`;
  renderVariantList();
  $("#p-new").addEventListener("click", () => newVariantForm());
  await showVariant(PROMPT_VIEW.variant);
}

function renderVariantList() {
  $("#p-list").innerHTML = STATE.variants.map((v) => `
    <div class="variant ${v.id === PROMPT_VIEW.variant ? "selected" : ""}" data-id="${esc(v.id)}">
      <div class="title"><strong>${esc(v.id)}</strong>
        ${v.locked ? '<span class="badge accent">baseline</span>' : `<span class="badge">${esc(v.dimension)}</span>`}
        <span class="muted small">${v.runs} run${v.runs === 1 ? "" : "s"}</span><div class="spacer"></div>
        ${v.locked ? "" : `<button class="btn small danger" data-delete="${esc(v.id)}">Remove</button>`}</div>
      <div class="small">${esc(v.description)}</div>
      ${v.edits.map(([o, n]) => `<div class="edit-line"><del>${esc(o)}</del> <span>becomes</span> <ins>${esc(n) || "(nothing)"}</ins></div>`).join("")}
    </div>`).join("");
  $$(".variant").forEach((n) => n.addEventListener("click", (e) => {
    if (e.target.dataset.delete) return;
    PROMPT_VIEW.variant = n.dataset.id;
    renderVariantList();
    showVariant(n.dataset.id);
  }));
  $$("[data-delete]").forEach((b) => b.addEventListener("click", async () => {
    const id = b.dataset.delete;
    const v = STATE.variants.find((x) => x.id === id);
    const note = v.runs ? ` Its ${v.runs} saved run(s) stay, with a copy of its changes.` : "";
    if (!confirm(`Remove prompt variant ${id}?${note}`)) return;
    try {
      await api(`/api/variants/${encodeURIComponent(id)}`, { method: "DELETE" });
      toast(`Removed ${id}`, "good");
      await refreshState();
      PROMPT_VIEW.variant = "P0";
      viewPrompts($("#view"));
    } catch (err) { toast(err.message, "bad"); }
  }));
}

function kindAndThemeControls(prefix) {
  return `<div class="row">
    <div class="field"><label for="${prefix}-kind">Prompt</label><select id="${prefix}-kind">
      <option value="single" ${PROMPT_VIEW.kind === "single" ? "selected" : ""}>Single-pass (up to 60 answers)</option>
      <option value="map" ${PROMPT_VIEW.kind === "map" ? "selected" : ""}>Map step (one per batch of 60)</option>
      <option value="reduce" ${PROMPT_VIEW.kind === "reduce" ? "selected" : ""}>Reduce step (merges the batches)</option></select></div>
    <div class="field"><label for="${prefix}-theme">Theme</label><select id="${prefix}-theme">
      ${STATE.themes.map((t) => `<option ${t === PROMPT_VIEW.theme ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></div></div>`;
}

function diffHtml(diff) {
  return diff.map((p) => p.op === "same" ? esc(p.text) : p.op === "del" ? `<del>${esc(p.text)}</del>` : `<ins>${esc(p.text)}</ins>`).join("");
}

async function showVariant(id) {
  const right = $("#p-right");
  const v = STATE.variants.find((x) => x.id === id);
  right.innerHTML = `<h2>${esc(id)}: what the model receives</h2>
    <p class="sub">${v.locked ? "The group pipeline's prompt, exactly as it runs today." : "Compared with P0: struck-through text is removed, highlighted text is added."}
      The answers at the end are a small example.</p>
    ${kindAndThemeControls("v")}<pre class="diff" id="v-diff">Loading...</pre>`;
  const draw = async () => {
    PROMPT_VIEW.kind = $("#v-kind").value;
    PROMPT_VIEW.theme = $("#v-theme").value;
    const data = await api(`/api/variants/${encodeURIComponent(id)}/prompt?${new URLSearchParams({ kind: PROMPT_VIEW.kind, theme: PROMPT_VIEW.theme })}`);
    $("#v-diff").innerHTML = diffHtml(data.diff);
  };
  $("#v-kind").addEventListener("change", draw);
  $("#v-theme").addEventListener("change", draw);
  await draw();
}

function newVariantForm() {
  $$(".variant").forEach((n) => n.classList.remove("selected"));
  const right = $("#p-right");
  right.innerHTML = `<h2>New prompt variant</h2>
    <p class="sub">Start from P0 and change one thing. Copy the exact text you want to change from the
      preview below; only text that is part of the prompt can be changed.</p>
    <div class="row">
      <div class="field"><label for="n-id">Name</label><input id="n-id" placeholder="P1-dutch-output"></div>
      <div class="field"><label for="n-dim">What it changes</label><input id="n-dim" placeholder="output_language"><div class="hint">One thing, as one name.</div></div>
    </div>
    <div class="field"><label for="n-desc">Description</label><textarea id="n-desc" rows="2" placeholder="Why this variant exists and what it tests"></textarea></div>
    <h3 style="margin:6px 0 8px">Changes</h3>
    <div id="n-edits"></div>
    <button class="btn small" id="n-add" type="button">Add a change</button>
    <div style="margin-top:14px">${kindAndThemeControls("n")}</div>
    <div id="n-problems"></div>
    <pre class="diff" id="n-diff">Loading...</pre>
    <div class="row" style="margin-top:12px"><div class="spacer"></div>
      <button class="btn" id="n-cancel">Cancel</button><button class="btn primary" id="n-save">Save variant</button></div>`;
  const edits = [{ old: "", new: "" }];
  const drawEdits = () => {
    $("#n-edits").innerHTML = edits.map((e, i) => `<div class="edit-row">
      <textarea rows="2" data-i="${i}" data-f="old" placeholder="Text in P0">${esc(e.old)}</textarea>
      <textarea rows="2" data-i="${i}" data-f="new" placeholder="Replace with (empty removes it)">${esc(e.new)}</textarea>
      <button class="btn small danger" data-remove="${i}" type="button">Remove</button></div>`).join("");
    $$("#n-edits textarea").forEach((t) => t.addEventListener("input", () => { edits[t.dataset.i][t.dataset.f] = t.value; refresh(); }));
    $$("#n-edits [data-remove]").forEach((b) => b.addEventListener("click", () => {
      edits.splice(Number(b.dataset.remove), 1);
      if (!edits.length) edits.push({ old: "", new: "" });
      drawEdits(); refresh();
    }));
  };
  const refresh = debounce(async () => {
    PROMPT_VIEW.kind = $("#n-kind").value;
    PROMPT_VIEW.theme = $("#n-theme").value;
    const filled = edits.filter((e) => e.old);
    const data = await api("/api/variants/preview", { body: { edits: filled, kind: PROMPT_VIEW.kind, theme: PROMPT_VIEW.theme } });
    $("#n-problems").innerHTML = filled.length && data.problems.length
      ? `<div class="notice bad small" style="margin-bottom:10px">${data.problems.map(esc).join("<br>")}</div>` : "";
    $("#n-diff").innerHTML = diffHtml(data.diff);
  }, 250);
  drawEdits();
  $("#n-add").addEventListener("click", () => { edits.push({ old: "", new: "" }); drawEdits(); });
  $("#n-kind").addEventListener("change", refresh);
  $("#n-theme").addEventListener("change", refresh);
  $("#n-cancel").addEventListener("click", () => viewPrompts($("#view")));
  $("#n-save").addEventListener("click", async () => {
    try {
      const id = $("#n-id").value.trim();
      await api("/api/variants", { body: { id, dimension: $("#n-dim").value, description: $("#n-desc").value,
        edits: edits.filter((e) => e.old) } });
      toast(`Saved ${id}`, "good");
      await refreshState();
      PROMPT_VIEW.variant = id;
      viewPrompts($("#view"));
    } catch (err) { toast(err.message, "bad"); }
  });
  refresh();
}

/* ── Settings ─────────────────────────────────────────────────────────── */
async function viewSettings(root) {
  const s = STATE.spend;
  const perModel = Object.entries(s.per_model || {});
  const budget = s.budget_eur;
  root.innerHTML = `
  <div class="cols even">
    <section class="card">
      <h2>Endpoint</h2>
      <p class="sub">The university's OpenAI-compatible endpoint. Saved in <code>.env</code> on this computer; never committed to git.</p>
      <div class="field"><label for="s-url">Base URL</label><input id="s-url" value="${esc(STATE.endpoint.base_url)}" placeholder="https://.../v1"></div>
      <div class="field"><label for="s-key">API key</label><input id="s-key" type="password" autocomplete="off"
        placeholder="${STATE.endpoint.api_key_set ? "set (leave empty to keep it)" : "not set"}"></div>
      <div class="row"><div class="spacer"></div><button class="btn primary" id="s-save-endpoint">Save endpoint</button></div>
    </section>
    <section class="card">
      <h2>Budget</h2>
      <p class="sub">Only calls that reached the endpoint count; answers from the cache are free.</p>
      ${s.all_priced && budget && perModel.length ? `<div class="meter"><div style="width:${Math.min(100, (s.total_eur / budget) * 100)}%"></div></div>
        <p>About EUR ${s.total_eur.toFixed(2)} of ${budget} spent.</p>`
        : `<p class="muted small">${perModel.length ? "Fill in prices per model below to see euros." : "No paid runs yet."} Budget: EUR ${budget ?? "?"}.</p>`}
      ${perModel.length ? `<div class="table-wrap"><table class="data"><thead><tr><th>Model</th><th class="num">Runs</th>
        <th class="num">Tokens in</th><th class="num">Tokens out</th><th class="num">of which reasoning</th><th class="num">EUR</th></tr></thead>
        <tbody>${perModel.map(([k, m]) => `<tr><td>${esc(modelName(k))}</td><td class="num">${m.runs}</td><td class="num">${fmt.int(m.prompt_tokens)}</td>
        <td class="num">${fmt.int(m.completion_tokens)}</td><td class="num">${fmt.int(m.reasoning_tokens)}</td><td class="num">${m.eur === null ? "-" : m.eur.toFixed(3)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    </section>
  </div>
  <section class="card" style="margin-top:16px">
    <h2>Models</h2>
    <p class="sub">Fill in the exact model id the endpoint expects. Prices are optional and only drive the budget display.</p>
    <div class="table-wrap"><table class="data"><thead><tr><th>Model</th><th>Role</th><th>Model id</th><th>Temperature</th>
      <th>Output limit field</th><th>EUR / M in</th><th>EUR / M out</th></tr></thead>
      <tbody>${STATE.models.map((m) => `<tr data-key="${esc(m.key)}"><td><strong>${esc(m.name)}</strong></td><td class="muted">${esc(m.role)}</td>
        <td><input data-f="model_id" value="${esc(m.model_id)}" placeholder="from the university" style="min-width:230px"></td>
        <td>${m.temperature ? "sent" : '<span class="badge warn">not sent</span>'}</td><td class="mono">${esc(m.token_limit_param)}</td>
        <td><input data-f="price_in_per_million" type="number" min="0" step="0.01" value="${m.price_in_per_million ?? ""}" style="width:90px"></td>
        <td><input data-f="price_out_per_million" type="number" min="0" step="0.01" value="${m.price_out_per_million ?? ""}" style="width:90px"></td></tr>`).join("")}</tbody></table></div>
    <div class="row" style="margin-top:12px"><div class="spacer"></div><button class="btn primary" id="s-save-models">Save models</button></div>
  </section>`;

  $("#s-save-endpoint").addEventListener("click", async () => {
    const body = { base_url: $("#s-url").value };
    if ($("#s-key").value) body.api_key = $("#s-key").value;
    try { await api("/api/settings", { body }); toast("Endpoint saved", "good"); await route(); }
    catch (err) { toast(err.message, "bad"); }
  });
  $("#s-save-models").addEventListener("click", async () => {
    const models = {};
    $$("tr[data-key]").forEach((tr) => {
      models[tr.dataset.key] = Object.fromEntries($$("input", tr).map((i) => [i.dataset.f, i.value]));
    });
    try { await api("/api/settings", { body: { models } }); toast("Models saved", "good"); await route(); }
    catch (err) { toast(err.message, "bad"); }
  });
}

route();
