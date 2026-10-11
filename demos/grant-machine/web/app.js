// Grant Machine presenter UI. Plain ES module, no build step, no CDN.

// ---------------------------------------------------------------------------
// small built-in Markdown renderer (escapes HTML first -- never innerHTML raw
// model output)
// ---------------------------------------------------------------------------

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const CITATION_RE = /\[\[([A-Za-z0-9_-]+)(?:#([A-Za-z0-9_-]+))?\]\]/g;

function renderMarkdown(raw) {
  const escaped = escapeHtml(raw || "");
  const lines = escaped.split("\n");
  const out = [];
  let inList = false;
  let table = null; // rows of cells while inside a pipe table
  const flushTable = () => {
    if (!table) return;
    const [head, ...body] = table;
    const cells = (row, tag) => row.map((c) => `<${tag}>${inline(c)}</${tag}>`).join("");
    out.push(`<div class="md-table-wrap"><table class="md-table"><thead><tr>${cells(head, "th")}</tr></thead>` +
      `<tbody>${body.map((r) => `<tr>${cells(r, "td")}</tr>`).join("")}</tbody></table></div>`);
    table = null;
  };
  for (const line of lines) {
    // Pipe tables: the loader turns every CSV (budgets, outcomes data) into one, and
    // shown as paragraphs they were rows of bars nobody could read.
    const row = line.match(/^\s*\|(.*)\|\s*$/);
    if (row) {
      if (inList) { out.push("</ul>"); inList = false; }
      const cells = row[1].split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-{3,}:?$/.test(c))) continue; // the header separator
      (table = table || []).push(cells);
      continue;
    }
    flushTable();
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      if (inList) { out.push("</ul>"); inList = false; }
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const item = line.match(/^[-*]\s+(.*)$/);
    if (item) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(item[1])}</li>`);
      continue;
    }
    if (inList) { out.push("</ul>"); inList = false; }
    if (line.trim() === "") {
      out.push("");
    } else {
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  if (inList) out.push("</ul>");
  flushTable();
  return out.join("\n");
}

function inline(escapedText) {
  let text = escapedText.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
  text = text.replace(CITATION_RE, (m, docId, chunkId) => {
    const label = chunkId ? `${docId}#${chunkId}` : docId;
    return `<span class="citation" data-doc-id="${docId}" data-chunk-id="${chunkId || ""}" tabindex="0">[[${label}]]</span>`;
  });
  return text;
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const state = {
  runId: null,
  mode: "draft",
  discoverSessionId: null,
  draftSessionId: null,
  gradeSessionId: null,
  samples: [],
  scenarios: [],
  scenario: null,
  opportunities: [],
  egress: [],
  scope: "quick",
  eventSource: null,
  compliance: new Map(),
  issues: new Map(),
  drafts: new Map(), // section_id -> {version, text, citations, word_count}
  scoreHistory: new Map(), // section_id -> [{version, mean_score, passed, criteria, must_fix}]
  budget: null,
  tally: { searches: 0, docsRead: new Set(), sectionsDrafted: new Set(), versionsBySection: new Map() },
  showReasoning: true,
  runStartedAt: null,
  isReplay: false,
  traceAutoScroll: true,
  // Whether THIS session has institution documents, per mode. Not the same as having a
  // session id: a Draft session may hold only a solicitation, which does not displace
  // the scenario's vault.
  hasDocuments: { draft: false, discover: false, grade: false },
  hasSolicitation: false, // Draft: an uploaded solicitation is in the session
  docs: null, // /api/scenarios/{id}/documents for the selected institution
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ---------------------------------------------------------------------------
// config + network
// ---------------------------------------------------------------------------

async function loadConfig() {
  try {
    const resp = await fetch("/api/config/public");
    const cfg = await resp.json();
    $("#device-label").textContent = cfg.device_label || "—";
    $("#model-name").textContent = cfg.model || "—";
    if (!cfg.pricing_configured) {
      $("#meter-cost").title = "Not configured in config/default.yaml";
    }
    return cfg;
  } catch (e) {
    return null;
  }
}

async function pollNetwork() {
  try {
    const resp = await fetch("/api/network");
    const net = await resp.json();
    applyNetworkStatus(net);
  } catch (e) {
    // network pill shows "unknown" state, no fabricated value
  }
}

function applyNetworkStatus(net) {
  const internetPill = $("#pill-internet");
  const internetText = $("#pill-internet-text");
  if (net.internet === "online") {
    internetPill.dataset.state = "online";
    internetText.textContent = "Internet: ONLINE";
  } else if (net.internet === "offline") {
    internetPill.dataset.state = "offline";
    internetText.textContent = "Internet: OFFLINE";
  }

  const modelPill = $("#pill-model");
  const modelText = $("#pill-model-text");
  const cls = net.llm_endpoint_class;
  if (cls === "loopback" || cls === "private") {
    modelPill.dataset.state = "online";
    modelText.textContent = "Model: running locally (" + cls + ")";
    $("#endpoint-banner").hidden = true;
  } else if (cls === "public") {
    modelPill.dataset.state = "offline";
    modelText.textContent = "Model: endpoint is public";
    $("#endpoint-banner").hidden = false;
  }

  // The old "App connections to the internet: 0" pill was removed when Discover made
  // that counter meaningless — this line outlived it and threw a TypeError on every
  // network_status event, which fires on a timer for the whole run.
  //
  // The live outbound count now comes from the egress audit (updateEgressPill), which
  // reports what this app actually sent rather than what psutil can see. The socket
  // count still matters for one thing though: connections this app did NOT make
  // through the guard. If it exceeds what we recorded, something bypassed the
  // boundary, and that is worth saying out loud rather than hiding.
  const observed = net.app_public_connections ?? 0;
  const recorded = state.egress.filter((r) => r.allowed).length;
  const pill = $("#pill-data");
  if (observed > 0 && recorded === 0) {
    pill.dataset.state = "warn";
    $("#pill-data-text").textContent =
      `Your documents: on this device — but ${observed} unaudited connection(s) seen`;
  } else {
    pill.dataset.state = "ok";
    $("#pill-data-text").textContent = "Your documents: on this device only";
  }
}

// ---------------------------------------------------------------------------
// meters
// ---------------------------------------------------------------------------

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function applyMetrics(data) {
  $("#meter-elapsed-value").textContent = formatElapsed(data.elapsed_s || 0);

  // Tokens and tok/s are rendered by the shared Demo Center chrome now; these
  // two meters were removed from index.html to stop two token counters sitting
  // one above the other with different totals. The metrics event still carries
  // the numbers -- /api/metrics reads them for the chrome.

  const cost = data.cost_equivalent;
  if (cost && cost.configured) {
    $("#meter-cost-value").textContent = `$${cost.usd.toFixed(2)}`;
    $("#meter-cost-label").textContent = cost.label || "";
  } else {
    $("#meter-cost-value").textContent = "—";
    $("#meter-cost-label").textContent = "";
  }

  const baseline = data.human_baseline;
  if (baseline && baseline.configured) {
    $("#meter-baseline-value").textContent = `${baseline.minutes} min`;
  } else {
    $("#meter-baseline-value").textContent = "—";
  }
}

// ---------------------------------------------------------------------------
// trace feed
// ---------------------------------------------------------------------------

function appendStepCard(kindLabel, kindClass, text, reasoningText) {
  const feed = $("#trace-feed");
  const card = document.createElement("div");
  card.className = `step-card ${kindClass || ""}`;
  const kindEl = document.createElement("div");
  kindEl.className = "kind";
  kindEl.textContent = kindLabel;
  card.appendChild(kindEl);
  const textEl = document.createElement("div");
  textEl.textContent = text;
  card.appendChild(textEl);
  if (reasoningText) {
    const r = document.createElement("div");
    r.className = "reasoning";
    r.textContent = reasoningText;
    r.hidden = !state.showReasoning;
    card.appendChild(r);
  }
  feed.appendChild(card);
  if (state.traceAutoScroll) {
    stickTraceToBottom();
  } else {
    $("#jump-to-live").hidden = false;
  }
  return card;
}

// scrollIntoView() scrolls EVERY scrollable ancestor, the window included. With the
// page itself scrolling (see styles.css), that dragged the viewport back down to the
// newest card every time one arrived, so nothing above it -- the token counter, the run
// controls -- could be read while the agent was working. Scroll the trace column and
// nothing else.
function stickTraceToBottom() {
  const trace = $("#trace");
  trace.scrollTop = trace.scrollHeight;
}

let pendingReasoningCard = null;
let pendingReasoningText = "";

function handleThinkingDelta(e) {
  const data = JSON.parse(e.data);
  pendingReasoningText += data.text;
  if (!pendingReasoningCard) {
    pendingReasoningCard = appendStepCard("thinking", "kind-thinking", "Reasoning…", pendingReasoningText);
  } else {
    const r = pendingReasoningCard.querySelector(".reasoning");
    if (r) r.textContent = pendingReasoningText;
    // A reasoning card grows for as long as the model thinks. Following it is only
    // right if the reader has not scrolled away.
    if (state.traceAutoScroll) stickTraceToBottom();
  }
}

function resetReasoningAccumulator() {
  pendingReasoningCard = null;
  pendingReasoningText = "";
}

// ---------------------------------------------------------------------------
// compliance / issues / budget / drafts / scores rendering
// ---------------------------------------------------------------------------

function renderCompliance() {
  const tbody = $("#compliance-table tbody");
  tbody.innerHTML = "";
  const rows = Array.from(state.compliance.entries());
  $("#compliance-empty").hidden = rows.length > 0;
  for (const [reqId, entry] of rows) {
    const tr = document.createElement("tr");
    const statusClass = `status-${entry.status}`;
    tr.innerHTML = `
      <td>${escapeHtml(reqId)}</td>
      <td><span class="status-pill ${statusClass}">${escapeHtml(entry.status)}</span></td>
      <td>${(entry.evidence || []).map((d) => `<span class="evidence-chip" data-doc-id="${escapeHtml(d)}">${escapeHtml(d)}</span>`).join(" ")}</td>
      <td>${escapeHtml(entry.note || "")}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderIssues() {
  const list = $("#issues-list");
  list.innerHTML = "";
  const issues = Array.from(state.issues.values());
  $("#issues-empty").hidden = issues.length > 0;
  const badge = $("#issues-badge");
  badge.textContent = String(issues.length);
  badge.hidden = issues.length === 0;
  for (const issue of issues) {
    const card = document.createElement("div");
    card.className = `issue-card severity-${issue.severity}`;
    card.innerHTML = `
      <div class="severity-label">${escapeHtml(issue.severity)}</div>
      <strong>${escapeHtml(issue.title)}</strong>
      <p>${escapeHtml(issue.detail)}</p>
      <div>${(issue.evidence || []).map((d) => `<span class="evidence-chip" data-doc-id="${escapeHtml(d)}">${escapeHtml(d)}</span>`).join(" ")}</div>
    `;
    list.appendChild(card);
  }
}

function renderBudget() {
  const el = $("#budget-summary");
  if (!state.budget) {
    $("#budget-empty").hidden = false;
    el.innerHTML = "";
    return;
  }
  $("#budget-empty").hidden = true;
  const b = state.budget;
  const over = b.over_by_usd > 0;
  el.innerHTML = `
    <table>
      <tr><th>Total</th><td>$${b.total_usd.toLocaleString()}</td></tr>
      <tr><th>Cap</th><td>$${b.cap_usd.toLocaleString()}</td></tr>
      <tr><th>Over by</th><td style="color:${over ? "var(--danger)" : "var(--ok)"}">$${b.over_by_usd.toLocaleString()}</td></tr>
      <tr><th>Fringe (budget)</th><td>${b.fringe_rate_budget != null ? (b.fringe_rate_budget * 100).toFixed(1) + "%" : "—"}</td></tr>
      <tr><th>Fringe (agreement)</th><td>${b.fringe_rate_agreement != null ? (b.fringe_rate_agreement * 100).toFixed(1) + "%" : "—"}</td></tr>
    </table>
  `;
}

function renderDrafts() {
  const list = $("#drafts-list");
  list.innerHTML = "";
  const drafts = Array.from(state.drafts.entries());
  $("#drafts-empty").hidden = drafts.length > 0;
  for (const [sectionId, draft] of drafts) {
    const block = document.createElement("div");
    block.className = "draft-block";
    block.innerHTML = `<h3>${escapeHtml(sectionId)} (v${draft.version}, ${draft.word_count || 0} words)</h3><div class="draft-text">${renderMarkdown(draft.text)}</div>`;
    list.appendChild(block);
  }
}

function renderScores() {
  const list = $("#scores-list");
  list.innerHTML = "";
  const entries = Array.from(state.scoreHistory.entries());
  $("#scores-empty").hidden = entries.length > 0;
  for (const [sectionId, versions] of entries) {
    const block = document.createElement("div");
    block.className = "draft-block";
    let html = `<h3>${escapeHtml(sectionId)}</h3>`;
    for (const v of versions) {
      html += `<div class="score-bar-row"><span>v${v.version}</span><div class="score-bar-track"><div class="score-bar-fill" style="width:${(v.mean_score / 5) * 100}%"></div></div><span>${v.mean_score.toFixed(1)}/5 ${v.passed ? "✓" : ""}</span></div>`;
    }
    block.innerHTML = html;
    list.appendChild(block);
  }
}

// ---------------------------------------------------------------------------
// SSE event handlers
// ---------------------------------------------------------------------------

async function refreshComplianceFromState(runId) {
  try {
    const resp = await fetch(`/api/runs/${runId}/state`);
    const body = await resp.json();
    state.compliance.clear();
    for (const [reqId, entry] of Object.entries(body.compliance || {})) {
      state.compliance.set(reqId, { status: entry.status, evidence: entry.evidence, note: entry.note });
    }
    renderCompliance();
  } catch (e) {
    // leave whatever partial data is already rendered rather than blank the table
  }
}

function connectEvents(runId) {
  const es = new EventSource(`/api/runs/${runId}/events`);
  state.eventSource = es;

  es.addEventListener("run_started", (e) => {
    const data = JSON.parse(e.data);
    state.isReplay = data.mode === "replay";
    $("#replay-banner").hidden = !state.isReplay;
    $("#replay-date").textContent = data.started_at || "—";
    $("#replay-device").textContent = data.device_label || "—";
    appendStepCard("run", "kind-run", `Run started (${MODE_LABELS[data.mode] || data.mode} · ${data.scope})`);
  });

  es.addEventListener("step_started", () => {
    resetReasoningAccumulator();
  });

  es.addEventListener("thinking_delta", handleThinkingDelta);

  es.addEventListener("tool_call", (e) => {
    const data = JSON.parse(e.data);
    appendStepCard(data.name, "kind-tool", data.display || `Calling ${data.name}`);
    if (data.name === "search_vault") state.tally.searches += 1;
  });

  es.addEventListener("tool_result", (e) => {
    const data = JSON.parse(e.data);
    if (data.name === "read_document" && data.result && data.result.doc_id) {
      state.tally.docsRead.add(data.result.doc_id);
    }
    if (data.name === "check_budget" && data.ok) {
      state.budget = data.result;
      renderBudget();
    }
  });

  es.addEventListener("compliance_update", async (e) => {
    const data = JSON.parse(e.data);
    if (data.requirement_id === "*") {
      // Bulk-init marker: get_requirements just seeded every requirement as
      // "missing" server-side. That per-row detail isn't in this event, so
      // pull the authoritative snapshot from run state instead of discarding it.
      await refreshComplianceFromState(runId);
      return;
    }
    state.compliance.set(data.requirement_id, { status: data.status, evidence: data.evidence, note: data.note });
    renderCompliance();
  });

  es.addEventListener("issue_flagged", (e) => {
    const data = JSON.parse(e.data).issue;
    state.issues.set(data.id, data);
    renderIssues();
    showIssueToast(data);
  });

  es.addEventListener("draft_started", (e) => {
    const data = JSON.parse(e.data);
    state.drafts.set(data.section_id, { version: data.version, text: "", citations: [], word_count: 0 });
    state.tally.sectionsDrafted.add(data.section_id);
    const versions = state.tally.versionsBySection.get(data.section_id) || 0;
    state.tally.versionsBySection.set(data.section_id, versions + 1);
    appendStepCard("draft", "kind-tool", `Drafting ${data.section_id} (v${data.version})`);
  });

  es.addEventListener("draft_delta", (e) => {
    const data = JSON.parse(e.data);
    const d = state.drafts.get(data.section_id);
    if (d) {
      d.text += data.delta;
      renderDrafts();
    }
  });

  es.addEventListener("draft_completed", (e) => {
    const data = JSON.parse(e.data);
    const d = state.drafts.get(data.section_id);
    if (d) {
      d.word_count = data.word_count;
      d.citations = data.citations;
      renderDrafts();
    }
  });

  es.addEventListener("section_scored", (e) => {
    const data = JSON.parse(e.data);
    const list = state.scoreHistory.get(data.section_id) || [];
    list.push({ version: data.version, mean_score: data.mean_score, passed: data.passed, criteria: data.criteria, must_fix: data.must_fix });
    state.scoreHistory.set(data.section_id, list);
    renderScores();
    appendStepCard("review", "kind-review", `${data.section_id} v${data.version}: ${data.mean_score.toFixed(1)} / 5 ${data.passed ? "(passed)" : "(revise)"}`);
  });

  es.addEventListener("citations_checked", (e) => {
    const data = JSON.parse(e.data);
    // Say how many citations were actually checked. "Citations OK" over a draft where
    // nothing parsed is what let a vacuous pass go unnoticed on screen.
    const n = data.citation_count ?? 0;
    const counted = `${n} citation${n === 1 ? "" : "s"}`;
    let label;
    if (data.passed) {
      label = `Citations OK — ${counted} verified (v${data.version})`;
    } else if ((data.malformed_citations || []).length) {
      label = `Citation syntax not readable — ${data.malformed_citations.length} marker(s) (v${data.version})`;
    } else if (n === 0) {
      label = `No citations to verify (v${data.version})`;
    } else {
      label = `Citations — issues found in ${counted} (v${data.version})`;
    }
    appendStepCard("verify", data.passed ? "kind-tool" : "kind-issue", label);
  });

  es.addEventListener("metrics", (e) => applyMetrics(JSON.parse(e.data)));

  es.addEventListener("network_status", (e) => applyNetworkStatus(JSON.parse(e.data)));

  // -- intake ---------------------------------------------------------------

  es.addEventListener("submission_loaded", (e) => {
    const d = JSON.parse(e.data);
    const list = (d.sections || []).map((s) => `${s.title} (${s.words}w)`).join(", ");
    appendStepCard("grade", "kind-tool", `Loaded submission ${d.filename} — ${(d.sections || []).length} section(s): ${list}`);
  });

  es.addEventListener("submission_failed", (e) => {
    const d = JSON.parse(e.data);
    appendStepCard("grade", "kind-issue", `Could not read ${d.filename}: ${d.detail}`);
  });

  es.addEventListener("intake_started", (e) => {
    const d = JSON.parse(e.data);
    $("#intake-progress").hidden = false;
    $("#intake-bar-fill").style.width = "2%";
    $("#intake-label").textContent = `Reading ${d.filename}…`;
    appendStepCard("intake", "kind-tool", `Reading uploaded solicitation: ${d.title || d.filename}`);
  });

  es.addEventListener("intake_progress", (e) => {
    const d = JSON.parse(e.data);
    const pct = d.total ? Math.round((d.index / d.total) * 100) : 0;
    $("#intake-bar-fill").style.width = `${Math.max(pct, 2)}%`;
    $("#intake-label").textContent = `Section ${d.index}/${d.total} — ${d.heading}`;
  });

  es.addEventListener("intake_complete", (e) => {
    const d = JSON.parse(e.data);
    $("#intake-bar-fill").style.width = "100%";
    const x = d.extraction || {};
    $("#intake-label").textContent = `${d.requirements} requirements from ${x.sections_scanned || "?"} sections`;
    const dropped = x.discarded_unverifiable_quote || 0;
    appendStepCard("intake", "kind-draft",
      `Extracted ${d.requirements} requirements from ${d.solicitation_id}` +
      (dropped ? ` — discarded ${dropped} the model could not quote from the document` : ""));
    appendStepCard("intake", "kind-tool",
      `Sections to draft: ${(d.sections || []).map((s) => s.title).join(", ")}`);
  });

  es.addEventListener("intake_failed", (e) => {
    const d = JSON.parse(e.data);
    $("#intake-progress").hidden = true;
    appendStepCard("intake", "kind-issue", `Could not read ${d.filename}: ${d.detail}`);
  });

  // -- discovery ------------------------------------------------------------

  es.addEventListener("egress", (e) => {
    const rec = JSON.parse(e.data);
    state.egress.push(rec);
    renderEgress();
    updateEgressPill();
    if (!rec.allowed) {
      const why = (rec.terms || []).filter((t) => !t.allowed).map((t) => t.reason)[0] || rec.reason;
      appendStepCard("egress", "kind-issue", `Refused to leave this device (${why})`);
    }
  });

  es.addEventListener("profile_drafted", (e) => {
    const d = JSON.parse(e.data);
    appendStepCard("profile", "kind-draft", `Institution profile v${d.version} — ${d.word_count} words`);
  });

  es.addEventListener("grants_searched", (e) => {
    const d = JSON.parse(e.data);
    if (d.error) {
      appendStepCard("search", "kind-issue", `Grant search refused: ${d.error}`);
      return;
    }
    appendStepCard("search", "kind-tool",
      `Federal catalogue: "${(d.terms_sent || []).join(", ")}" → ${d.hit_count} opportunities`);
  });

  es.addEventListener("web_searched", (e) => {
    const d = JSON.parse(e.data);
    appendStepCard("search", "kind-tool",
      d.error ? `Web search unavailable (${d.error})` : `Web search → ${(d.results || []).length} result(s)`);
  });

  es.addEventListener("page_fetched", (e) => {
    const d = JSON.parse(e.data);
    appendStepCard("fetch", "kind-tool", `Fetched ${d.title || d.url} — citable as [[${d.doc_id}]]`);
  });

  es.addEventListener("opportunity_detail", (e) => {
    const d = JSON.parse(e.data).opportunity || {};
    appendStepCard("opportunity", "kind-tool", `Read ${d.number}: ${d.title}`);
  });

  es.addEventListener("opportunity_proposed", (e) => {
    const opp = JSON.parse(e.data);
    state.opportunities = state.opportunities.filter((o) => o.opportunity_id !== opp.opportunity_id);
    state.opportunities.push(opp);
    renderOpportunities();
    appendStepCard("opportunity", "kind-draft", `Proposed ${opp.number} — ${opp.title}`);
  });

  es.addEventListener("error", (e) => {
    try {
      const data = JSON.parse(e.data);
      appendStepCard("error", "kind-issue", data.message || "error");
    } catch (_) {
      /* transport-level EventSource error, not an app error event */
    }
  });

  es.addEventListener("run_finished", (e) => {
    const data = JSON.parse(e.data);
    // A Find Funding run ends in the report a research office actually uses; every
    // other run ends in the generic summary.
    if (state.mode === "discover" || state.opportunities.length) openFundingReport(data);
    else showRunSummary(data);
    setRunning(false);
    es.close();
  });
}

function showIssueToast(issue) {
  const container = $("#toast-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = `[${issue.severity}] ${issue.title}`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 8000);
}

// -- the Find Funding report ------------------------------------------------------

const CONFIDENCE_RANK = { high: 0, medium: 1, low: 2 };

function safeHttps(url) {
  // Links come from the catalogue, not the model -- still, only ever render https.
  return typeof url === "string" && url.startsWith("https://") ? url : "";
}

function shortDate(dateText) {
  // "Mar 05, 2029 12:00:00 AM EST" -> "Mar 05, 2029". Cut the text, never convert it:
  // converting midnight Eastern to a Central browser moved a Dec 31 deadline to Dec 30.
  return String(dateText).replace(/\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*[A-Z]{2,4}$/i, "").trim();
}

function daysUntil(dateText) {
  const t = Date.parse(dateText);
  if (Number.isNaN(t)) return null;
  return Math.ceil((t - Date.now()) / 86400000);
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function formatCeiling(raw) {
  if (!raw || /^none$/i.test(raw)) return "";
  const n = Number(String(raw).replace(/[$,]/g, ""));
  return Number.isFinite(n) && n > 0 ? `ceiling $${n.toLocaleString()}` : `ceiling ${raw}`;
}

// A citation chip in the report names its source in words. The id stays on hover,
// and the chip still opens the document in the reader.
function citationLabel(docId) {
  const known = ((state.docs && state.docs.documents) || []).find((d) => d.doc_id === docId);
  if (known) return known.title;
  if (docId.startsWith("web-")) return "web page: " + docId.slice(4).replace(/-[0-9a-f]{6,}$/, "").replace(/-/g, ".");
  return docId.replace(/^[a-z]{2,5}-(?=[a-z])/, "").replace(/-/g, " ");
}

function tidyCitations(root) {
  root.querySelectorAll(".citation").forEach((c) => {
    c.title = c.textContent;
    c.textContent = citationLabel(c.dataset.docId);
  });
}

function openFundingReport(data) {
  if (data) state.lastFinish = data;
  renderFundingReport();
  $("#funding-report").hidden = false;
  $("#funding-report-close").focus();
}

function renderFundingReport() {
  const finish = state.lastFinish || {};
  const opps = [...state.opportunities].sort(
    (a, b) => (CONFIDENCE_RANK[a.confidence] ?? 3) - (CONFIDENCE_RANK[b.confidence] ?? 3));
  // Only name an institution we know this run was about. A replay does not say which
  // one it was, and the picker's default showed up here as the wrong institution.
  const sc = state.scenarios.find((x) => x.id === state.scenario);
  const who = state.isReplay || state.mode === "replay" ? "the institution's documents"
    : state.hasDocuments.discover ? "your uploaded documents" : (sc ? sc.label : "the institution's documents");
  $("#funding-report-sub").textContent =
    `${opps.length} opportunit${opps.length === 1 ? "y" : "ies"} recommended from ${state.tally.searches} ` +
    `catalogue search${state.tally.searches === 1 ? "" : "es"}, matched against ${who}. ` +
    "Every recommendation needs a person to confirm it before anyone applies.";

  const body = $("#funding-report-body");
  body.innerHTML = "";

  if (finish.summary) {
    const s = el("section", "report-summary");
    s.append(el("div", "doc-group-label", "Summary for the research office"));
    const p = el("div", "report-summary-text");
    p.innerHTML = renderMarkdown(finish.summary);
    tidyCitations(p);
    s.appendChild(p);
    body.appendChild(s);
  }

  if (!opps.length) {
    body.appendChild(el("p", "empty-state",
      "The agent found nothing it could honestly recommend. That is a result: a short, true list beats a padded one."));
    return;
  }

  // At-a-glance table, then one card per opportunity.
  const table = el("table", "md-table report-table");
  table.innerHTML = "<thead><tr><th>#</th><th>Opportunity</th><th>Agency</th><th>Closes</th><th>Fit</th><th>Reviewed</th></tr></thead>";
  const tbody = el("tbody");
  opps.forEach((o, i) => {
    const tr = el("tr");
    const days = daysUntil(o.close_date);
    tr.append(
      el("td", null, String(i + 1)),
      el("td", null, `${o.number || o.opportunity_id} — ${o.title || ""}`),
      el("td", null, o.agency || "—"),
      el("td", null, o.close_date ? `${shortDate(o.close_date)}${days != null ? ` (${days < 0 ? "closed" : `${days} days`})` : ""}` : "—"),
    );
    const fit = el("td"); fit.appendChild(el("span", `opp-confidence conf-${o.confidence}`, o.confidence));
    const rev = el("td");
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = !!(state.reviewed && state.reviewed[o.opportunity_id]);
    box.setAttribute("aria-label", `Mark ${o.number} reviewed`);
    box.addEventListener("change", () => { (state.reviewed = state.reviewed || {})[o.opportunity_id] = box.checked; });
    rev.appendChild(box);
    tr.append(fit, rev);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  const wrap = el("div", "md-table-wrap"); wrap.appendChild(table);
  body.appendChild(wrap);

  opps.forEach((o, i) => {
    const card = el("section", "report-opp");
    const head = el("div", "report-opp-head");
    head.append(el("span", `opp-confidence conf-${o.confidence}`, `${o.confidence} fit`),
                el("h3", null, `${i + 1}. ${o.title || o.number}`));
    card.appendChild(head);

    const facts = [o.number, o.agency, o.close_date ? `closes ${shortDate(o.close_date)}` : "",
                   formatCeiling(o.award_ceiling),
                   o.cost_sharing === true ? "cost sharing required" : o.cost_sharing === false ? "no cost sharing" : ""];
    card.appendChild(el("div", "opp-meta", facts.filter(Boolean).join(" · ")));

    if ((o.aligned_with || []).length) {
      const serves = el("div", "report-serves");
      serves.appendChild(el("span", "report-label", "Serves: "));
      for (const a of o.aligned_with) serves.appendChild(el("span", "report-chip", a));
      card.appendChild(serves);
    }

    card.appendChild(el("div", "report-label", "Why it fits"));
    const why = el("div", "opp-rationale");
    why.innerHTML = renderMarkdown(o.fit_rationale || ""); // escaped, citations become chips
    tidyCitations(why);
    card.appendChild(why);

    if ((o.concerns || []).length) {
      card.appendChild(el("div", "report-label report-label-warn", "A person must check before applying"));
      const ul = el("ul", "opp-concerns");
      for (const c of o.concerns) ul.appendChild(el("li", null, c));
      card.appendChild(ul);
    }

    const links = el("div", "report-links");
    const add = (href, label) => {
      const url = safeHttps(href);
      if (!url) return;
      const a = el("a", "btn report-link", label);
      a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer";
      links.appendChild(a);
    };
    add(o.url, "Review on Grants.gov ↗");
    add(o.announcement_url, (o.announcement_url || "").includes("/searchguide/")
      ? "Find it in the NIH Guide ↗" : "Read the full announcement ↗");
    if (!links.childElementCount) links.appendChild(el("span", "opp-meta", "No link was returned by the catalogue for this one."));
    card.appendChild(links);
    body.appendChild(card);
  });
}

function showRunSummary(data) {
  const searches = state.tally.searches;
  const docs = state.tally.docsRead.size;
  const sections = state.tally.sectionsDrafted.size;
  const versions = Array.from(state.tally.versionsBySection.values()).reduce((a, b) => a + b, 0);
  const issues = state.issues.size;
  const elapsed = formatElapsed(data.elapsed_s || 0);
  const text = `In ${elapsed} the agent checked ${state.compliance.size} requirement(s), ran ${searches} search(es), read ${docs} document(s), drafted ${sections} section(s) (${versions} version(s)), and flagged ${issues} issue(s). Status: ${data.status}.`;
  $("#run-summary-text").textContent = text;
  $("#run-summary-overlay").hidden = false;
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

function setRunning(running) {
  $("#btn-start").disabled = running;
  $("#btn-stop").disabled = !running;
}

async function startRun() {
  resetRunState();
  const mode = $("#mode-select").value;
  const scope = $("#scope-select").value;
  const resp = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode,
      scope,
      scenario: state.scenario,
      show_thinking: state.showReasoning,
      // Find Funding reads the documents uploaded into this session, so the run has to
      // be told which session that is.
      session_id: mode === "discover" ? state.discoverSessionId
                : mode === "draft" ? state.draftSessionId
                : mode === "grade" ? state.gradeSessionId
                : undefined,
      // Draft says which uploads it wants, so a file uploaded earlier cannot override
      // "this institution's" after the presenter switches back to it.
      ...(mode === "draft" ? {
        use_uploaded_solicitation: draftSource("sol") === "upload",
        use_uploaded_documents: draftSource("docs") === "upload",
      } : {}),
    }),
  });
  if (resp.status === 409) {
    const body = await resp.json();
    appendStepCard("error", "kind-issue", `A run is already active: ${body.detail.active_run_id}`);
    return;
  }
  const body = await resp.json();
  state.runId = body.run_id;
  setRunning(true);
  connectEvents(state.runId);
}

async function playReplay() {
  resetRunState();
  const name = $("#replay-select").value;
  const speed = $("#replay-speed").value;
  const resp = await fetch(`/api/replays/${name}/play`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speed: Number(speed) }),
  });
  const body = await resp.json();
  state.runId = body.run_id;
  setRunning(true);
  connectEvents(state.runId);
}

function resetRunState() {
  state.compliance.clear();
  state.issues.clear();
  state.drafts.clear();
  state.scoreHistory.clear();
  state.opportunities = [];
  state.lastFinish = null;
  state.reviewed = {};
  state.egress = [];
  state.budget = null;
  state.tally = { searches: 0, docsRead: new Set(), sectionsDrafted: new Set(), versionsBySection: new Map() };
  resetReasoningAccumulator();
  $("#trace-feed").innerHTML = "";
  $("#replay-banner").hidden = true;
  renderCompliance();
  renderIssues();
  renderBudget();
  renderDrafts();
  renderScores();
  renderOpportunities();
  renderEgress();
}

async function stopRun() {
  if (!state.runId) return;
  await fetch(`/api/runs/${state.runId}/cancel`, { method: "POST" });
  setRunning(false);
}

async function resetAll(skipConfirm) {
  if (!skipConfirm && !window.confirm("Reset the current run and clear all session data?")) return;
  await fetch("/api/reset", { method: "POST" });
  resetRunState();
  // /api/reset deletes runtime/sessions, so the id we were holding is now dangling.
  state.discoverSessionId = null;
  state.draftSessionId = null;
  state.gradeSessionId = null;
  state.hasDocuments = { draft: false, discover: false, grade: false };
  state.hasSolicitation = false;
  describeScenario();
  $("#grade-upload-status").textContent = "";
  $("#grade-sol-status").textContent = "";
  $$("[data-sample-status]").forEach((el) => { el.textContent = ""; });
  $$("[data-profile-status]").forEach((el) => { el.textContent = ""; });
  $$("[data-profile]").forEach((el) => { el.value = ""; });
  $("#discover-upload-status").textContent = "";
  $("#draft-docs-status").textContent = "";
  $("#upload-status").textContent = "";
  $("#intake-progress").hidden = true;
  state.runId = null;
  setRunning(false);
  $("#run-summary-overlay").hidden = true;
}

function switchTab(tabName) {
  $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === tabName)));
  $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `panel-${tabName}`; });
}

async function loadReplayList() {
  const resp = await fetch("/api/replays");
  const body = await resp.json();
  const select = $("#replay-select");
  select.innerHTML = "";
  for (const r of body.replays) {
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = r.name;
    select.appendChild(opt);
  }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function init() {
  loadConfig();
  pollNetwork();
  setInterval(pollNetwork, 5000);

  $("#mode-select").addEventListener("change", (e) => {
    state.mode = e.target.value;
    $("#draft-group").hidden = state.mode !== "draft";
    $("#discover-group").hidden = state.mode !== "discover";
    $("#grade-group").hidden = state.mode !== "grade";
    $("#replay-group").hidden = state.mode !== "replay";
    // Neither Find Funding nor Grade has sections to scope: one searches a catalogue,
    // the other reads whatever sections the submission actually has.
    $("#scope-group").hidden = ["replay", "discover", "grade"].includes(state.mode);
    $("#scenario-group").hidden = state.mode === "replay";
    describeScenario();
    // Start's own click handler already dispatches to playReplay() vs startRun()
    // based on state.mode, so it stays visible in every mode.
    if (state.mode === "replay") loadReplayList();
  });

  $("#btn-start").addEventListener("click", () => {
    if (state.mode === "replay") playReplay();
    else startRun();
  });
  $("#btn-stop").addEventListener("click", stopRun);
  $("#btn-reset").addEventListener("click", () => resetAll(false));
  $("#btn-summary-reset").addEventListener("click", () => resetAll(true));
  $("#btn-summary-close").addEventListener("click", () => { $("#run-summary-overlay").hidden = true; });
  $("#run-summary-overlay").addEventListener("click", (e) => {
    if (e.target.id === "run-summary-overlay") $("#run-summary-overlay").hidden = true;
  });

  $("#scenario-select").addEventListener("change", (e) => {
    state.scenario = e.target.value;
    describeScenario();
    loadScenarioDocuments();
  });

  // Draft's two source choices: each reveals its upload controls only when chosen.
  $$('input[name="draft-sol"], input[name="draft-docs"]').forEach((r) => r.addEventListener("change", () => {
    $("#draft-sol-upload").hidden = draftSource("sol") !== "upload";
    $("#draft-docs-upload").hidden = draftSource("docs") !== "upload";
    describeScenario();
  }));

  $("#btn-open-brief").addEventListener("click", openBrief);
  $("#brief-close").addEventListener("click", closeBrief);
  $("#brief-overlay").addEventListener("click", (e) => {
    if (e.target.id === "brief-overlay") closeBrief();
  });
  // Two copies of one toggle: the panel's sits behind the brief once it is open.
  let notesOn = false;
  try { notesOn = localStorage.getItem("gm-presenter-notes") === "1"; } catch (_) { /* blocked storage: notes start off */ }
  $("#presenter-notes").checked = $("#brief-presenter-notes").checked = notesOn;
  const setPresenterNotes = (on) => {
    $("#presenter-notes").checked = $("#brief-presenter-notes").checked = on;
    try { localStorage.setItem("gm-presenter-notes", on ? "1" : "0"); } catch (_) { /* per-viewer nicety only */ }
    renderScenarioDocuments();
    if (!$("#brief-overlay").hidden) renderBrief();
  };
  $("#presenter-notes").addEventListener("change", (e) => setPresenterNotes(e.target.checked));
  $("#brief-presenter-notes").addEventListener("change", (e) => setPresenterNotes(e.target.checked));
  $("#scenario-docs-list").addEventListener("click", (e) => {
    if (e.target.closest("[data-open-overview]")) { openOverview(); return; }
    const row = e.target.closest("[data-open-doc]");
    if (row) openSourceDrawer(row.dataset.openDoc);
  });
  $("#brief-body").addEventListener("click", (e) => {
    const row = e.target.closest("[data-open-doc]");
    if (row) openSourceDrawer(row.dataset.openDoc);
  });
  showBuild();
  loadScenarioList();

  $$(".btn-load-sample").forEach((b) => b.addEventListener("click", () => loadSample(b.dataset.mode)));
  $$(".btn-save-profile").forEach((b) => b.addEventListener("click", () => saveProfile(b.dataset.mode)));
  loadSampleList();

  // The egress pill opens the audit panel: seeing what left is the point of it.
  $("#pill-egress").addEventListener("click", () => switchTab("egress"));

  $("#show-reasoning").addEventListener("change", (e) => {
    state.showReasoning = e.target.checked;
    $$(".reasoning").forEach((r) => { r.hidden = !state.showReasoning; });
  });

  $$(".tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));

  $("#trace").addEventListener("scroll", () => {
    const el = $("#trace");
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    state.traceAutoScroll = atBottom;
    $("#jump-to-live").hidden = atBottom;
  });
  $("#jump-to-live").addEventListener("click", () => {
    state.traceAutoScroll = true;
    $("#jump-to-live").hidden = true;
    stickTraceToBottom();
  });

  // citation / evidence chips open the source drawer
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".citation, .evidence-chip");
    if (chip) openSourceDrawer(chip.dataset.docId, chip.dataset.chunkId);
  });

  $("#source-drawer-close").addEventListener("click", () => { $("#source-drawer").hidden = true; });
  $("#funding-report-close").addEventListener("click", () => { $("#funding-report").hidden = true; });
  $("#funding-report").addEventListener("click", (e) => {
    if (e.target.id === "funding-report") $("#funding-report").hidden = true;
  });
  $("#btn-open-funding-report").addEventListener("click", () => openFundingReport());
  $("#btn-report-print").addEventListener("click", () => window.print());
  $("#btn-report-packet").addEventListener("click", () => {
    if (state.runId) window.open(`/api/runs/${state.runId}/packet.pdf`, "_blank");
  });
  $("#source-drawer").addEventListener("click", (e) => {
    if (e.target.id === "source-drawer") $("#source-drawer").hidden = true;
  });

  $("#btn-download-packet").addEventListener("click", () => {
    // .pdf, not .md: the markdown endpoint renders as unstyled plain text in a
    // browser tab, which is not something a grants director can file or email.
    if (state.runId) window.open(`/api/runs/${state.runId}/packet.pdf`, "_blank");
  });
  $("#btn-save-replay").addEventListener("click", async () => {
    if (!state.runId) return;
    const name = window.prompt("Replay name:", `run-${state.runId}`);
    if (!name) return;
    await fetch(`/api/runs/${state.runId}/save-replay`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  });

  // Draft: an uploaded solicitation
  const uploadZone = $("#upload-zone");
  const uploadInput = $("#draft-sol-input");
  uploadZone.addEventListener("click", () => uploadInput.click());
  uploadZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") uploadInput.click(); });
  uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  uploadInput.addEventListener("change", () => { if (uploadInput.files.length) uploadFile(uploadInput.files[0]); });

  // Find Funding takes several documents into ONE session corpus, so these upload
  // sequentially and carry the session id forward from the first response.
  const discoverZone = $("#discover-upload-zone");
  const discoverInput = $("#discover-upload");
  discoverZone.addEventListener("click", () => discoverInput.click());
  discoverZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") discoverInput.click();
  });
  discoverZone.addEventListener("dragover", (e) => { e.preventDefault(); discoverZone.classList.add("dragover"); });
  discoverZone.addEventListener("dragleave", () => discoverZone.classList.remove("dragover"));
  discoverZone.addEventListener("drop", (e) => {
    e.preventDefault();
    discoverZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadDiscoverFiles(e.dataTransfer.files);
  });
  discoverInput.addEventListener("change", () => {
    if (discoverInput.files.length) uploadDiscoverFiles(discoverInput.files);
  });

  // Grade takes the proposal being assessed, and optionally the solicitation to
  // assess it against.
  wireSingleUpload("#grade-upload-zone", "#grade-upload", "#grade-upload-status", "proposal", "gradeSessionId");
  wireSingleUpload("#grade-sol-zone", "#grade-solicitation-upload", "#grade-sol-status", "solicitation", "gradeSessionId");

  // Draft may also carry the institution's own evidence, so an uploaded solicitation
  // is not stuck being answered out of a shipped scenario's vault.
  const exDocsZone = $("#draft-docs-zone");
  const exDocsInput = $("#draft-docs-input");
  exDocsZone.addEventListener("click", () => exDocsInput.click());
  exDocsZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") exDocsInput.click();
  });
  exDocsZone.addEventListener("dragover", (e) => { e.preventDefault(); exDocsZone.classList.add("dragover"); });
  exDocsZone.addEventListener("dragleave", () => exDocsZone.classList.remove("dragover"));
  exDocsZone.addEventListener("drop", (e) => {
    e.preventDefault();
    exDocsZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadSessionDocuments(e.dataTransfer.files, "draft");
  });
  exDocsInput.addEventListener("change", () => {
    if (exDocsInput.files.length) uploadSessionDocuments(exDocsInput.files, "draft");
  });

  // theme toggle persisted per-viewer (best-effort; UI works fine if storage is unavailable)
  try {
    const saved = localStorage.getItem("gm-theme");
    if (saved) document.body.dataset.theme = saved;
  } catch (_) { /* private browsing or blocked storage: fall back to default theme */ }

  document.addEventListener("keydown", handleKeydown);
  $("#shortcut-help-close").addEventListener("click", () => { $("#shortcut-help").hidden = true; });

  renderCompliance();
  renderIssues();
  renderBudget();
  renderDrafts();
  renderScores();
  renderOpportunities();
  renderEgress();
}

function updateEgressPill() {
  const sent = state.egress.filter((r) => r.allowed).length;
  const blocked = state.egress.length - sent;
  const pill = $("#pill-egress");
  pill.dataset.state = blocked > 0 ? "warn" : "ok";
  $("#pill-egress-text").textContent =
    `Agent web access: ${sent} request${sent === 1 ? "" : "s"}` + (blocked ? ` · ${blocked} refused` : "");
}

function renderOpportunities() {
  $("#btn-open-funding-report").hidden = state.opportunities.length === 0 && !state.lastFinish;
  const list = $("#opps-list");
  const badge = $("#opps-badge");
  list.innerHTML = "";
  $("#opps-empty").hidden = state.opportunities.length > 0;
  badge.hidden = state.opportunities.length === 0;
  badge.textContent = String(state.opportunities.length);

  const rank = { high: 0, medium: 1, low: 2 };
  const sorted = [...state.opportunities].sort(
    (a, b) => (rank[a.confidence] ?? 3) - (rank[b.confidence] ?? 3));

  for (const opp of sorted) {
    const card = document.createElement("div");
    card.className = "opp-card";

    const head = document.createElement("div");
    head.className = "opp-head";
    const conf = document.createElement("span");
    conf.className = `opp-confidence conf-${opp.confidence}`;
    conf.textContent = opp.confidence;
    const num = document.createElement("strong");
    num.textContent = opp.number || opp.opportunity_id;
    head.append(conf, num);
    card.appendChild(head);

    const title = document.createElement("div");
    title.className = "opp-title";
    title.textContent = opp.title || "";
    card.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "opp-meta";
    meta.textContent = [opp.agency, opp.close_date ? `closes ${opp.close_date}` : ""]
      .filter(Boolean).join(" · ");
    card.appendChild(meta);

    // Model text, so it goes through the same escape-then-render path as every
    // other model output in this UI (see renderMarkdown).
    const why = document.createElement("div");
    why.className = "opp-rationale";
    why.innerHTML = renderMarkdown(opp.fit_rationale || "");
    card.appendChild(why);

    if ((opp.concerns || []).length) {
      const concerns = document.createElement("ul");
      concerns.className = "opp-concerns";
      for (const c of opp.concerns) {
        const li = document.createElement("li");
        li.textContent = c;
        concerns.appendChild(li);
      }
      const label = document.createElement("div");
      label.className = "opp-concerns-label";
      label.textContent = "A human must check:";
      card.append(label, concerns);
    }

    if (opp.url) {
      const link = document.createElement("div");
      link.className = "opp-url";
      link.textContent = opp.url;
      card.appendChild(link);
    }
    list.appendChild(card);
  }
}

function renderEgress() {
  const list = $("#egress-list");
  const badge = $("#egress-badge");
  list.innerHTML = "";
  $("#egress-empty").hidden = state.egress.length > 0;
  const blocked = state.egress.filter((r) => !r.allowed).length;
  badge.hidden = blocked === 0;
  badge.textContent = String(blocked);

  for (const rec of state.egress) {
    const row = document.createElement("div");
    row.className = `egress-row ${rec.allowed ? "egress-sent" : "egress-blocked"}`;

    const verdict = document.createElement("span");
    verdict.className = "egress-verdict";
    verdict.textContent = rec.allowed ? "SENT" : "REFUSED";

    const url = document.createElement("span");
    url.className = "egress-url";
    url.textContent = rec.url || "";

    const detail = document.createElement("div");
    detail.className = "egress-detail";
    const terms = (rec.terms || []).map((t) =>
      `${t.term}${t.allowed ? "" : ` (${t.reason})`}`).join(", ");
    detail.textContent = [
      terms ? `terms: ${terms}` : "",
      rec.allowed ? `${rec.bytes_in || 0} bytes in` : rec.reason,
    ].filter(Boolean).join(" · ");

    const head = document.createElement("div");
    head.className = "egress-head";
    head.append(verdict, url);
    row.append(head, detail);
    list.appendChild(row);
  }
}

async function uploadFile(file) {
  const status = $("#upload-status");
  status.textContent = `Uploading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    // kind=solicitation: this upload is the call for proposals to extract
    // requirements from, not the institution's evidence. It joins the Draft session
    // so evidence uploaded before or after it is not stranded in another one.
    const sid = state.draftSessionId;
    const qs = "kind=solicitation" + (sid ? `&session_id=${encodeURIComponent(sid)}` : "");
    const resp = await fetch(`/api/explore/upload?${qs}`, { method: "POST", body: form });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      status.textContent = `Upload failed: ${body.detail || resp.status}`;
      return;
    }
    const body = await resp.json();
    state.draftSessionId = body.session_id;
    state.hasSolicitation = true;
    describeScenario();
    status.textContent = `${body.filename} ready — Start to extract its requirements.`;
  } catch (e) {
    status.textContent = "Upload failed.";
  }
}

function uploadDiscoverFiles(fileList) {
  return uploadSessionDocuments(fileList, "discover");
}

function wireSingleUpload(zoneSel, inputSel, statusSel, kind, sessionKey) {
  const zone = $(zoneSel);
  const input = $(inputSel);
  const upload = async (file) => {
    const status = $(statusSel);
    status.textContent = `Uploading ${file.name}…`;
    const form = new FormData();
    form.append("file", file);
    const sid = state[sessionKey];
    const qs = `kind=${kind}` + (sid ? `&session_id=${encodeURIComponent(sid)}` : "");
    try {
      const resp = await fetch(`/api/explore/upload?${qs}`, { method: "POST", body: form });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        status.textContent = `Upload failed: ${body.detail || resp.status}`;
        return;
      }
      const body = await resp.json();
      state[sessionKey] = body.session_id;
      if (kind === "documents") state.hasDocuments[state.mode] = true;
      describeScenario(); // the note says which corpus a run will use; uploads change that
      status.textContent = `${body.filename} ready.`;
    } catch (e) {
      status.textContent = "Upload failed.";
    }
  };
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) upload(input.files[0]); });
}

// -- samples and typed profiles -----------------------------------------------

// Each mode owns the session its uploads, samples and profile all land in, so a
// sample and a typed profile combine rather than each starting a fresh session.
const SESSION_KEY_BY_MODE = {
  draft: "draftSessionId",
  discover: "discoverSessionId",
  grade: "gradeSessionId",
};

// Draft splits one old sample list in two: solicitations go with "upload your own
// solicitation", institution documents with "upload your own evidence". The manifest
// still tags them "explore", the mode Draft replaced.
const hasMode = (s, ...modes) => (s.modes || []).some((m) => modes.includes(m));
const SAMPLE_PICKERS = {
  "draft-sol": { mode: "draft", accepts: (s) => hasMode(s, "draft", "explore") && (s.kinds || []).includes("solicitation") },
  "draft-docs": { mode: "draft", accepts: (s) => hasMode(s, "draft", "explore") && !(s.kinds || []).includes("solicitation") },
  discover: { mode: "discover", accepts: (s) => hasMode(s, "discover") },
  grade: { mode: "grade", accepts: (s) => hasMode(s, "grade") },
};

// Runs report the resolved server-side mode; saved replays say "story" or "explore".
const MODE_LABELS = {
  draft: "Draft",
  story: "Draft",
  explore: "Draft · uploaded solicitation",
  discover: "Find Funding",
  grade: "Grade",
  replay: "Replay",
};

function draftSource(which) {
  const checked = document.querySelector(`input[name="draft-${which}"]:checked`);
  return checked ? checked.value : "scenario";
}

async function showBuild() {
  // Compare what this page was served with against what the server has now. They
  // differ only when a cached copy of app.js is in play, which is invisible otherwise.
  const el = $("#footer-build");
  if (!el) return;
  const loaded = (document.querySelector('script[src*="app.js"]')?.getAttribute("src") || "").split("v=")[1] || "unstamped";
  try {
    const { build } = await (await fetch("/api/build")).json();
    el.textContent = build === loaded ? `build ${build}` : `build ${loaded} — STALE (server has ${build}), reload`;
    if (build !== loaded) el.style.color = "var(--danger)";
  } catch (e) {
    el.textContent = `build ${loaded}`;
  }
}

async function loadScenarioList() {
  try {
    const body = await (await fetch("/api/scenarios")).json();
    state.scenarios = body.scenarios || [];
    state.scenario = state.scenario || body.default;
  } catch (e) {
    return; // no endpoint: the picker stays hidden and the default scenario is used
  }
  const select = $("#scenario-select");
  select.innerHTML = "";
  for (const sc of state.scenarios) {
    const opt = document.createElement("option");
    opt.value = sc.id;
    // A scenario with no requirements cannot be run, and silently offering it would
    // just produce a confusing empty compliance matrix.
    opt.textContent = sc.label + (sc.ready ? "" : " (not built)");
    opt.disabled = !sc.ready;
    opt.selected = sc.id === state.scenario;
    select.appendChild(opt);
  }
  describeScenario();
  loadScenarioDocuments();
}

// -- the institution's documents: panel and brief ------------------------------

async function loadScenarioDocuments() {
  const id = state.scenario;
  if (!id) return;
  try {
    const resp = await fetch(`/api/scenarios/${encodeURIComponent(id)}/documents`);
    if (!resp.ok) throw new Error(String(resp.status));
    const body = await resp.json();
    if (state.scenario !== id) return; // the presenter switched again while this loaded
    state.docs = body;
  } catch (e) {
    state.docs = null;
  }
  renderScenarioDocuments();
  if (!$("#brief-overlay").hidden) renderBrief();
}

function presenterNotesOn() {
  return $("#presenter-notes").checked;
}

// doc_id -> planted issues that cite it, for the flags presenter notes add.
function plantedByDoc() {
  const map = new Map();
  for (const issue of (state.docs && state.docs.planted_issues) || []) {
    for (const docId of issue.evidence || []) {
      if (!map.has(docId)) map.set(docId, []);
      map.get(docId).push(issue);
    }
  }
  return map;
}

function docRow(doc, flags) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "doc-row";
  row.dataset.openDoc = doc.doc_id;
  const title = document.createElement("span");
  title.className = "doc-title";
  title.textContent = doc.title;
  const meta = document.createElement("span");
  meta.className = "doc-meta";
  meta.textContent = [doc.doc_type.toUpperCase(), doc.date].filter(Boolean).join(" · ");
  row.append(title, meta);
  if (flags && flags.length) {
    const flag = document.createElement("span");
    flag.className = "doc-flag";
    flag.textContent = `⚑ ${flags.length}`;
    flag.title = flags.map((i) => `${i.code}: ${i.detail}`).join("\n");
    row.appendChild(flag);
  }
  return row;
}

function solicitationRow(sol) {
  return docRow({
    doc_id: "solicitation",
    title: sol.title,
    doc_type: sol.is_real ? "real" : "synthetic",
    date: sol.solicitation_id || "",
  });
}

// The scenario overview is not a vault document -- the agent never sees it -- so it
// opens in the reader from what the panel already holds rather than by fetching.
function overviewRow(sc) {
  const row = docRow({ doc_id: "", title: "Scenario overview", doc_type: "summary", date: "" });
  delete row.dataset.openDoc;
  row.dataset.openOverview = "1";
  row.classList.add("doc-row-overview");
  return row;
}

function openOverview() {
  const sc = state.docs && state.docs.scenario;
  if (!sc) return;
  $("#source-drawer-title").textContent = `${sc.label} (fictional) — overview`;
  $("#source-drawer-body").innerHTML = renderMarkdown(sc.summary);
  $("#source-drawer").hidden = false;
}

function renderScenarioDocuments() {
  const list = $("#scenario-docs-list");
  list.innerHTML = "";
  const d = state.docs;
  $("#scenario-docs-summary").textContent = d
    ? `Documents (${d.documents.length}${d.solicitation ? " + solicitation" : ""})`
    : "Documents";
  if (!d) {
    list.textContent = "Could not load this institution's documents.";
    return;
  }
  const flags = presenterNotesOn() ? plantedByDoc() : new Map();
  if (d.scenario.summary) {
    const h = document.createElement("div");
    h.className = "doc-group-label";
    h.textContent = "Scenario";
    list.append(h, overviewRow(d.scenario));
  }
  if (d.solicitation) {
    const h = document.createElement("div");
    h.className = "doc-group-label";
    h.textContent = "Solicitation";
    list.append(h, solicitationRow(d.solicitation));
  }
  for (const group of d.groups) {
    const h = document.createElement("div");
    h.className = "doc-group-label";
    h.textContent = group;
    list.appendChild(h);
    for (const doc of d.documents.filter((x) => x.group === group)) {
      list.appendChild(docRow(doc, flags.get(doc.doc_id)));
    }
  }
}

function openBrief() {
  renderBrief();
  $("#brief-overlay").hidden = false;
  $("#brief-close").focus();
}

function closeBrief() {
  $("#brief-overlay").hidden = true;
}

function renderBrief() {
  const d = state.docs;
  const body = $("#brief-body");
  body.innerHTML = "";
  if (!d) {
    $("#brief-title").textContent = "Institution brief";
    $("#brief-sub").textContent = "";
    body.textContent = "Could not load this institution's documents.";
    return;
  }
  const sc = d.scenario;
  $("#brief-title").textContent = `${sc.label} (fictional)`;
  $("#brief-sub").textContent = sc.type || "";

  if (sc.summary) {
    const about = document.createElement("section");
    about.className = "brief-overview";
    const label = document.createElement("div");
    label.className = "doc-group-label";
    label.textContent = "Scenario overview";
    const text = document.createElement("p");
    text.textContent = sc.summary;
    about.append(label, text);
    body.appendChild(about);
  }

  if (d.solicitation) {
    const sol = document.createElement("section");
    sol.className = "brief-solicitation";
    const label = document.createElement("div");
    label.className = "doc-group-label";
    label.textContent = "Applying to";
    const badge = document.createElement("span");
    badge.className = `brief-badge ${d.solicitation.is_real ? "badge-real" : "badge-synthetic"}`;
    badge.textContent = d.solicitation.is_real ? "Real, public solicitation" : "Synthetic, written for this scenario";
    sol.append(label, solicitationRow(d.solicitation), badge);
    body.appendChild(sol);
  }

  const notes = presenterNotesOn();
  const flags = notes ? plantedByDoc() : new Map();
  for (const group of d.groups) {
    const section = document.createElement("section");
    const h = document.createElement("div");
    h.className = "doc-group-label";
    h.textContent = group;
    const grid = document.createElement("div");
    grid.className = "brief-grid";
    for (const doc of d.documents.filter((x) => x.group === group)) {
      const card = docRow(doc, flags.get(doc.doc_id));
      card.classList.add("brief-doc");
      grid.appendChild(card);
    }
    section.append(h, grid);
    body.appendChild(section);
  }

  if (notes && d.planted_issues.length) {
    const section = document.createElement("section");
    section.className = "brief-planted";
    const h = document.createElement("div");
    h.className = "doc-group-label";
    h.textContent = `Planted issues — presenter only (${d.planted_issues.length})`;
    const ul = document.createElement("ul");
    const titles = new Map(d.documents.map((x) => [x.doc_id, x.title]));
    for (const issue of d.planted_issues) {
      const li = document.createElement("li");
      const code = document.createElement("strong");
      code.textContent = `${issue.code}`;
      const detail = document.createElement("span");
      detail.textContent = ` — ${issue.detail} `;
      li.append(code, detail);
      for (const docId of issue.evidence || []) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "brief-chip";
        chip.dataset.openDoc = docId;
        chip.textContent = titles.get(docId) || docId;
        li.appendChild(chip);
      }
      ul.appendChild(li);
    }
    section.append(h, ul);
    body.appendChild(section);
  }
}

function describeScenario() {
  const sc = state.scenarios.find((x) => x.id === state.scenario);
  if (!sc) return;
  // Name the agency only when the call really is theirs: leaving NSF named under a
  // synthetic solicitation attributed a document to an agency that did not write it.
  const footer = $("#footer-solicitation");
  if (footer) {
    const agency = sc.solicitation_agency || "the issuing agency";
    footer.textContent = sc.solicitation_is_real
      ? `Solicitation text: ${sc.solicitation_id} (U.S. Government work) · Not affiliated with or endorsed by ${agency}`
      : "Solicitation: synthetic, written for this scenario";
  }

  // Say what the choice governs in the mode the user is actually in. The picker was
  // reported as doing nothing, and in Discover and Grade with documents uploaded that
  // was close to true: the session's documents are the corpus, and the scenario only
  // supplies the requirements to work to.
  const uploaded = state.hasDocuments[state.mode];
  const base = `${sc.type} · ${sc.documents} documents · ` +
    (sc.solicitation_is_real ? "real solicitation" : "synthetic solicitation");
  const solKind = sc.solicitation_is_real ? "real" : "synthetic";
  $("#draft-sol-scenario-label").textContent = `This institution's (${solKind})`;
  $("#draft-docs-scenario-label").textContent = `This institution's ${sc.documents} documents`;
  let scopeNote;
  if (state.mode === "draft") {
    const sol = draftSource("sol") !== "upload" ? "its own solicitation"
      : state.hasSolicitation ? "your uploaded solicitation"
      : "its own solicitation (nothing uploaded yet)";
    const docs = draftSource("docs") !== "upload" ? "this institution's documents"
      : uploaded ? "your uploaded documents"
      : "this institution's documents (nothing uploaded yet)";
    scopeNote = `This run drafts to ${sol}, citing ${docs}.`;
  } else if (uploaded) {
    scopeNote = "Your uploaded documents are the corpus for this run; this institution supplies the solicitation only.";
  } else {
    scopeNote = "No documents uploaded yet, so this institution's vault is the corpus.";
  }
  $("#scenario-note").textContent = `${base} — ${scopeNote}`;
}

async function loadSampleList() {
  let samples = [];
  try {
    const resp = await fetch("/api/samples");
    samples = (await resp.json()).samples || [];
  } catch (e) {
    return; // no samples endpoint: the pickers stay empty and uploading still works
  }
  state.samples = samples;
  for (const [picker, spec] of Object.entries(SAMPLE_PICKERS)) {
    const select = document.getElementById(`sample-select-${picker}`);
    if (!select) continue;
    const forMode = samples.filter(spec.accepts);
    select.innerHTML = "";
    if (!forMode.length) {
      $(`[data-sample-mode="${picker}"]`).hidden = true;
      continue;
    }
    for (const s of forMode) {
      const opt = document.createElement("option");
      opt.value = s.id;
      // Say plainly which samples are real documents and which are invented.
      opt.textContent = `${s.label}${s.synthetic === false ? " (real)" : ""}`;
      opt.title = s.description || "";
      select.appendChild(opt);
    }
  }
}

async function loadSample(picker) {
  const select = document.getElementById(`sample-select-${picker}`);
  const status = $(`[data-sample-status="${picker}"]`);
  if (!select || !select.value) return;
  const mode = SAMPLE_PICKERS[picker].mode;
  const key = SESSION_KEY_BY_MODE[mode];
  const sid = state[key];
  status.textContent = "Loading…";
  try {
    const url = `/api/samples/${encodeURIComponent(select.value)}/load` +
                (sid ? `?session_id=${encodeURIComponent(sid)}` : "");
    const resp = await fetch(url, { method: "POST" });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      status.textContent = `Could not load: ${body.detail || resp.status}`;
      return;
    }
    const body = await resp.json();
    state[key] = body.session_id;
    if ((body.kinds || []).includes("documents")) state.hasDocuments[mode] = true;
    if (mode === "draft" && (body.kinds || []).includes("solicitation")) state.hasSolicitation = true;
    describeScenario();
    const chosen = (state.samples || []).find((s) => s.id === body.sample_id);
    status.textContent = `Loaded ${body.files.length} file(s) — ${chosen ? chosen.label : body.sample_id}. Press Start.`;
  } catch (e) {
    status.textContent = "Could not load the sample.";
  }
}

async function saveProfile(mode) {
  const form = document.getElementById(`profile-form-${mode}`);
  const status = $(`[data-profile-status="${mode}"]`);
  if (!form) return;
  const payload = {};
  for (const el of form.querySelectorAll("[data-profile]")) {
    payload[el.dataset.profile] = el.value.trim();
  }
  if (!Object.values(payload).some(Boolean)) {
    status.textContent = "Fill in at least one field.";
    return;
  }
  const key = SESSION_KEY_BY_MODE[mode];
  const sid = state[key];
  status.textContent = "Saving…";
  try {
    const url = "/api/session/profile" + (sid ? `?session_id=${encodeURIComponent(sid)}` : "");
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      status.textContent = `Could not save: ${body.detail || resp.status}`;
      return;
    }
    const body = await resp.json();
    state[key] = body.session_id;
    state.hasDocuments[mode] = true; // the profile is saved as a session document
    describeScenario();
    status.textContent = "Saved on this device — the agent can cite it as [[profile]].";
  } catch (e) {
    status.textContent = "Could not save the profile.";
  }
}

async function uploadSessionDocuments(fileList, mode) {
  const status = $(mode === "draft" ? "#draft-docs-status" : "#discover-upload-status");
  const sessionKey = SESSION_KEY_BY_MODE[mode];
  const files = Array.from(fileList);
  let uploaded = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    status.textContent = `Uploading ${file.name} (${i + 1}/${files.length})…`;
    const form = new FormData();
    form.append("file", file);
    // Only the first upload mints a session; the rest join it, or each document
    // would land in its own corpus and the agent would see only the last one.
    // kind=documents: the institution's evidence, which becomes the search corpus.
    const sid = state[sessionKey];
    const url = sid
      ? `/api/explore/upload?kind=documents&session_id=${encodeURIComponent(sid)}`
      : "/api/explore/upload?kind=documents";
    try {
      const resp = await fetch(url, { method: "POST", body: form });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        status.textContent = `${file.name} failed: ${body.detail || resp.status}`;
        return;
      }
      const body = await resp.json();
      state[sessionKey] = body.session_id;
      state.hasDocuments[mode] = true;
      describeScenario(); // the note says which corpus a run will use; uploads change that
      uploaded = body.files || uploaded.concat(file.name);
    } catch (e) {
      status.textContent = `${file.name} failed to upload.`;
      return;
    }
  }
  status.textContent = `${uploaded.length} document${uploaded.length === 1 ? "" : "s"} on this device: ${uploaded.join(", ")}`;
}

async function openSourceDrawer(docId, chunkId) {
  if (!docId) return;
  $("#source-drawer-title").textContent = chunkId ? `${docId}#${chunkId}` : docId;
  $("#source-drawer-body").textContent = "Loading…";
  $("#source-drawer").hidden = false;
  try {
    // The scenario and session say where to look first: a doc id is only a file name,
    // and it used to be looked up in Harrow Creek's vault whatever the run was for.
    const params = new URLSearchParams();
    if (chunkId) params.set("chunk_id", chunkId);
    if (state.scenario) params.set("scenario", state.scenario);
    const sid = state[SESSION_KEY_BY_MODE[state.mode]];
    if (sid) params.set("session_id", sid);
    const url = `/api/vault/${encodeURIComponent(docId)}?${params}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      $("#source-drawer-body").textContent = "Document not found.";
      return;
    }
    const body = await resp.json();
    if (body.title) $("#source-drawer-title").textContent = chunkId ? `${body.title} #${chunkId}` : body.title;
    $("#source-drawer-body").innerHTML = renderMarkdown(body.text);
  } catch (e) {
    $("#source-drawer-body").textContent = "Failed to load document.";
  }
}

function handleKeydown(e) {
  // Esc closes the innermost layer: the drawer opened from the brief, then the brief.
  if (e.key === "Escape") {
    if (!$("#source-drawer").hidden) $("#source-drawer").hidden = true;
    else if (!$("#funding-report").hidden) $("#funding-report").hidden = true;
    else if (!$("#brief-overlay").hidden) closeBrief();
    return;
  }
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
  switch (e.key.toLowerCase()) {
    case "s":
      if (!$("#btn-start").disabled) startRun();
      break;
    case "x":
      if (!$("#btn-stop").disabled) stopRun();
      break;
    case "r":
      resetAll(false);
      break;
    case "t":
      $("#show-reasoning").checked = !$("#show-reasoning").checked;
      $("#show-reasoning").dispatchEvent(new Event("change"));
      break;
    case "f":
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
      break;
    case "1": switchTab("compliance"); break;
    case "2": switchTab("issues"); break;
    case "3": switchTab("budget"); break;
    case "4": switchTab("drafts"); break;
    case "5": switchTab("scores"); break;
    case "?":
      $("#shortcut-help").hidden = !$("#shortcut-help").hidden;
      break;
  }
}

init();
